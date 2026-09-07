"""Evaluate PersonViT and every local ReID checkpoint on one AG protocol.

This script does not train or alter any model checkpoint.  It discovers
``best_model.pth`` files below the outputs directory, reconstructs the matching
source model from checkpoint keys/config, evaluates every model from scratch on
the requested AG-ReID.v2 query/gallery protocol, and writes CSV/Markdown tables.

Run from the repository root::

    python tools/compare_experiments.py

The default comparison includes the untouched PersonViT SSL backbone, followed
by every checkpoint that currently exists below ``outputs/``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

# Silence TensorFlow discovery messages before importing torch/tensorboard users.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from reid_advance.config import (
    BNNeckConfig,
    PersonViTFineTuneConfig,
    ProposedConfig,
    ProposedV2Config,
    ProposedV3Config,
    TransReIDConfig,
    UntransReIDConfig,
)
from reid_advance.identity import require_current_identity_scheme
from reid_advance.pipelines.bnneck import BNNeckModel
from reid_advance.pipelines.personvit_finetune import PersonViTFineTuneModel
from reid_advance.pipelines.proposed import LeWMReID
from reid_advance.pipelines.proposed_v2 import ProposedV2Model
from reid_advance.pipelines.proposed_v3 import ProposedV3Model
from reid_advance.pipelines.transreid import TransReIDSmall
from reid_advance.pipelines.untransreid import UntransReIDSmall, load_ssl_backbone
from reid_advance.runtime import build_eval_loaders, evaluate_model, resolve_device


@dataclass(frozen=True)
class Candidate:
    name: str
    checkpoint: Path | None
    kind: str = "auto"


class PersonViTBase(nn.Module):
    """Untouched PersonViT teacher backbone with normalized CLS retrieval."""

    def __init__(self, cfg: ProposedV2Config, checkpoint: str):
        super().__init__()
        self.encoder = timm.create_model(
            cfg.encoder_name,
            pretrained=False,
            num_classes=0,
            img_size=cfg.image_size,
        )
        load_ssl_backbone(self.encoder, checkpoint)

    def forward(self, images):
        return F.normalize(self.encoder(images), dim=1)


class LegacySSLModel(nn.Module):
    """Retrieval adapter for the archived DINOv2+VICReg checkpoint."""

    def __init__(self, cfg, full_state):
        super().__init__()
        image_size = cfg.image_size
        timm_size = image_size[0] if image_size[0] == image_size[1] else image_size
        self.encoder = timm.create_model(
            cfg.encoder_name,
            pretrained=False,
            num_classes=0,
            img_size=timm_size,
        )
        encoder_state = {
            key[len("encoder.") :]: value
            for key, value in full_state.items()
            if key.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if len(encoder_state) < len(self.encoder.state_dict()) // 2:
            raise RuntimeError("Too few legacy encoder tensors were recovered")
        print(
            f"Loaded legacy SSL encoder: {len(encoder_state)} tensors | "
            f"missing {len(missing)} | unexpected {len(unexpected)}"
        )

    def forward(self, images):
        return F.normalize(self.encoder(images), dim=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare raw PersonViT and all local ReID checkpoints."
    )
    parser.add_argument("--data-root", default="AG-ReID.v2")
    parser.add_argument("--protocol", default="exp1_aerial_to_cctv.txt")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--personvit-checkpoint", default="pretrained/checkpoint0240.pth"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument("--skip-personvit-base", action="store_true")
    parser.add_argument(
        "--only",
        help="Comma-separated candidate names to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add a checkpoint outside the outputs directory (repeatable).",
    )
    parser.add_argument(
        "--report-dir", default="outputs/comparison", help="CSV/Markdown output."
    )
    parser.add_argument("--list", action="store_true", help="List candidates only.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _parse_custom(value: str) -> Candidate:
    if "=" not in value:
        raise ValueError(f"--checkpoint must use NAME=PATH, got: {value}")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError(f"--checkpoint must use NAME=PATH, got: {value}")
    return Candidate(name.strip(), Path(raw_path.strip()))


def discover_candidates(args):
    candidates = []
    if not args.skip_personvit_base:
        candidates.append(Candidate("personvit_base", None, "personvit_base"))

    outputs_root = Path(args.outputs_root)
    if outputs_root.is_dir():
        for checkpoint in sorted(outputs_root.rglob("best_model.pth")):
            relative_parent = checkpoint.parent.relative_to(outputs_root)
            name = str(relative_parent).replace("\\", "/")
            if name == ".":
                name = "root_best_model"
            candidates.append(Candidate(name, checkpoint))
    candidates.extend(_parse_custom(value) for value in args.checkpoint)

    deduplicated = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate.name,
            str(candidate.checkpoint.resolve()) if candidate.checkpoint else None,
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(candidate)

    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        deduplicated = [
            candidate
            for candidate in deduplicated
            if candidate.name in wanted or candidate.kind in wanted
        ]
    return deduplicated


def _checkpoint_state(payload):
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state


def infer_kind(state):
    keys = set(state)
    if any(key.startswith("local_bnnecks.") for key in keys):
        return "proposed_v3"
    if "view_adapter.weight" in keys and "bnneck.weight" in keys:
        return "proposed_v2"
    if "bnneck.weight" in keys and "identity_classifier.weight" in keys:
        return "personvit_finetune"
    if "projector.0.weight" in keys and "identity_head.weight" in keys:
        return "proposed_lewm"
    if "encoder.reg_token" in keys and any(
        key.startswith(("projector.", "predictor.")) for key in keys
    ):
        return "legacy_vicreg"
    if "mask_token" in keys and "bottleneck.weight" in keys:
        return "untransreid"
    if "global_bn.weight" in keys and "global_classifier.weight" in keys:
        return "transreid"
    if "bottleneck.weight" in keys and "classifier.weight" in keys:
        return "bnneck"
    raise ValueError("Could not infer a source model from checkpoint tensor names")


def _config_from_checkpoint(config_class, saved, args):
    cfg = config_class()
    allowed = {field.name for field in fields(config_class)}
    for name, value in (saved or {}).items():
        if name in allowed:
            setattr(cfg, name, value)
    cfg.data_root = args.data_root
    cfg.eval_txt_file = args.protocol
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.device = args.device
    cfg.flip_tta = not args.no_flip_tta
    # Full task checkpoints are loaded immediately, so model constructors must
    # not download or replace their encoder with timm ImageNet weights.
    cfg.pretrained = False
    return cfg


def _num_classes(state, key):
    if key not in state:
        raise KeyError(f"Cannot infer class count: {key} is absent")
    return int(state[key].shape[0])


def build_checkpoint_model(kind, state, saved_config, args):
    if kind == "bnneck":
        cfg = _config_from_checkpoint(BNNeckConfig, saved_config, args)
        model = BNNeckModel(cfg, _num_classes(state, "classifier.weight"))
    elif kind == "personvit_finetune":
        cfg = _config_from_checkpoint(PersonViTFineTuneConfig, saved_config, args)
        cfg.ssl_pretrained_path = args.personvit_checkpoint
        model = PersonViTFineTuneModel(
            cfg, _num_classes(state, "identity_classifier.weight")
        )
    elif kind == "transreid":
        cfg = _config_from_checkpoint(TransReIDConfig, saved_config, args)
        model = TransReIDSmall(
            cfg, _num_classes(state, "global_classifier.weight")
        )
    elif kind == "untransreid":
        cfg = _config_from_checkpoint(UntransReIDConfig, saved_config, args)
        # The complete encoder is already in the task checkpoint.
        cfg.ssl_pretrained_path = ""
        model = UntransReIDSmall(cfg)
    elif kind == "proposed_lewm":
        cfg = _config_from_checkpoint(ProposedConfig, saved_config, args)
        model = LeWMReID(cfg, _num_classes(state, "identity_head.weight"))
    elif kind == "legacy_vicreg":
        cfg = _config_from_checkpoint(ProposedConfig, saved_config, args)
        model = LegacySSLModel(cfg, state)
    elif kind == "proposed_v2":
        cfg = _config_from_checkpoint(ProposedV2Config, saved_config, args)
        cfg.ssl_pretrained_path = args.personvit_checkpoint
        model = ProposedV2Model(
            cfg, _num_classes(state, "identity_classifier.weight")
        )
    elif kind == "proposed_v3":
        cfg = _config_from_checkpoint(ProposedV3Config, saved_config, args)
        cfg.ssl_pretrained_path = args.personvit_checkpoint
        model = ProposedV3Model(
            cfg, _num_classes(state, "identity_classifier.weight")
        )
    else:
        raise ValueError(f"Unsupported checkpoint kind: {kind}")
    if kind != "legacy_vicreg":
        model.load_state_dict(state, strict=True)
    return model, cfg


def build_personvit_base(args):
    checkpoint = Path(args.personvit_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PersonViT checkpoint not found: {checkpoint}")
    cfg = ProposedV2Config()
    cfg.data_root = args.data_root
    cfg.eval_txt_file = args.protocol
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.device = args.device
    cfg.flip_tta = not args.no_flip_tta
    cfg.pretrained = False
    cfg.ssl_pretrained_path = str(checkpoint)
    return PersonViTBase(cfg, str(checkpoint)), cfg


def _default_label_fraction(kind, saved_config):
    if "labeled_fraction" in saved_config:
        return saved_config["labeled_fraction"]
    if kind in ("untransreid", "personvit_base", "legacy_vicreg"):
        return 0.0
    if kind in ("bnneck", "transreid"):
        return 1.0
    return ""


def evaluate_candidate(candidate, args):
    if candidate.kind == "personvit_base":
        kind = "personvit_base"
        payload = {}
        saved_config = {}
        model, cfg = build_personvit_base(args)
        checkpoint_label = args.personvit_checkpoint
        epoch = "pretrained"
    else:
        if candidate.checkpoint is None or not candidate.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {candidate.checkpoint}")
        payload = torch.load(
            candidate.checkpoint, map_location="cpu", weights_only=False
        )
        state = _checkpoint_state(payload)
        kind = infer_kind(state)
        if kind in {
            "bnneck",
            "personvit_finetune",
            "transreid",
            "proposed_lewm",
            "proposed_v2",
            "proposed_v3",
        }:
            require_current_identity_scheme(payload, str(candidate.checkpoint))
        saved_config = payload.get("config", {}) if isinstance(payload, dict) else {}
        model, cfg = build_checkpoint_model(kind, state, saved_config, args)
        checkpoint_label = str(candidate.checkpoint)
        epoch = payload.get("epoch", "") if isinstance(payload, dict) else ""

    resolve_device(cfg)
    model = model.to(cfg.device).eval()
    query_loader, gallery_loader = build_eval_loaders(cfg)
    rank1, mean_ap = evaluate_model(model, cfg, query_loader, gallery_loader)
    row = {
        "model": candidate.name,
        "kind": kind,
        "checkpoint": checkpoint_label,
        "epoch": epoch,
        "label_fraction": _default_label_fraction(kind, saved_config),
        "protocol": cfg.eval_txt_file,
        "image_size": f"{cfg.image_size[0]}x{cfg.image_size[1]}",
        "flip_tta": cfg.flip_tta,
        "rank1": rank1,
        "mAP": mean_ap,
        "status": "ok",
    }
    del model, query_loader, gallery_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _display_fraction(value):
    if isinstance(value, (float, int)):
        return f"{100.0 * float(value):.0f}%"
    return str(value)


def print_results(rows):
    print("\nAG-ReID.v2 experiment comparison")
    print(f"{'Model':<28} {'Labels':>8} {'Rank-1':>10} {'mAP':>10}  Status")
    print("-" * 78)
    for row in rows:
        rank1 = f"{100.0 * row['rank1']:.2f}%" if row["status"] == "ok" else "-"
        mean_ap = f"{100.0 * row['mAP']:.2f}%" if row["status"] == "ok" else "-"
        print(
            f"{row['model'][:28]:<28} "
            f"{_display_fraction(row['label_fraction']):>8} "
            f"{rank1:>10} {mean_ap:>10}  {row['status']}"
        )


def write_reports(rows, args):
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "metrics.csv"
    markdown_path = report_dir / "metrics.md"
    columns = (
        "model",
        "kind",
        "checkpoint",
        "epoch",
        "label_fraction",
        "protocol",
        "image_size",
        "flip_tta",
        "rank1",
        "mAP",
        "status",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# AG-ReID.v2 experiment comparison\n\n")
        handle.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        handle.write(f"Protocol: `{args.protocol}`\n\n")
        handle.write("| Model | Kind | Labels | Epoch | Rank-1 | mAP | Status |\n")
        handle.write("|---|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            rank1 = (
                f"{100.0 * row['rank1']:.2f}%" if row["status"] == "ok" else "-"
            )
            mean_ap = (
                f"{100.0 * row['mAP']:.2f}%" if row["status"] == "ok" else "-"
            )
            status = str(row["status"]).replace("|", "\\|")
            handle.write(
                f"| {row['model']} | {row['kind']} | "
                f"{_display_fraction(row['label_fraction'])} | {row['epoch']} | "
                f"{rank1} | {mean_ap} | {status} |\n"
            )
    print(f"\nCSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    candidates = discover_candidates(args)
    if not candidates:
        raise RuntimeError("No experiments matched the requested filters")
    print("Evaluation candidates:")
    for candidate in candidates:
        source = candidate.checkpoint or Path(args.personvit_checkpoint)
        exists = source.is_file()
        print(f"  {candidate.name:<28} {source} [{'ok' if exists else 'missing'}]")
    if args.list:
        return

    rows = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"\n[{index}/{len(candidates)}] Evaluating {candidate.name}")
        try:
            row = evaluate_candidate(candidate, args)
            print(
                f"{candidate.name} | Rank-1 {row['rank1']:.2%} | "
                f"mAP {row['mAP']:.2%}"
            )
        except Exception as error:  # keep a long comparison run useful
            if args.fail_fast:
                raise
            print(f"ERROR {candidate.name}: {error}")
            row = {
                "model": candidate.name,
                "kind": candidate.kind,
                "checkpoint": str(candidate.checkpoint or args.personvit_checkpoint),
                "epoch": "",
                "label_fraction": "",
                "protocol": args.protocol,
                "image_size": "",
                "flip_tta": not args.no_flip_tta,
                "rank1": "",
                "mAP": "",
                "status": f"error: {type(error).__name__}: {error}",
            }
        rows.append(row)

    print_results(rows)
    write_reports(rows, args)


if __name__ == "__main__":
    main()
