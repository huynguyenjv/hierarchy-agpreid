"""Single launcher for all project experiments."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

# Must be set before any pipeline imports TensorBoard/TensorFlow discovery.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from reid_advance.config import (
    BNNeckConfig,
    PersonViTFineTuneConfig,
    ProposedConfig,
    ProposedV2Config,
    ProposedV3Config,
    ProposedV3FineTuneConfig,
    TransReIDConfig,
    UntransReIDConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        choices=(
            "bnneck",
            "personvit_ft",
            "personvit-ft",
            "transreid",
            "untransreid",
            "ssl",
            "proposed",
            "proposed_v2",
            "proposed-v2",
            "proposed_v3",
            "proposed-v3",
            "proposed_v3_ft",
            "proposed-v3-ft",
        ),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--protocol", dest="eval_txt_file")
    parser.add_argument("--labeled-fraction", type=float)
    parser.add_argument("--ssl-checkpoint", dest="ssl_pretrained_path")
    parser.add_argument("--dbscan-eps", type=float)
    parser.add_argument("--dbscan-eps-start", type=float)
    parser.add_argument("--dbscan-eps-end", type=float)
    parser.add_argument("--recluster-interval", type=int)
    parser.add_argument("--init-checkpoint", dest="initialization_checkpoint")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def build_experiment(method):
    if method == "bnneck":
        from reid_advance.pipelines.bnneck import train_bnneck

        return BNNeckConfig(), train_bnneck
    if method in ("personvit_ft", "personvit-ft"):
        from reid_advance.pipelines.personvit_finetune import (
            train_personvit_finetune,
        )

        return PersonViTFineTuneConfig(), train_personvit_finetune
    if method == "transreid":
        from reid_advance.pipelines.transreid import train_transreid

        return TransReIDConfig(), train_transreid
    if method in ("untransreid", "ssl"):
        from reid_advance.pipelines.untransreid import train_untransreid

        return UntransReIDConfig(), train_untransreid
    if method in ("proposed_v2", "proposed-v2"):
        from reid_advance.pipelines.proposed_v2 import train_proposed_v2

        return ProposedV2Config(), train_proposed_v2
    if method in ("proposed_v3", "proposed-v3"):
        from reid_advance.pipelines.proposed_v3 import train_proposed_v3

        return ProposedV3Config(), train_proposed_v3
    if method in ("proposed_v3_ft", "proposed-v3-ft"):
        from reid_advance.pipelines.proposed_v3 import train_proposed_v3

        return ProposedV3FineTuneConfig(), train_proposed_v3
    from reid_advance.pipelines.proposed import train_proposed

    return ProposedConfig(), train_proposed


def main():
    args = parse_args()
    config, train = build_experiment(args.method)
    for name in (
        "epochs",
        "batch_size",
        "eval_txt_file",
        "labeled_fraction",
        "ssl_pretrained_path",
        "dbscan_eps",
        "dbscan_eps_start",
        "dbscan_eps_end",
        "recluster_interval",
        "initialization_checkpoint",
        "output_dir",
    ):
        value = getattr(args, name, None)
        if value is not None:
            if not hasattr(config, name):
                raise ValueError(f"{name} is not supported by {args.method}")
            setattr(config, name, value)
    print(f"Running {args.method} with:")
    for name, value in asdict(config).items():
        print(f"  {name}: {value}")
    train(config)


if __name__ == "__main__":
    main()
