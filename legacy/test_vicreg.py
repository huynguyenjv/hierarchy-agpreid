"""Compare two images with a trained hybrid SSL encoder."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from config import Config
from dataset import build_eval_transform
from model import HybridSSLModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image1")
    parser.add_argument("image2")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    checkpoint = args.checkpoint or f"{cfg.output_dir}/{cfg.hybrid_ckpt_name}"
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    for key, value in state.get("config", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.pretrained = False
    model = HybridSSLModel(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    transform = build_eval_transform(cfg.image_size, cfg.norm_mean, cfg.norm_std)
    images = [
        transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        for path in (args.image1, args.image2)
    ]
    with torch.no_grad():
        feature1, feature2 = (model(image) for image in images)
    similarity = F.cosine_similarity(feature1, feature2).item()
    print(f"Cosine similarity: {similarity:.6f}")


if __name__ == "__main__":
    main()
