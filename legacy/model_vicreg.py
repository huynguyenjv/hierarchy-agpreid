import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from attributes import ATTRIBUTE_CLASS_COUNTS
from config import Config

def apply_partial_freeze(model: nn.Module, finetune_last_n_blocks: int = 2):
    if finetune_last_n_blocks < 0:
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters(): p.requires_grad = False
    if hasattr(model, 'blocks') and finetune_last_n_blocks > 0:
        for blk in model.blocks[-finetune_last_n_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
    if hasattr(model, 'norm'):
        for p in model.norm.parameters(): p.requires_grad = True

class SupervisedBaseline(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(cfg.encoder_name, pretrained=cfg.pretrained, num_classes=0, img_size=cfg.image_size[0])
        apply_partial_freeze(self.encoder, finetune_last_n_blocks=2)
        
        self.bottleneck = nn.BatchNorm1d(cfg.embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(cfg.embed_dim, cfg.num_classes, bias=False)
        
        self.bottleneck.apply(self._weights_init_kaiming)
        self.classifier.apply(self._weights_init_classifier)

    def _weights_init_kaiming(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def _weights_init_classifier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, std=0.001)

    def forward(self, x):
        features = self.encoder(x) 
        feat_bn = self.bottleneck(features)
        
        if not self.training:
            return F.normalize(feat_bn, dim=1) 
            
        logits = self.classifier(feat_bn) 
        return features, logits

    def get_optim_params(self, base_lr):
        return [
            {'params': filter(lambda p: p.requires_grad, self.encoder.parameters()), 'lr': base_lr},
            {'params': self.bottleneck.parameters(), 'lr': base_lr * 10},
            {'params': self.classifier.parameters(), 'lr': base_lr * 10}
        ]

class HybridSSLModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(cfg.encoder_name, pretrained=cfg.pretrained, num_classes=0, img_size=cfg.image_size[0])
        apply_partial_freeze(self.encoder, cfg.finetune_last_n_blocks)
        encoder_dim = int(getattr(self.encoder, "num_features", cfg.embed_dim))
        if encoder_dim != cfg.embed_dim:
            raise ValueError(
                f"Configured embed_dim={cfg.embed_dim}, but {cfg.encoder_name} outputs {encoder_dim}."
            )

        # VICReg operates on projection-space embeddings. Retrieval and
        # attributes use the encoder representation, never this projection.
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, cfg.projector_hidden_dim, bias=False),
            nn.BatchNorm1d(cfg.projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.projector_hidden_dim, cfg.projector_out_dim),
        )

        self.attr_heads = nn.ModuleDict({
            name: nn.Linear(encoder_dim, num_cls)
            for name, num_cls in ATTRIBUTE_CLASS_COUNTS.items()
        })

        self.attr_heads.apply(self._weights_init_classifier)

    def _weights_init_classifier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, v1, v2=None):
        if v2 is None:
            return F.normalize(self.encoder(v1), dim=1)

        h1, h2 = self.encoder(v1), self.encoder(v2)
        z1, z2 = self.projector(h1), self.projector(h2)

        attr_logits1 = {name: head(h1) for name, head in self.attr_heads.items()}
        attr_logits2 = {name: head(h2) for name, head in self.attr_heads.items()}

        return z1, z2, h1, h2, attr_logits1, attr_logits2

    def get_optim_params(self, base_lr):
        return [
            {'params': filter(lambda p: p.requires_grad, self.encoder.parameters()), 'lr': base_lr},
            {'params': self.projector.parameters(), 'lr': base_lr * self.cfg.head_lr_multiplier},
            {'params': self.attr_heads.parameters(), 'lr': base_lr * self.cfg.head_lr_multiplier}
        ]
