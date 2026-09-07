import torch
import torch.cuda.amp as amp
from tqdm import tqdm

class ReIDTrainer:
    def __init__(self, model, optimizer, scheduler, criterion_fn, device, cfg, writer):
        self.model, self.opt, self.sch, self.crit = model, optimizer, scheduler, criterion_fn
        self.device, self.cfg, self.writer = device, cfg, writer
        self.scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    def train_epoch(self, epoch, loader):
        self.model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{self.cfg.epochs}", leave=False)
        for data in pbar:
            data = [d.to(self.device, non_blocking=True) for d in data]
            self.opt.zero_grad(set_to_none=True)
            
            with amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                loss, logs = self.crit(self.model, data)
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            
            pbar.set_postfix({k: f"{v:.3f}" for k, v in logs.items()})
        
        self.sch.step()
        return logs