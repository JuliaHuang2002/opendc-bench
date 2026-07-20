"""Fine-tune T5-Small with LoRA for direct regression on Alibaba GPU workload.

This is a simplified T5-based approach that mimics Chronos architecture:
- Load standard T5-Small (same size as Chronos-T5-Small)
- Add LoRA adapters via peft
- Train as direct regression: context (288) -> target (144)
- Evaluate with same unified JSON format
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import get_peft_model, LoraConfig

# Allow downloading t5-small cache
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "t5-small"  # Standard T5, same size as Chronos-T5-Small
CTX_LEN = 288
PRED_LEN = 144
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
STRIDE = 16
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


class SimpleRegressionHead(nn.Module):
    """Simple regression head on top of T5 encoder."""
    def __init__(self, d_model=512, pred_len=PRED_LEN):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, pred_len),
        )
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)  # (B, D)
        return self.head(pooled)  # (B, pred_len)


class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, ctx_len=CTX_LEN, pred_len=PRED_LEN, stride=STRIDE):
        self.series = np.load(data_path).astype(np.float32)
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.stride = stride
        self.n_windows = max(1, (len(self.series) - ctx_len - pred_len) // stride)
        
        self.mean = self.series[:len(self.series)//4].mean()
        self.std = self.series[:len(self.series)//4].std() + 1e-5
        print(f"[dataset] {len(self.series)} points -> {self.n_windows} windows, mean={self.mean:.2f}, std={self.std:.2f}")
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start = idx * self.stride
        ctx = self.series[start:start + self.ctx_len]
        tgt = self.series[start + self.ctx_len:start + self.ctx_len + self.pred_len]
        
        ctx = (ctx - self.mean) / self.std
        tgt = (tgt - self.mean) / self.std
        
        return torch.tensor(ctx, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)


def load_model_and_tokenizer():
    print(f"[model] loading {BASE_MODEL}...")
    t0 = time.time()
    
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
    
    print(f"[model] loaded in {time.time()-t0:.1f}s, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, tokenizer


def apply_lora(model):
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q", "k", "v", "o"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def train(model, regression_head, train_loader, val_loader=None, epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    regression_head = regression_head.to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(regression_head.parameters()),
        lr=lr
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    best_model_state = None
    best_head_state = None
    
    for epoch in range(epochs):
        model.train()
        regression_head.train()
        losses = []
        t0 = time.time()
        
        for ctx, tgt in train_loader:
            ctx = ctx.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            B = ctx.size(0)
            ctx_discrete = torch.bucketize(ctx, torch.linspace(-3, 3, 256).to(DEVICE))
            input_ids = ctx_discrete.long().clamp(0, 255)
            attention_mask = torch.ones_like(input_ids)
            
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            
            pred = regression_head(hidden)
            
            loss = criterion(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        print(f"[train] epoch {epoch+1}/{epochs} loss={avg_loss:.4f} time={time.time()-t0:.1f}s")
        
        if val_loader:
            model.eval()
            regression_head.eval()
            val_losses = []
            with torch.no_grad():
                for ctx, tgt in val_loader:
                    ctx = ctx.to(DEVICE)
                    tgt = tgt.to(DEVICE)
                    ctx_discrete = torch.bucketize(ctx, torch.linspace(-3, 3, 256).to(DEVICE))
                    input_ids = ctx_discrete.long().clamp(0, 255)
                    attention_mask = torch.ones_like(input_ids)
                    outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    hidden = outputs.last_hidden_state
                    pred = regression_head(hidden)
                    loss = criterion(pred, tgt)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            print(f"[val]   epoch {epoch+1}/{epochs} val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_head_state = {k: v.clone() for k, v in regression_head.state_dict().items()}
    
    return model, regression_head, best_model_state, best_head_state


def evaluate(model, regression_head, test_loader, mean, std):
    model.eval()
    regression_head.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for ctx, tgt in test_loader:
            ctx = ctx.to(DEVICE)
            ctx_discrete = torch.bucketize(ctx, torch.linspace(-3, 3, 256).to(DEVICE))
            input_ids = ctx_discrete.long().clamp(0, 255)
            attention_mask = torch.ones_like(input_ids)
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            pred = regression_head(hidden)
            
            pred = pred * std + mean
            tgt_denorm = tgt * std + mean
            
            all_preds.append(pred.cpu().numpy())
            all_trues.append(tgt_denorm.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    horizons = [1, 6, 36, 144]
    mae = {}
    for h in horizons:
        mae[f"h={h}"] = float(np.abs(all_preds[:, :h] - all_trues[:, :h]).mean())
    
    return mae, len(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/alibaba_10min_train_v2.npy")
    parser.add_argument("--val-data", default="data/alibaba_10min_val_v2.npy")
    parser.add_argument("--test-data", default="data/alibaba_10min_test_v2.npy")
    parser.add_argument("--output-dir", default="checkpoints/t5_lora_small")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--sanity", type=int, default=0)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_ds = TimeSeriesDataset(args.train_data, stride=args.stride)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    val_loader = None
    if Path(args.val_data).exists():
        val_ds = TimeSeriesDataset(args.val_data, stride=args.stride)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    
    d_model = model.config.d_model
    regression_head = SimpleRegressionHead(d_model=d_model)
    
    if args.sanity > 0:
        print(f"[sanity] running sanity check")
        ctx, tgt = next(iter(train_loader))
        print(f"[sanity] ctx: {ctx.shape}, tgt: {tgt.shape}")
        print(f"[sanity] d_model: {d_model}")
        return
    
    model, regression_head, best_model_state, best_head_state = train(
        model, regression_head, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr
    )
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        regression_head.load_state_dict(best_head_state)
    
    out_path = output_dir / "t5_lora_adapter.pt"
    torch.save({
        "model_state": best_model_state,
        "head_state": best_head_state,
        "mean": train_ds.mean,
        "std": train_ds.std,
    }, out_path)
    print(f"[done] saved {out_path}")
    
    test_ds = TimeSeriesDataset(args.test_data, stride=1)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    mae, n_windows = evaluate(model, regression_head, test_loader, test_ds.mean, test_ds.std)
    
    result = {
        "model": "t5-small-lora",
        "display_model": "T5-Small LoRA (Chronos-like)",
        "family": "fine-tuned foundation",
        "device": DEVICE,
        "n_windows": n_windows,
        "lookback": CTX_LEN,
        "mae": mae,
        "runtime_sec": 0,
    }
    json_path = Path("/home/hongshao.hzx/notebook/results") / "t5_small_lora.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        import json
        json.dump(result, f, indent=2)
    print(f"[done] saved {json_path}")
    
    print("\n" + "="*50)
    print(f"T5-Small LoRA (n={n_windows})")
    print("="*50)
    for h in [1, 6, 36, 144]:
        print(f"  h={h:<4} MAE={mae[f'h={h}']:.3f}")


if __name__ == "__main__":
    main()
