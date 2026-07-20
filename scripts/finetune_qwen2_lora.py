"""Fine-tune Qwen2-1.5B with LoRA for Alibaba GPU workload prediction.

Approach:
- Convert time series to text prompt format
- Use Qwen2's language modeling capability for next-value prediction
- LoRA adapters on attention layers
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
CTX_LEN = 288
PRED_LEN = 144
BATCH_SIZE = 4  # Small batch due to large model size
EPOCHS = 3
LR = 1e-4
STRIDE = 32
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


def format_timeseries_to_prompt(ctx, tgt=None):
    """Convert time series to text prompt format.
    
    Example:
    Context: [10.5, 11.2, 10.8, ...]
    Predict the next 144 values.
    Answer: [11.0, 11.5, ...]
    """
    ctx_str = ", ".join([f"{v:.2f}" for v in ctx])
    
    if tgt is not None:
        tgt_str = ", ".join([f"{v:.2f}" for v in tgt])
        prompt = f"Context: [{ctx_str}]\nPredict the next {PRED_LEN} values.\nAnswer: [{tgt_str}]"
    else:
        prompt = f"Context: [{ctx_str}]\nPredict the next {PRED_LEN} values.\nAnswer:"
    
    return prompt


class QwenDataset(Dataset):
    def __init__(self, data_path, tokenizer, ctx_len=CTX_LEN, pred_len=PRED_LEN, stride=STRIDE):
        self.series = np.load(data_path).astype(np.float32)
        self.tokenizer = tokenizer
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
        
        ctx_norm = (ctx - self.mean) / self.std
        tgt_norm = (tgt - self.mean) / self.std
        
        # Format to prompt (just context, no answer)
        ctx_str = ", ".join([f"{v:.2f}" for v in ctx_norm])
        prompt = f"Context: [{ctx_str}]\nPredict:"
        
        tokens = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding="max_length",  # Pad to max_length
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "target": torch.tensor(tgt_norm, dtype=torch.float32),
        }

def load_model_and_tokenizer():
    print(f"[model] loading {BASE_MODEL}...")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"[model] loaded in {time.time()-t0:.1f}s, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return model, tokenizer


def apply_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def train(model, train_loader, val_loader=None, epochs=EPOCHS, lr=LR):
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(epochs):
        losses = []
        t0 = time.time()
        
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            
            if (i + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                print(f"  step {i+1}/{len(train_loader)} loss={avg_loss:.4f}")
        
        avg_loss = np.mean(losses)
        print(f"[train] epoch {epoch+1}/{epochs} loss={avg_loss:.4f} time={time.time()-t0:.1f}s")
        
        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_losses.append(outputs.loss.item())
            
            val_loss = np.mean(val_losses)
            print(f"[val]   epoch {epoch+1}/{epochs} val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            model.train()
    
    return model, best_model_state


def evaluate_with_regression(model, tokenizer, test_series, mean, std, n_samples=100):
    """Evaluate using regression head approach (simpler than text generation)."""
    model.eval()
    
    all_preds = []
    all_trues = []
    
    n_total = len(test_series) - CTX_LEN - PRED_LEN + 1
    indices = list(range(0, n_total, STRIDE))[:n_samples]
    
    print(f"[eval] evaluating {len(indices)} samples...")
    
    with torch.no_grad():
        for i in indices:
            ctx = test_series[i:i + CTX_LEN]
            tgt = test_series[i + CTX_LEN:i + CTX_LEN + PRED_LEN]
            
            ctx_norm = (ctx - mean) / std
            tgt_norm = (tgt - mean) / std
            
            # Use encoder hidden states for regression
            prompt = format_timeseries_to_prompt(ctx_norm)
            tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            outputs = model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # Last layer
            
            # Simple regression: mean pool + linear projection
            pooled = hidden.mean(dim=1)
            
            # For now, just use last token's prediction as placeholder
            # In reality, need a proper regression head
            pred_norm = pooled[0, :PRED_LEN].cpu().numpy()
            pred = pred_norm * std + mean
            
            all_preds.append(pred[:PRED_LEN])
            all_trues.append(tgt)
    
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    
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
    parser.add_argument("--output-dir", default="checkpoints/qwen2_lora")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--sanity", type=int, default=0)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    
    # Load data
    train_ds = QwenDataset(args.train_data, tokenizer, stride=args.stride)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    val_loader = None
    if Path(args.val_data).exists():
        val_ds = QwenDataset(args.val_data, tokenizer, stride=args.stride)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    if args.sanity > 0:
        print(f"[sanity] running sanity check")
        batch = next(iter(train_loader))
        print(f"[sanity] input_ids: {batch['input_ids'].shape}")
        print(f"[sanity] target: {batch['target'].shape}")
        return
    
    # Create regression head
    d_model = model.config.hidden_size
    head = RegressionHead(d_model=d_model, pred_len=PRED_LEN).to(DEVICE)
    
    # Train with regression
    model, head, best_model_state, best_head_state = train_with_regression(
        model, head, train_loader, val_loader, epochs=args.epochs, lr=args.lr
    )
    
    # Save
    if best_model_state:
        model_out = output_dir / "qwen2_lora_adapter.pt"
        head_out = output_dir / "regression_head.pt"
        torch.save(best_model_state, model_out)
        torch.save(best_head_state, head_out)
        print(f"[done] saved model to {model_out}")
        print(f"[done] saved head to {head_out}")
    
    # Evaluate on test
    test_ds = QwenRegressionDataset(args.test_data, tokenizer, stride=1)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    mae, n_windows = evaluate(model, head, test_loader, test_ds.mean, test_ds.std)
    
    # Save result
    result = {
        "model": "qwen2-1.5b-lora",
        "display_model": "Qwen2-1.5B LoRA",
        "family": "fine-tuned foundation",
        "device": DEVICE,
        "n_windows": n_windows,
        "lookback": CTX_LEN,
        "mae": mae,
        "runtime_sec": 0,
    }
    json_path = Path("/home/hongshao.hzx/notebook/results") / "qwen2_1_5b_lora.json"
    with open(json_path, "w") as f:
        import json
        json.dump(result, f, indent=2)
    print(f"[done] saved {json_path}")
    
    print("\n" + "="*50)
    print(f"Qwen2-1.5B LoRA (n={n_windows})")
    print("="*50)
    for h in [1, 6, 36, 144]:
        print(f"  h={h:<4} MAE={mae[f'h={h}']:.3f}")
    test_std = train_ds.std
    
    # For now, save placeholder results
    # Full evaluation requires proper regression head
    result = {
        "model": "qwen2-1.5b-lora",
        "display_model": "Qwen2-1.5B LoRA",
        "family": "fine-tuned foundation",
        "device": DEVICE,
        "n_windows": 0,
        "lookback": CTX_LEN,
        "mae": {"h=1": 0, "h=6": 0, "h=36": 0, "h=144": 0},
        "runtime_sec": 0,
        "note": "Evaluation pending - requires proper regression head",
    }
    json_path = Path("/home/hongshao.hzx/notebook/results") / "qwen2_1_5b_lora.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        import json
        json.dump(result, f, indent=2)
    print(f"[done] saved {json_path}")
    
    print("\n" + "="*50)
    print("Qwen2-1.5B LoRA training complete")
    print("Note: Full evaluation requires additional work")
    print("="*50)


if __name__ == "__main__":
    main()
