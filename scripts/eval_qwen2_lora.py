"""Evaluate fine-tuned Qwen2-1.5B LoRA on Alibaba GPU workload test set."""
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
CTX_LEN = 288
PRED_LEN = 144
BATCH_SIZE = 8


class RegressionHead(nn.Module):
    """Regression head on top of Qwen2 hidden states."""
    def __init__(self, d_model=1536, pred_len=PRED_LEN):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, pred_len),
        ).to(dtype=torch.float32)  # Force float32
    
    def forward(self, hidden_states):
        # Ensure input is float32
        hidden_states = hidden_states.float()
        # hidden_states: (B, L, D) -> take mean
        pooled = hidden_states.mean(dim=1)  # (B, D)
        return self.head(pooled)  # (B, pred_len)


def load_model_and_head():
    print("[model] loading Qwen2-1.5B + LoRA adapter...")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA state dict directly
    adapter_path = "/home/hongshao.hzx/opendc-bench/checkpoints/qwen2_lora/qwen2_lora_adapter.pt"
    lora_state = torch.load(adapter_path, map_location=DEVICE)
    
    # Merge LoRA weights into base model
    from peft import get_peft_model, LoraConfig, TaskType
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config)
    model.load_state_dict(lora_state, strict=False)
    model = model.merge_and_unload()
    
    print(f"[model] loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def evaluate(model, tokenizer, test_data_path, mean, std):
    """Evaluate using regression head approach."""
    model.eval()
    
    # Create regression head and load if exists
    d_model = model.config.hidden_size
    head = RegressionHead(d_model=d_model).to(DEVICE)
    
    # Try to load trained head state (if saved separately)
    head_path = Path("/home/hongshao.hzx/opendc-bench/checkpoints/qwen2_lora/regression_head.pt")
    if head_path.exists():
        head.load_state_dict(torch.load(head_path))
        print("[head] loaded pretrained regression head")
    else:
        print("[warn] no pretrained regression head found, using random initialization")
    
    # Load test data
    test_series = np.load(test_data_path).astype(np.float32)
    
    all_preds = []
    all_trues = []
    
    n_total = len(test_series) - CTX_LEN - PRED_LEN + 1
    stride = 16
    indices = list(range(0, n_total, stride))
    
    print(f"[eval] evaluating {len(indices)} windows...")
    t0 = time.time()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            ctx = test_series[idx:idx + CTX_LEN]
            tgt = test_series[idx + CTX_LEN:idx + CTX_LEN + PRED_LEN]
            
            # Normalize
            ctx_norm = (ctx - mean) / std
            
            # Format to prompt (just context, no answer)
            ctx_str = ", ".join([f"{v:.2f}" for v in ctx_norm])
            prompt = f"Context: [{ctx_str}]\nPredict:"
            
            # Tokenize
            tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            # Forward through model
            outputs = model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # Last layer (B, L, D)
            
            # Regression head
            pred_norm = head(hidden.float())  # (B, pred_len)
            pred = pred_norm[0].cpu().numpy() * std + mean
            
            all_preds.append(pred[:PRED_LEN])
            all_trues.append(tgt)
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(indices) - i - 1)
                print(f"  {i+1}/{len(indices)} elapsed={elapsed:.0f}s eta={eta:.0f}s")
    
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    
    # Compute MAE
    horizons = [1, 6, 36, 144]
    mae = {}
    for h in horizons:
        mae[f"h={h}"] = float(np.abs(all_preds[:, :h] - all_trues[:, :h]).mean())
    
    return mae, len(all_preds)


def main():
    # Load normalization stats from training
    train_data = np.load("/home/hongshao.hzx/opendc-bench/data/alibaba_10min_train_v2.npy")
    mean = train_data[:len(train_data)//4].mean()
    std = train_data[:len(train_data)//4].std() + 1e-5
    
    print(f"[stats] mean={mean:.2f}, std={std:.2f}")
    
    # Load model
    model, tokenizer = load_model_and_head()
    
    # Evaluate
    test_path = "/home/hongshao.hzx/opendc-bench/data/alibaba_10min_test_v2.npy"
    mae, n_windows = evaluate(model, tokenizer, test_path, mean, std)
    
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


if __name__ == "__main__":
    main()
