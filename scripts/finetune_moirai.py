"""Fine-tune Moirai-1.1-R-base on Alibaba GPU workload.
Fixes applied: sample_id 2D, module= (not state_dict), dynamic warmup."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule
from uni2ts.data.builder.simple import SimpleFinetuneDatasetBuilder
from uni2ts.data.loader import DataLoader as Uni2tsDataLoader

DATASET_NAME = "alibaba_gpu_train"
CSV_PATH = Path("data/alibaba_gpu_train.csv")
ARROW_STORAGE = Path("data/uni2ts_cache")
CTX_LEN = 288
PRED_LEN = 144
PATCH_SIZE = 16
BATCH_SIZE = 32
MODEL_NAME = "Salesforce/moirai-1.1-R-base"


class MoiraiFinetuneFixed(MoiraiFinetune):
    """Inject sample_id (B, num_patches) when DataLoader skips packing."""
    def _inject_sample_id(self, batch):
        if "sample_id" not in batch:
            B, L = batch["target"].shape[0], batch["target"].shape[1]
            batch["sample_id"] = torch.arange(B, device=batch["target"].device).unsqueeze(1).expand(B, L)
        return batch

    def training_step(self, batch, batch_idx):
        return super().training_step(self._inject_sample_id(batch), batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step(self._inject_sample_id(batch), batch_idx)


def prepare_data():
    train_arr_path = Path("data/alibaba_10min_train_v2.npy")
    if not train_arr_path.exists():
        alts = list(Path("data").glob("alibaba*train*.npy"))
        if not alts:
            raise FileNotFoundError(f"No train npy found in data/")
        train_arr_path = alts[0]
    series = np.load(train_arr_path).astype(np.float32)
    idx = pd.date_range("2020-01-01", periods=len(series), freq="10min")
    df = pd.DataFrame({"alibaba_gpu": series}, index=idx)
    df.index.name = "timestamp"
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH)
    print(f"[prepare_data] {len(series)} steps -> {CSV_PATH}")
    return len(series)


def build_arrow_if_needed():
    arrow_dir = ARROW_STORAGE / DATASET_NAME
    if arrow_dir.exists():
        print(f"[build_arrow] reusing {arrow_dir}")
        return
    ARROW_STORAGE.mkdir(parents=True, exist_ok=True)
    builder = SimpleFinetuneDatasetBuilder(
        dataset=DATASET_NAME, windows=1, distance=1,
        prediction_length=PRED_LEN, context_length=CTX_LEN,
        patch_size=PATCH_SIZE, mode="S", storage_path=ARROW_STORAGE,
    )
    builder.build_dataset(file=CSV_PATH, dataset_type="wide", offset=None, freq="10min")
    print(f"[build_arrow] done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=["full", "freeze_ffn", "head_only"], default="head_only")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sanity", type=int, default=0)
    args = parser.parse_args()

    prepare_data()
    build_arrow_if_needed()

    n_train = np.load("data/alibaba_10min_train_v2.npy").shape[0]
    n_windows = max(1, (n_train - CTX_LEN - PRED_LEN) // PRED_LEN)
    steps_per_epoch = max(1, n_windows // BATCH_SIZE)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(100, total_steps // 5)
    print(f"[main] n_windows={n_windows}, steps/epoch={steps_per_epoch}, total={total_steps}, warmup={warmup_steps}")

    builder = SimpleFinetuneDatasetBuilder(
        dataset=DATASET_NAME, windows=n_windows, distance=PRED_LEN,
        prediction_length=PRED_LEN, context_length=CTX_LEN,
        patch_size=PATCH_SIZE, mode="S", storage_path=ARROW_STORAGE,
    )

    model = MoiraiFinetuneFixed(
        min_patches=2,
        min_mask_ratio=0.15,
        max_mask_ratio=0.5,
        max_dim=128,
        num_training_steps=total_steps,
        num_warmup_steps=warmup_steps,
        module=MoiraiModule.from_pretrained(MODEL_NAME),
        finetune_pattern=args.pattern,
        lr=args.lr,
    )

    train_ds = builder.load_dataset(model.train_transform_map)
    train_loader = Uni2tsDataLoader(
        dataset=train_ds, batch_size=BATCH_SIZE, batch_size_factor=1.0,
        cycle=True, num_batches_per_epoch=steps_per_epoch,
        shuffle=False, num_workers=4, collate_fn=None,
        pin_memory=True, drop_last=True,
    )

    ckpt_dir = Path(f"checkpoints/moirai_ft_{args.pattern}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer_kwargs = dict(
        max_epochs=args.epochs, accelerator="auto", devices=1,
        callbacks=[
            ModelCheckpoint(dirpath=str(ckpt_dir), filename="best-{epoch}-{train_loss:.4f}",
                            save_top_k=1, monitor="train_loss", mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_val=1.0, precision="32-true",
        log_every_n_steps=1, enable_progress_bar=True,
    )
    if args.sanity > 0:
        trainer_kwargs["fast_dev_run"] = args.sanity

    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloaders=train_loader)

    if not args.sanity:
        out_path = ckpt_dir / "final.pt"
        torch.save(model.module.state_dict(), out_path)
        print(f"[done] saved {out_path}")


if __name__ == "__main__":
    main()
