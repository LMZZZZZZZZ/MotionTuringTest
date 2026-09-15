import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptrnet.data import (
    SCORE_SCALE,
    add_gaussian_noise,
    load_records,
    make_batches,
    split_records,
)
from ptrnet.model import PtrNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PtrNet to regress human-likeness scores from body-pose trajectories"
    )
    parser.add_argument("--excel", type=Path, default=ROOT / "data" / "labels.xlsx",
                        help="Excel table with an `index` column and a human-score column")
    parser.add_argument("--converted-pt-dir", type=Path, default=ROOT / "data" / "converted_pt",
                        help="Directory of {index}.pt files with SMPL/pose data")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--run-name", type=str, default="ptr_net",
                        help="Subfolder name under checkpoint-dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Global std ratio for gaussian noise augmentation on the train split")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of valid samples used for training")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of valid samples used for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/val/test split")
    parser.add_argument("--select-best-on", type=str, default="train_loss",
                        choices=["train_loss", "val_loss"],
                        help="Metric used to select and save net_best.pth")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="If > 0, only use the first N valid rows (smoke test)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_checkpoint(network, save_path: Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = network.module.state_dict() if hasattr(network, "module") else network.state_dict()
    torch.save(state_dict, save_path)
    return save_path


def save_network(network, checkpoint_dir: Path, run_name: str, epoch_label) -> Path:
    save_filename = f"net_{epoch_label}.pth" if isinstance(epoch_label, str) else f"net_{epoch_label:03d}.pth"
    return save_checkpoint(network, checkpoint_dir / run_name / save_filename)


def evaluate(model, batches, device, desc="val"):
    """Returns (rmse, mae, spearman) on the 0-5 score scale."""
    model.eval()
    total = 0
    running_mse = 0.0
    running_mae = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, mask, _ in batches:
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            outputs = model(inputs, mask=mask) * SCORE_SCALE
            labels5 = labels * SCORE_SCALE
            running_mse += torch.sum((outputs.squeeze() - labels5) ** 2).item()
            running_mae += torch.sum(torch.abs(outputs.squeeze() - labels5)).item()
            all_preds.extend(outputs.squeeze().cpu().numpy().tolist())
            all_labels.extend(labels5.cpu().numpy().tolist())
            total += labels.size(0)
    rmse = math.sqrt(running_mse / total) if total else float("nan")
    mae = running_mae / total if total else float("nan")
    corr = float("nan")
    if len(all_preds) > 1 and len(set(all_preds)) > 1 and len(set(all_labels)) > 1:
        corr, _ = spearmanr(all_preds, all_labels)
    return rmse, mae, corr


def train_model(
    model, train_batches, val_batches, optimizer, scheduler, device,
    checkpoint_dir, run_name, epochs, save_interval, select_best_on,
):
    since = time.time()
    best = {"train_loss": (float("inf"), -1), "val_loss": (float("inf"), -1)}

    def _track(key, value, epoch):
        if best[key][0] <= value:
            return
        best[key] = (value, epoch)
        tag = "train" if key == "train_loss" else "val"
        path = save_network(model, checkpoint_dir, run_name, f"best_{tag}")
        print(f"[best {key}] updated at epoch {epoch} | value: {value:.6f} | {path}")

    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        sample_count = 0
        pbar = tqdm(total=len(train_batches), desc=f"train {epoch + 1}/{epochs}")

        for data in train_batches:
            inputs, labels, mask, _ = data
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad()

            if torch.isnan(inputs).any() or torch.isinf(inputs).any() or \
                    torch.isnan(labels).any() or torch.isinf(labels).any():
                pbar.update(1)
                continue

            outputs = model(inputs, mask=mask)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                pbar.update(1)
                continue

            mse = F.mse_loss(outputs, labels)
            output_var = outputs.var()
            var_loss = torch.tensor(0.0, device=outputs.device) \
                if torch.isnan(output_var) or torch.isinf(output_var) else (-0.05 * output_var)
            loss = mse + var_loss

            if torch.isnan(loss) or torch.isinf(loss):
                pbar.update(1)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            has_nan_grad = any(
                (p.grad is not None) and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad()
                pbar.update(1)
                continue

            optimizer.step()

            loss_value = float(loss.item())
            if math.isnan(loss_value) or math.isinf(loss_value):
                pbar.update(1)
                continue

            batch_size = labels.size(0)
            running_loss += loss_value * batch_size
            running_mse += torch.sum((outputs.squeeze() * SCORE_SCALE - labels * SCORE_SCALE) ** 2).item()
            sample_count += batch_size
            pbar.update(1)

        pbar.close()

        if sample_count == 0:
            print(f"Warning: epoch {epoch + 1} has no valid samples, skipping")
            scheduler.step()
            continue

        epoch_loss = running_loss / sample_count
        epoch_rmse = math.sqrt(running_mse / sample_count)
        print(f"train | loss(MSE): {epoch_loss:.6f} | rmse(0-5): {epoch_rmse:.6f}")

        _track("train_loss", epoch_loss, epoch + 1)

        if val_batches:
            val_rmse, val_mae, val_corr = evaluate(model, val_batches, device, desc="val")
            print(f"val   | rmse(0-5): {val_rmse:.6f} | mae(0-5): {val_mae:.6f} | spearman: {val_corr:.6f}")
            _track("val_loss", val_rmse, epoch + 1)

        if (epoch + 1) % save_interval == 0:
            path = save_network(model, checkpoint_dir, run_name, epoch + 1)
            print(f"checkpoint saved at epoch {epoch + 1}: {path}")

        scheduler.step()

    elapsed = time.time() - since
    print(f"\nTraining complete in {int(elapsed // 60)}m {int(elapsed % 60)}s")
    for key, (value, epoch) in best.items():
        print(f"best {key}: {value:.6f} (epoch {epoch})")

    best_value, best_epoch = best[select_best_on]
    src_suffix = "train" if select_best_on == "train_loss" else "val"
    src_path = checkpoint_dir / run_name / f"net_best_{src_suffix}.pth"
    if src_path.exists():
        model.load_state_dict(torch.load(src_path, map_location=device))
        best_path = save_checkpoint(model, checkpoint_dir / run_name / "net_best.pth")
        print(f"final {select_best_on} checkpoint: {best_value:.6f} (epoch {best_epoch}) -> {best_path}")
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print("loading records ...")
    records = load_records(
        excel_path=args.excel,
        converted_pt_dir=args.converted_pt_dir,
        max_samples=args.max_samples,
    )
    print(f"valid samples: {len(records)}")

    train_records, val_records, test_records = split_records(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    add_gaussian_noise(train_records, args.noise_std)

    train_batches = make_batches(train_records, args.batch_size, shuffle=False, seed=args.seed)
    val_batches = make_batches(val_records, args.batch_size, shuffle=False, seed=args.seed)
    print(f"split (seed={args.seed}): train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    model = PtrNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PtrNet params: {n_params / 1e6:.3f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_model(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        save_interval=args.save_interval,
        select_best_on=args.select_best_on,
    )


if __name__ == "__main__":
    main()
