import argparse
import json
import math
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptrnet.data import (
    SCORE_SCALE,
    load_records,
    make_batches,
    split_records,
)
from ptrnet.model import PtrNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PtrNet checkpoint on the train / val / test splits"
    )
    parser.add_argument("--excel", type=Path, default=ROOT / "data" / "labels.xlsx",
                        help="Excel table with an `index` column and a human-score column")
    parser.add_argument("--converted-pt-dir", type=Path, default=ROOT / "data" / "converted_pt",
                        help="Directory of {index}.pt files with SMPL/pose data")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "ptr_net" / "net_best.pth",
                        help="Model checkpoint to evaluate")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Which split to evaluate (must match the training split)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="If > 0, only use the first N valid rows (smoke test)")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "eval_results.json",
                        help="Path to save the detailed results (json)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> PtrNet:
    model = PtrNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def evaluate_split(model, batches, device: str):
    model.eval()
    total = 0
    running_mse = 0.0
    running_mae = 0.0
    all_preds = []
    all_labels = []
    all_records = []
    with torch.no_grad():
        for inputs, labels, mask, records in batches:
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            outputs = model(inputs, mask=mask) * SCORE_SCALE
            labels5 = labels * SCORE_SCALE
            running_mse += torch.sum((outputs.squeeze() - labels5) ** 2).item()
            running_mae += torch.sum(torch.abs(outputs.squeeze() - labels5)).item()
            all_preds.extend(outputs.squeeze().cpu().numpy().tolist())
            all_labels.extend(labels5.cpu().numpy().tolist())
            all_records.extend(records)
            total += labels.size(0)

    rmse = math.sqrt(running_mse / total) if total else float("nan")
    mae = running_mae / total if total else float("nan")
    corr, p_value = float("nan"), float("nan")
    if len(all_preds) > 1 and len(set(all_labels)) > 1:
        corr, p_value = spearmanr(all_preds, all_labels)

    details = [
        {
            "index": r.index,
            "name": r.name,
            "category": r.category,
            "pt_path": r.pt_path,
            "pred": round(float(p), 6),
            "label": round(float(l), 6),
        }
        for p, l, r in zip(all_preds, all_labels, all_records)
    ]
    return {
        "num_samples": total,
        "rmse": rmse,
        "mae": mae,
        "spearman_corr": corr,
        "spearman_pvalue": p_value,
        "details": details,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("loading records ...")
    records = load_records(
        excel_path=args.excel,
        converted_pt_dir=args.converted_pt_dir,
        max_samples=args.max_samples,
    )
    train_records, val_records, test_records = split_records(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    splits = {"train": train_records, "val": val_records, "test": test_records}
    print(f"split (seed={args.seed}): train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    print(f"loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    eval_splits = [args.split] if args.split != "all" else ["train", "val", "test"]
    summary = {}
    for name in eval_splits:
        batches = make_batches(splits[name], args.batch_size, shuffle=False, seed=args.seed)
        print(f"\n========== {name} split ({len(splits[name])} samples) ==========")
        result = evaluate_split(model, batches, device)
        print(f"RMSE: {result['rmse']:.4f} | MAE: {result['mae']:.4f} | "
              f"Spearman: {result['spearman_corr']:.4f} (p={result['spearman_pvalue']:.4f})")
        summary[name] = {k: v for k, v in result.items() if k != "details"}

    if args.output:
        args.output = Path(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "splits": summary,
            "details": {name: result["details"] for name in eval_splits},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nresults saved to {args.output}")


if __name__ == "__main__":
    main()
