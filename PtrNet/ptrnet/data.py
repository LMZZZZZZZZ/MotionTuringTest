from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import torch

SCORE_SCALE = 5.0

SCORE_COLUMN_CANDIDATES = [
    "clipped_avg_score",
    "avg_score",
    "score",
    "修剪后平均分",
    "修剪后平均分取整",
    "原始平均分",
]


@dataclass
class MotionRecord:
    """A single motion sample: a (T, 63) body-pose trajectory + a human-likeness score.

    The score is stored normalized to [0, 1] (original 0-5 rating divided by 5).
    """

    index: int
    pose: torch.Tensor  # (T, 63) float32
    score: float  # normalized to [0, 1]
    name: str = ""
    category: str = "unknown"
    pt_path: str = ""


def find_score_column(columns: Sequence[str]) -> Optional[str]:
    return next((c for c in SCORE_COLUMN_CANDIDATES if c in columns), None)


def extract_body_pose(pt_path: Path) -> torch.Tensor:
    """Extract a (T, 63) body-pose trajectory from a converted .pt file.

    Supported dict layouts:
      - data["pose"]: (T, 24, 3)  -> first 21 joints flattened to 63
      - data["smpl_params_global"]["body_pose"]: (T, 63) or (1, T, 21, 3) ...
      - data["body_pose"]: (T, 63) or (T, 21, 3)
    """
    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"unsupported pt content type at {pt_path}: {type(data)}")

    if "smpl_params_global" in data and isinstance(data["smpl_params_global"], dict):
        body_pose = data["smpl_params_global"].get("body_pose")
        if body_pose is not None:
            t = torch.as_tensor(body_pose, dtype=torch.float32)
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.reshape(t.shape[0], -1)
            return t[:, :63]

    if "body_pose" in data:
        t = torch.as_tensor(data["body_pose"], dtype=torch.float32)
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.reshape(t.shape[0], -1)
        return t[:, :63]

    if "pose" in data:
        t = torch.as_tensor(data["pose"], dtype=torch.float32)
        if t.ndim == 3 and t.shape[-1] == 3:
            if t.shape[1] < 21:
                raise ValueError(f"pose joint count < 21 in {pt_path}: {t.shape}")
            t = t[:, :21, :].reshape(t.shape[0], 63)
        elif t.ndim == 2 and t.shape[1] >= 63:
            t = t[:, :63]
        else:
            raise ValueError(f"unsupported pose shape in {pt_path}: {t.shape}")
        return t

    raise ValueError(f"cannot find body pose in {pt_path}")


def resolve_pt_path(row: pd.Series, converted_pt_dir: Path) -> Optional[Path]:
    idx = row.get("index")
    if pd.notna(idx):
        try:
            idx_int = int(float(idx))
            p = converted_pt_dir / f"{idx_int}.pt"
            if p.exists():
                return p
        except (TypeError, ValueError):
            pass

    input_path = row.get("input_path")
    if isinstance(input_path, str) and input_path.strip():
        p = Path(input_path)
        if p.suffix == ".pt" and p.exists():
            return p
        hmr = p.parent / "hmr4d_results.pt"
        if hmr.exists():
            return hmr

    return None


def _read_valid_rows(excel_path: Path, max_samples: int = 0) -> Tuple[pd.DataFrame, str]:
    if not excel_path.exists():
        raise FileNotFoundError(f"excel not found: {excel_path}")

    df = pd.read_excel(excel_path)
    score_col = find_score_column(df.columns)
    if score_col is None:
        raise ValueError(
            f"missing score column in {excel_path}, expected one of {SCORE_COLUMN_CANDIDATES}"
        )
    if "index" not in df.columns:
        raise ValueError(f"excel missing required column: index")

    df = df.copy()
    df = df[pd.to_numeric(df[score_col], errors="coerce").notna()].copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[(df[score_col] >= 0.0) & (df[score_col] <= SCORE_SCALE)].copy()

    if max_samples > 0:
        df = df.head(max_samples).copy()
    return df, score_col


def load_records(
    excel_path: Path,
    converted_pt_dir: Path,
    max_samples: int = 0,
) -> List[MotionRecord]:
    """Load and validate all (pose, score) samples from the excel + pt directory."""
    if not converted_pt_dir.exists():
        raise FileNotFoundError(f"converted pt dir not found: {converted_pt_dir}")

    df, score_col = _read_valid_rows(excel_path, max_samples=max_samples)
    records: List[MotionRecord] = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            pt_path = resolve_pt_path(row, converted_pt_dir)
            if pt_path is None:
                skipped += 1
                continue

            pose = extract_body_pose(pt_path)
            if pose.ndim != 2 or pose.shape[1] != 63:
                skipped += 1
                continue
            if pose.shape[0] == 0 or torch.isnan(pose).any() or torch.isinf(pose).any():
                skipped += 1
                continue

            records.append(
                MotionRecord(
                    index=int(float(row["index"])) if pd.notna(row.get("index")) else -1,
                    pose=pose,
                    score=float(row[score_col]) / SCORE_SCALE,
                    name=str(row.get("final_name", "")),
                    category=str(row.get("category", "unknown")),
                    pt_path=str(pt_path),
                )
            )
        except Exception:
            skipped += 1

    if not records:
        raise ValueError(
            "no valid training samples found; check that the excel `index` column "
            "matches {index}.pt files in the converted pt directory"
        )
    return records


def split_records(
    records: List[MotionRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[MotionRecord], List[MotionRecord], List[MotionRecord]]:
    """Deterministically split records into train / val / test (in excel order)."""
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"split too small for {n} samples: need >= 1 per split "
            f"(train={n_train}, val={n_val}, test={n_test})"
        )

    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def add_gaussian_noise(records: List[MotionRecord], noise_std_ratio: float) -> None:
    """Add global gaussian noise; std = ratio * global std of all pose values."""
    if noise_std_ratio <= 0:
        return
    all_poses = torch.cat([r.pose for r in records], dim=0)
    noise_std = float(all_poses.std().item()) * noise_std_ratio
    for r in records:
        r.pose = r.pose + torch.randn_like(r.pose) * noise_std


def pad_sequences(
    poses: List[torch.Tensor],
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length (T, 63) poses to a fixed length; returns padded + bool mask."""
    lengths = [p.shape[0] for p in poses]
    feat_dim = poses[0].shape[1]
    max_len = max_len or max(lengths)

    padded = torch.zeros(len(poses), max_len, feat_dim)
    mask = torch.zeros(len(poses), max_len)

    for i, pose in enumerate(poses):
        length = pose.shape[0]
        padded[i, :length] = pose
        mask[i, :length] = 1

    return padded, mask.bool()


def make_batches(
    records: Sequence[MotionRecord],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[MotionRecord]]]:
    """Group records into batches of (padded_pose, score, mask, records).

    The returned list can be passed directly to the training / eval loops.
    """
    order = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(order)

    batches = []
    for start in range(0, len(order), batch_size):
        batch_records = [records[i] for i in order[start:start + batch_size]]
        padded, mask = pad_sequences([r.pose for r in batch_records])
        scores = torch.tensor([r.score for r in batch_records], dtype=torch.float32)
        if torch.isnan(padded).any() or torch.isinf(padded).any():
            continue
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            continue
        batches.append((padded, scores, mask, batch_records))
    return batches
