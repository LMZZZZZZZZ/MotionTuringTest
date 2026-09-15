# PtrNet

PtrNet is a neural regressor that predicts a **human-likeness score** for a body
motion from its SMPL/body-pose trajectory. Given a sequence of body poses
extracted from a video or a generated motion, it returns how "human-like" the
motion is on the original 0-5 rating scale.

## Getting started

```bash
# install
pip install -r requirements.txt

# train (data paths default to ./data; override with --excel / --converted-pt-dir)
python scripts/train_ptr_net.py --run-name my_ptr_net --epochs 50

# evaluate the best checkpoint on the held-out test split
python scripts/eval_ptr_net.py --split test
```

## Data

The labels (human-likeness scores) are stored in an `.xlsx` mapping table and
the motions are stored as per-sample `.pt` trajectory files, named by an integer
`index`. See [`data/README.md`](data/README.md) for the exact format.

```bash
# reproduce a full run
python scripts/train_ptr_net.py \
  --run-name ptr_net_seed42 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.002 \
  --weight-decay 1e-4

# evaluate the selected model
python scripts/eval_ptr_net.py \
  --checkpoint checkpoints/ptr_net_seed42/net_best.pth \
  --split test
```

Evaluation reports RMSE / MAE / Spearman correlation on the 0-5 scale.

## Repository layout

```
ptrnet/model.py        PtrNet architecture
ptrnet/data.py         data loading, pose extraction, batching, train/val/test split
scripts/train_ptr_net.py  training entry point
scripts/eval_ptr_net.py   evaluation entry point
```
