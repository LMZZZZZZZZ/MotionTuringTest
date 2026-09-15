# Data layout


Please download the dataset from the
[Motion Turing Test project page](http://www.lidarhumanmotion.net/mtt/).

To train or evaluate PtrNet, prepare the following files under `data/`:

```text
data/
├── labels.xlsx                 # label mapping table (see below)
└── converted_pt/
    ├── 1.pt
    ├── 2.pt
    └── ...

# layout C: plain body pose
data = {"body_pose": torch.Tensor}                 # (T, 21, 3) or (T, 63)
```
Layout A: per-joint rotations
(T, 24, 3) or (T, 21, 3); the first 21 joints are used
data = {"pose": torch.Tensor}

Layout B: SMPL body-pose parameters (axis-angle)
data = {
    "smpl_params_global": {
        "body_pose": torch.Tensor
    }
}  # (T, 21, 3) or (T, 63)

Layout C: plain body pose
data = {"body_pose": torch.Tensor}  # (T, 21, 3) or (T, 63)
