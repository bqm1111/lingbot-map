# B5 is blocked on one path: KITTI-360 raw data is mounted somewhere the code cannot reach

**Status:** implementation and checks are done; target construction, training and evaluation cannot start.
**Nothing is running. No training step has run. No result exists.**

## What happened

`core/datasets/kitti360_splits.py:50` hardcodes

    KITTI360_ROOT = "/media/welf/MINH/datasets/kitti360/KITTI-360"

`/media/welf/MINH` is a symlink to `/media/minh/MINH1`, which is **not mounted**. The same physical disk (label `MINH`,
7.3 TB, it carries `lingbot_b4_scaling` and `lingbot_like_for_like`) is currently mounted at **`/media/albert/MINH`**,
and the data is intact and readable there:

    $ head -1 /media/albert/MINH/datasets/kitti360/KITTI-360/data_2d_raw/2013_05_28_drive_0003_sync/image_00/timestamps.txt
    2013-05-28 09:57:31.995373568

So the failure is purely a path, not missing or damaged data.

## Why I did not fix it myself

- **Repointing the symlink** (`ln -sfn /media/albert/MINH /media/welf/MINH`) was refused: `/media/welf` is a shared
  resource outside this repository.
- **Editing `KITTI360_ROOT`** would break the source freeze. That file is hashed in
  `tools/baseline_reconstruction/freeze.py` and recorded in `artifacts/baseline_reconstruction/code_freeze.json`;
  `require_source_freeze()` guards the frozen CompletionUNet geometry that B5 depends on. Editing it to add an
  environment override would change the hash and make the frozen geometry refuse to load — a worse problem than the one
  it solves. There is no existing environment seam for this root (`G8C1_DATA_ROOT` exists, for a different root).

## What unblocks it — any one of these

1. `ln -sfn /media/albert/MINH /media/welf/MINH`   *(one line, no repo change, restores the documented path)*
2. Remount the disk at `/media/minh/MINH1`, its original mount point.
3. Tell me to add an environment override to `KITTI360_ROOT` and re-freeze the baseline hash. This is the most
   invasive option and I would not choose it without being asked.

## What is unaffected

These are on the local NVMe and were verified present: the Gate 15 evidence caches (`flat_index`, `feature [E,1024]`
fp16, `count`), the frozen delayed-target caches, KITTI-360 poses and calibration
(`/media/SSD1/MINH_DATASETS/sscbench_kitti360/{data_poses,calibration}`), and the frozen registration record.

## If the mount cannot be restored

There is a fallback that needs no raw LiDAR or images: take the future teacher feature from the **Gate 15 evidence
cache of a later anchor**, moved into the anchor grid with GT poses (both local). For `t+5` the evidence window is
`t+1..t+5`, which is entirely in the future of `t`, so the target stays causally clean. It is a different construction
from the frozen LiDAR-projection rule, so it would have to be declared as such rather than presented as the same
supervision — which is why I am naming it here instead of quietly substituting it.
