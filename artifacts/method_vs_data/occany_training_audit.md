# OccAny training audit — what the released checkpoint actually is

**Audited 2026-10-20.** Repository state at audit time: HEAD
`148c46e935858457887b17944f5c5427147ee964`, branch `feature-completion-study`, 33 uncommitted
entries, `require_source_freeze()` passing on 54 files. Nothing was modified to produce this
report.

Source: `/home/minh/workspace/OccAny`. Evidence is the released checkpoint's own embedded
`args` namespace, read directly from `checkpoints/occany.pth`, plus the shipped shell scripts.
The brief's instruction not to assume a script produced the checkpoint turns out to matter:
**it did not.**

---

## 1. Verified facts

### 1.1 Checkpoint identity

`checkpoints/occany.pth`, 3,745,124,158 bytes.

| audit item | verified value | evidence |
|---|---|---|
| OccAny or OccAny+ | **OccAny** | `args.distill_model='SAM2_large'`, `args.pretrained='checkpoints/MUSt3R_512.pth'`. The `sh/train_occany_plus_*.sh` scripts instead pass `distill_model_name='SAM3', base_model='da3'`, which this checkpoint's args do not contain. |
| pretrained backbone | **MUSt3R_512.pth** (1,693,926,903 B, present locally) | `args.pretrained` |
| warm start | **also initialized from a prior OccAny run** | `args.pretrained_occany='/scratch/project/eu-25-92//tb_log/occany512_v1_reconOnly_samlarge/checkpoint-last.pth'` |
| file composition | **a merge of two checkpoints** | `is_merged_must3r_checkpoint=True`, `merged_from_gen_checkpoint='occany_must3r_gen.pth'`, `merged_from_decoder_checkpoint='occany_must3r_dec_recon_weigth.pth'` |
| top-level keys | `model`, `raymap_encoder` (74 entries), `gen_decoder` (371 entries), `optimizer`, `scaler`, `args`, `epoch`, 3 merge flags | direct read |

### 1.2 Training datasets, from `args.train_dataset`

All five, temporal sampling only:

| dataset | samples / epoch | sequence pickle |
|---|---:|---|
| Waymo (`WaymoSeqMultiView`) | 10,000 | `seq_exact_len_sub5_stride9_all.pkl` |
| DDAD (`DDADSeqMultiView`) | 10,000 | `seq_exact_len_sub5_stride9_all.pkl` |
| PandaSet (`PandasetSeqMultiView`) | 10,000 | `seq_exact_len_sub5_stride9_all.pkl` |
| ONCE (`OnceSeqMultiView`) | 10,000 | `seq_exact_len_sub5_stride9_all.pkl` |
| VKITTI2 (`VKittiSeqMultiView`) | 5,000 | `seq_exact_len_sub5_stride9.pkl` |
| **total** | **45,000** | |

Shared settings: `min_memory_num_views=5`, `max_memory_num_views=10`, `ray_map_prob=0.5`,
`z_far=50`, `aug_crop=128`, `aug_focal=0.9`, `reverse_seq=True`, `frame_interval=1`,
`resolution=[(512,336),(512,288),(512,256),(512,160)]`, `transform=SeqColorJitter`.

**Note the absence of surround arms.** The shipped `train_occany_recon.sh` adds four
`seq_surround_all.pkl` arms (Waymo/DDAD/PandaSet/ONCE, 4,000 each). This checkpoint's args
contain none.

### 1.3 Optimization budget

| quantity | value | derivation |
|---|---:|---|
| epochs | 100 | `args.epochs`, and `epoch=100` stored |
| per-GPU batch | 2 | `args.batch_size` |
| accumulation | 2 | `args.accum_iter` |
| world size | 16 | `args.world_size` |
| **effective batch** | **64** | 2 × 2 × 16 |
| steps / epoch | 703.1 | 45,000 / 64 |
| **optimizer steps** | **≈ 70,300** | 703.1 × 100 |
| **sample draws** | **4,500,000** | 45,000 × 100 |
| lr / min_lr / warmup | 7e-5 / 1e-6 / 3 epochs | `args` |
| weight decay, AMP, seed | 0.05, bf16, 0 | `args` |

**This is a lower bound on total training.** `pretrained_occany` shows the run was warm-started
from an earlier recon-only OccAny checkpoint whose own budget is not recorded in this file and
is **not verifiable from local evidence**.

### 1.4 Objectives

```
train_criterion = ConfLoss_multiview(Regr3D_multiview(L21, norm_mode='?avg_dis',
    loss_in_log=False, sky_loss_value=0, pose_loss_value=1, rgb_loss_value=0.0,
    perceptual_loss_value=0.0), alpha=0.2)
distill_criterion = DistillLoss(MSELoss(), use_conf=True)      # teacher: SAM2_large
```

**No semantic classification loss**, consistent with the earlier finding that OccAny assembles
semantics at inference (GroundingDINO + SAM2 + distilled features). Supervision is 3D pointmap
regression plus pose plus feature distillation.

Decoder: `Must3rDecoder(img_size=(512,512), enc_embed_dim=1024, embed_dim=768,
pointmaps_activation=LINEAR, pred_sam_features=True, feedback_type='single_mlp',
memory_mode='kv', ray_map_encoder_depth=6, use_multitask_token=True)`.

### 1.5 Its own validation setting

`args.test_dataset`: 206 @ `KittiSeqMultiView(..., min_memory_num_views=10,
max_memory_num_views=10, recon_view_idx=[0,2,4,6,8], ray_map_idx=[1,3,5,7],
resolution=[(512,160)], split='val')`.

So OccAny's own model selection used **10 memory views** — 5 reconstruction views and 4 raymap
views — not the 5-frame causal setting. Relevant to §4 of the brief: a 5-frame causal
evaluation is *not* the condition this checkpoint was selected under, and that must be stated
whenever a 5-frame number is reported for it.

### 1.6 The shipped script did not produce this checkpoint

`sh/train_occany_recon.sh` differs from `args` in at least four ways:

| | shipped script | released checkpoint |
|---|---|---|
| per-dataset temporal samples | 6,000 (VKITTI 5,000) | 10,000 (VKITTI 5,000) |
| surround arms | 4 arms, 4,000 each | none |
| `ray_map_prob` | 0.4 | 0.5 |
| criterion extras | no `sky_loss_value` / `rgb_loss_value` / `perceptual_loss_value` | all three present, set to 0 |
| total samples/epoch | 45,000 | 45,000 |

The totals coincide at 45,000 by arithmetic accident (29,000 temporal + 16,000 surround vs
45,000 temporal). **Do not cite the shipped script as the recipe for the released weights.**

### 1.7 Local pretrained dependencies

| file | bytes | role |
|---|---:|---|
| `MUSt3R_512.pth` | 1,693,926,903 | backbone initialization |
| `sam2.1_hiera_large.pt` | 898,083,611 | distillation teacher during training |
| `groundingdino_swinb_cogcoor.pth` | 938,057,991 | inference-time semantics only |

---

## 2. Our side, for the data-efficiency comparison

Stated here so §5's claim can be audited, not to argue it.

| | OccAny (released) | ours (CompletionUNet) |
|---|---:|---:|
| trained parameters | full MUSt3R-scale decoder | **967 k** |
| optimizer steps | ≈ 70,300 (lower bound) | **6,000** |
| sample draws | 4,500,000 | **24,000** |
| effective batch | 64 | 4 |
| backbone initialization | MUSt3R_512 + prior OccAny run | **from scratch** |
| source datasets | 5 | 1 |

**But the comparison is not clean, and must not be presented as if it were.** Our *pipeline*
carries its own frozen pretrained dependencies that the CompletionUNet sits on top of:
LingBotMap (`checkpoints/lingbot-map/204754b/lingbot-map.pt`), MoGe-2 for metric scaling, and
Trident-H for semantics. The honest claim is about **task-specific completion training**, not
about total pretrained compute. A previous session measured our end-to-end inference cost at
≈2.9× fewer GPU-seconds per sample for semantics and ≈5× for geometry, with 2.6× more total
parameters of which 0.99 M are trained — those figures are log-based and carry their own
caveats.

---

## 3. Unverified or unavailable

* **The warm-start budget.** `occany512_v1_reconOnly_samlarge/checkpoint-last.pth` is not
  present locally and its step count, dataset mixture and duration are unknown. Total OccAny
  training is therefore **unbounded above** by this audit.
* **Unique scenes, frames and training examples per dataset.** `args` records *sample draws
  per epoch*, which is a sampling weight, not a corpus size. The underlying counts live in the
  `seq_*.pkl` files on the authors' cluster; none is present locally. What a `..._all.pkl`
  includes (all cameras? all scenes?) is not determinable from the code alone.
* **Whether the merge sources were trained identically.** The file merges a *gen* and a *recon
  decoder* checkpoint; `args` describes one run. Which component each top-level key came from,
  and under what settings, is not recorded.
* **Published vs locally reproduced numbers.** This audit establishes provenance only. Any
  performance figure must still be tagged by origin per §4 of the brief.
* **Subset-training support.** The `--train_dataset` string is a free-form spec, so dropping
  datasets is mechanically trivial. Whether a *scientifically useful* subset run is affordable
  is a separate question, deferred to `matched_data_control_feasibility.md`.

---

## 4. Consequences for the experiment design

1. **A 5-frame causal number for OccAny is off-setting for that checkpoint** (§1.5). It must be
   labelled as such, and the published surround and 10-view numbers must not be mixed with it.
2. **"Fewer datasets" cannot be claimed from dataset *names* alone** (§3). We can defensibly
   claim fewer *task-specific completion training steps and draws* (§2), because those are
   verified on both sides. Scene and frame counts are not yet verified for OccAny and the claim
   must not lean on them.
3. **OccAny's total training is a lower bound.** Any efficiency claim should be stated against
   ≥70,300 steps, not =70,300.
4. **DDAD is confirmed as a genuine OccAny source dataset** at 10,000 draws/epoch, so an A2
   DDAD-only arm does address the brief's subset question — provided DDAD can support our
   supervision recipe, which `ddad_feasibility.md` settles.
