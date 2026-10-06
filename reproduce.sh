#!/usr/bin/env bash
# =====================================================================================================
#  Zero-shot semantic scene completion: the whole pipeline, raw data to final numbers, in one file.
#
#  Read it top to bottom: each STAGE says what it makes, what it reads, and where it writes.
#
#    KITTI-360 (train 0003/0007/0010, dev 0006)                   SemanticKITTI 08 / Occ3D-nuScenes val
#    ──────────────────────────────────────────                   ──────────────────────────────────────
#    1  frozen-model caches (LingBot, MoGe-2)                     9   OccAny released geometry (sample list,
#    2  LiDAR occupancy targets + 0006 eval masks                     labels, geometry comparison)
#    3  local-reset 5-frame LingBot clips (t-4..t)                10  causal 5-frame LingBot clips
#    4  hard map S/F (the geometry network's input)               11  predict → score → analyse → report
#    5  TRAIN geometry: CompletionUNet, 3 seeds    ─┐                 (frozen geometry seed 0, B0, B3 s0-2)
#    6  Trident-H tokens → voxel evidence           ├─ frozen ──►
#    7  delayed (future) semantic targets           │
#    8  TRAIN semantics: Gate 16 B3 head, 3 seeds  ─┘
#
#  Only stages 5 and 8 train anything; everything else is a frozen model or a hand-written rule.
#  Target datasets are opened only in stages 9-11.
#
#  Usage:   bash reproduce.sh                  # everything
#           bash reproduce.sh --from 11        # only the final evaluation
#           bash reproduce.sh --from 5 --to 8  # a range
#           bash reproduce.sh --dry-run        # print the commands, run nothing
#
#  Resumable: the per-sample builders skip outputs that already exist, so rerunning on this machine
#  re-scores the cached predictions instead of rebuilding them. For a true from-scratch run, start from
#  empty data roots (see the *_DATA_ROOT variables each tools/<pkg>/config.py reads).
#
#  Freezes: stages 5, 7, 8 and 11 hash-freeze their protocol or code before scoring, and every later
#  step re-checks those hashes. freeze_once below never overwrites an existing freeze record, so a
#  rerun cannot silently replace the original pre-registration.
#
#  Status: stage 11 was rerun end to end on 2026-10-06, and its predictions were byte-identical to the
#  stored ones. Stages 1-10 are assembled from each package's README and argument parser; they have
#  not been run as one script. Training (5, 8) is not bit-reproducible.
#  Per-stage details: tools/<package>/README.md. Results: REPRODUCE.md and artifacts/final_zero_shot/.
# =====================================================================================================
set -euo pipefail

# ---- environment ------------------------------------------------------------------------------------
REPO=/home/minh/workspace/lingbot-map_fork
PY=/home/minh/anaconda3/envs/cu128/bin/python                 # conda cu128, torch 2.7.1+cu128
TRI=/home/minh/workspace/third_party/Trident                   # Trident-H teacher (CLIP ViT-H + SAM ViT-H)
TPY=/home/minh/workspace/third_party/trident_env/bin/python
OCCANY=/home/minh/workspace/third_party/OccAny
OCCANY_OUT=/media/SSD1/MINH_DATASETS/occany_out
GPUS=(${GPUS:-0 1 2 3})
export PYTHONPATH=$REPO HF_HUB_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=12.0a                         # sm_120 needs this with nvcc 12.8 (CLAUDE.md)
cd "$REPO"

K360_TRAIN=(0003 0007 0010); K360_DEV=0006; K360_ALL=(0003 0006 0007 0010)
drive() { echo "2013_05_28_drive_$1_sync"; }

# ---- helpers ----------------------------------------------------------------------------------------
FROM=1; TO=11; DRY_RUN=0
while [[ $# -gt 0 ]]; do case $1 in
  --from) FROM=$2; shift 2;; --to) TO=$2; shift 2;; --dry-run) DRY_RUN=1; shift;;
  *) echo "unknown argument $1"; exit 2;; esac; done

stage() { (( $1 >= FROM && $1 <= TO )) || return 1; printf '\n==== STAGE %s: %s ====\n' "$1" "$2"; }
run()   { echo "+ $*"; (( DRY_RUN )) || "$@"; }
py()    { run "$PY" "$@"; }                                     # main env
pyc()   { run "$PY" -c "$1"; }                                  # main env, a python one-liner
teacher() { echo "+ (in $TRI) $TPY $*"; (( DRY_RUN )) || (cd "$TRI" && PYTHONPATH=$REPO:$TRI "$TPY" "$@"); }
waitall() { local bad=0; for p in $(jobs -p); do wait "$p" || bad=1; done; (( bad == 0 )) || { echo "a parallel job failed"; exit 1; }; }
freeze_once() { local rec=$1; shift; if [[ -e $rec ]]; then echo "= keeping existing freeze $rec"; else "$@"; fi; }
NG=${#GPUS[@]}

# =====================================================================================================
# SOURCE SIDE — KITTI-360 only. No SemanticKITTI or Occ3D file is opened here.
# =====================================================================================================

if stage 1 "frozen-model caches: LingBot stream, calibrated-FOV MoGe-2 depth, per-frame scale"; then
  # Dev drive 0006: SSCBench-KITTI-360 val labels (byte-range download of 0006 only), the 5-frame clip
  # manifest, and the calibrated-FOV MoGe-2 cache the local clips use for metric scale.
  #   -> /media/SSD1/MINH_DATASETS/{sscbench_kitti360, lingbot_gate5_2/{cache_lingbot,moge/B}}
  py tools/gate5_2/prepare_data.py --what all
  py tools/gate5_2/build_manifest.py
  py tools/gate5_2/cache_lingbot.py
  py tools/gate5_2/cache_scales.py --variant B
  # Training drives: LingBot streaming + MoGe-B depth + scale candidates, per frame.
  #   -> /media/SSD1/MINH_DATASETS/lingbot_gate8b/{stream,moge_b,scale}/k360_train
  for i in 0 1 2; do py tools/gate8b/stream_source.py --shard $i --shards 3 --device cuda:${GPUS[$i]} & done; waitall
fi

if stage 2 "raw-LiDAR occupancy targets (training supervision) and 0006 evaluation masks"; then
  # Occupied = LiDAR returns, free = ray interiors, open sky above the local roofline = free, else unknown
  # (core/supervision/rawtarget.py). drives7.py only registers drives and redirects records.
  #   -> /media/SSD1/MINH_DATASETS/lingbot_k360_7drives/{targets,eval_masks}
  W=tools/kitti360_data_scaling_7drives/drives7.py
  for d in "${K360_TRAIN[@]}"; do py $W core/run/build_targets.py --drive "$(drive $d)" --device cuda:${GPUS[0]}; done
  py $W core/run/build_targets.py --drive "$(drive $K360_DEV)" --stride 3 --device cuda:${GPUS[0]}
  for g in "${!GPUS[@]}"; do py tools/kitti360_data_scaling_7drives/build_eval_masks.py --shard $g --shards $NG --device cuda:${GPUS[$g]} & done; waitall
fi

if stage 3 "local-reset LingBot clips: frames t-4..t only, KV cache cleared per clip, MoGe-2 clip scale"; then
  #   -> /media/SSD1/MINH_DATASETS/lingbot_gate12_suf/local_lingbot (train, stride 1)
  #      /media/SSD1/MINH_DATASETS/lingbot_local_registration_audit/local_lingbot (0006, stride 3)
  for d in "${K360_TRAIN[@]}"; do py tools/gate12_suf/gen_local_clips.py --drive "$(drive $d)" --device cuda:${GPUS[0]}; done
  py tools/local_registration_audit/run_local_lingbot.py --drive "$(drive $K360_DEV)" --device cuda:${GPUS[0]}
fi

if stage 4 "hard map: each clip's depth rays → S (surface) and F (free) voxels on the 256x256x32 grid"; then
  #   -> /media/SSD1/MINH_DATASETS/lingbot_gate12_suf/{inputs,regions}
  for d in "${K360_ALL[@]}"; do py tools/gate12_suf/build_inputs.py --drive "$(drive $d)" --device cuda:${GPUS[0]}; done
  py tools/gate12_suf/build_regions.py --device cuda:${GPUS[0]}
fi

if stage 5 "TRAIN geometry: CompletionUNet on [S, 0, F] → occupancy, seeds 0/1/2 (~9 min each)"; then
  #   -> artifacts/baseline_reconstruction/{checkpoints/geo_seed*_step06000.pt, code_freeze.json}
  # Every later stage loads geometry seed 0 and checks its sha256 against code_freeze.json.
  T=tools/baseline_reconstruction/cli
  for s in 0 1 2; do py $T/train_current.py --seed $s --device cuda:${GPUS[$s]} & done; waitall
  py $T/eval_source.py --device cuda:${GPUS[0]}                # 0006 development evaluation
  freeze_once artifacts/baseline_reconstruction/code_freeze.json py $T/freeze_source.py   # refuses on FAIL
fi

if stage 6 "semantic evidence: Trident-H token maps, lifted to voxels where an accepted ray ends in S"; then
  #   -> /media/SSD1/MINH_DATASETS/lingbot_feature_first_3d/{trident_tokens,text_embeddings.npz}
  #      /media/SSD1/MINH_DATASETS/lingbot_gate15_semantic/evidence (~73 GB)
  for d in "${K360_ALL[@]}"; do teacher "$REPO/tools/feature_first_3d_semantics/extract_features.py" --drive "$(drive $d)" --device cuda:${GPUS[0]}; done
  teacher "$REPO/tools/feature_first_3d_semantics/encode_queries.py" --device cuda:${GPUS[0]}
  for d in "${K360_ALL[@]}"; do
    for g in "${!GPUS[@]}"; do py tools/semantic_completion/cli/build_evidence.py --drive "$(drive $d)" --shard $g --shards $NG --device cuda:${GPUS[$g]} & done; waitall
  done
fi

if stage 7 "delayed supervision: what future frames t+1..t+20 later observe, as soft semantic targets"; then
  # Future LiDAR projected into future images and read through Trident-H. Targets only; never an input.
  #   -> /media/SSD1/MINH_DATASETS/lingbot_delayed_semantic/{provenance,delayed_targets[_4s],train_cache,...}
  #      artifacts/delayed_semantic_pilot/ (frozen_*.json, pilot heads B1/B2/B2s, report)
  T=tools/delayed_semantic/cli; A=artifacts/delayed_semantic_pilot
  dsfreeze() { freeze_once $A/frozen_$1.json pyc "from tools.delayed_semantic import rules; rules.freeze('$1')"; }
  dsfreeze predeclaration
  for d in "${K360_ALL[@]}"; do py $T/build_provenance.py --drive "$(drive $d)" --device cuda:${GPUS[0]}; done
  dsfreeze reliability
  for d in "${K360_ALL[@]}"; do py $T/build_targets.py --drive "$(drive $d)" --device cuda:${GPUS[0]}; done
  py $T/run.py reliability --device cuda:${GPUS[0]}           # label-free gate: stops here if targets are unreliable
  dsfreeze addendum_A
  for d in "${K360_TRAIN[@]}"; do
    py $T/build_targets.py --drive "$(drive $d)" --variant 4s --device cuda:${GPUS[0]}
    py $T/run.py train_cache --drive "$(drive $d)" --device cuda:${GPUS[0]}
  done
  dsfreeze decision
  for m in B1 B2 B2s; do py $T/run.py train --method $m --device cuda:${GPUS[0]}; done   # pilot controls
  for g in "${!GPUS[@]}"; do py $T/run.py evaluate --shard $g --shards $NG --device cuda:${GPUS[$g]} & done; waitall
  py $T/run.py analyse; py $T/run.py verify --device cuda:${GPUS[0]}; py $T/run.py report
fi

if stage 8 "TRAIN semantics: Gate 16 B3 — pick among the 16 nearest evidence voxels, seeds 0/1/2"; then
  #   -> artifacts/gate16_candidate_selection/{checkpoints/B3_seed{0,1,2}_step03000.pt, code_freeze.json}
  X=tools/delayed_semantic/gate16/cli.py; A=artifacts/gate16_candidate_selection
  freeze_once $A/frozen_predeclaration.json py $X freeze --stage predeclaration
  py $X ceiling --device cuda:${GPUS[0]}
  freeze_once $A/frozen_ceiling.json py $X freeze --stage ceiling
  py $X train --variant B3 --device cuda:${GPUS[1]} & py $X train --variant B3geom --device cuda:${GPUS[2]} & waitall
  freeze_once $A/frozen_code.json py $X freeze --stage code
  for g in 0 1; do py $X evaluate --shard $g --shards 4 --device cuda:${GPUS[$g]} & done; waitall   # two at a time:
  for g in 2 3; do py $X evaluate --shard $g --shards 4 --device cuda:${GPUS[$g]} & done; waitall   # host memory
  py $X analyse; py $X report --device cuda:${GPUS[0]}; py $X verify --device cuda:${GPUS[0]}   # report writes code_freeze.json
fi

# =====================================================================================================
# TARGET SIDE — SemanticKITTI 08 and Occ3D-nuScenes val. Nothing below trains or tunes anything.
# =====================================================================================================

if stage 9 "OccAny released 5-frame geometry (defines the scored samples; supplies labels for comparison)"; then
  # OccAny's own repo and env; experiment list sh/exp_lists/occany.sh, id 0 = KITTI 5frames, 2 = nuScenes.
  # Needs its GroundingDINO box caches (OccAny README). Its semantic presets do not build on sm_120.
  #   -> $OCCANY_OUT/ssc_voxel_pred/OccAny_5frames_{kitti,nuscenes}512_rot60_vpi10_fwd3_sTrans2
  for id in 0 2; do
    echo "+ (in $OCCANY, occany_env) EXP_LIST=occany EXP_ID=$id SCRATCH=$OCCANY_OUT bash sh/eval_occany.sh"
    (( DRY_RUN )) || (cd "$OCCANY" && PATH=$OCCANY/../occany_env/bin:$PATH EXP_LIST=occany EXP_ID=$id SCRATCH=$OCCANY_OUT bash sh/eval_occany.sh)
  done
fi

if stage 10 "causal target clips: frames t-4..t through LingBot + MoGe-2, same frozen recipe as stage 3"; then
  #   -> /media/SSD1/MINH_DATASETS/lingbot_occany_comparison/{moge_b,clips/<dataset>/causal}
  T=tools/occany_comparison/cli
  py $T/build_moge.py --dataset occ3d --device cuda:${GPUS[0]}  # MoGe for Occ3D frames past the stored streams
  for g in "${!GPUS[@]}"; do py $T/build_clips.py --dataset occ3d --shard $g --shards $NG --device cuda:${GPUS[$g]} & done; waitall
  py $T/build_clips.py --dataset semantickitti --device cuda:${GPUS[0]}
fi

if stage 11 "final zero-shot evaluation: 801 SemanticKITTI + 4,219 Occ3D samples, B0 vs B3 (~1 h on 4 GPUs)"; then
  # Per sample: causal clip → hard map → CompletionUNet (geometry seed 0) → occupancy;
  # Trident-H tokens of t-4..t → evidence; B0 = nearest evidence label, B3 = Gate 16 head (seeds 0-2).
  #   -> /media/SSD1/MINH_DATASETS/lingbot_zero_shot_eval/{trident_tokens,predictions}
  #      artifacts/final_zero_shot/{scores,analysis.json,tables,figures,results_numbers.md,verification.md}
  A=artifacts/final_zero_shot
  py -m tools.zero_shot_eval.samples                           # the frames each sample reads
  for ds in semantickitti occ3d; do
    for g in "${!GPUS[@]}"; do teacher "$REPO/tools/zero_shot_eval/teacher/extract_tokens.py" --dataset $ds --shard $g --shards $NG --device cuda:${GPUS[$g]} & done; waitall
  done
  freeze_once $A/frozen_protocol.json   pyc "from tools.zero_shot_eval import freeze; freeze.freeze('protocol')"
  freeze_once $A/frozen_evaluators.json pyc "from tools.zero_shot_eval import freeze; freeze.freeze('evaluators')"
  for ds in semantickitti occ3d; do
    for g in "${!GPUS[@]}"; do pyc "from tools.zero_shot_eval.prediction.pipeline import run_shard; run_shard('$ds','cuda:${GPUS[$g]}',$g,$NG)" & done; waitall
  done
  for ds in semantickitti occ3d; do pyc "from tools.zero_shot_eval.run_eval import evaluate_dataset; evaluate_dataset('$ds')"; done
  pyc "from tools.zero_shot_eval import analysis; analysis.analyse()"      # B3−B0 block bootstrap, OccAny deltas
  pyc "from tools.zero_shot_eval import report; report.write_all()"
  pyc "from tools.zero_shot_eval import verify; verify.run('cuda:${GPUS[0]}')"
  echo; echo "Results: $A/results_numbers.md   (headline: $A/final_zero_shot_results.md)"
fi
