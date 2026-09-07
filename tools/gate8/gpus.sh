# Shared GPU pool for every Gate-8 script. Override per run, e.g.
#     GATE8_GPUS="0 1 2 3" tools/gate8/run_caches.sh
# Default deliberately excludes GPU 0, which is reserved for another user.
GATE8_GPUS=${GATE8_GPUS:-"1 2 3"}
read -r -a G8 <<< "$GATE8_GPUS"
g8() { echo "${G8[$(( $1 % ${#G8[@]} ))]}"; }      # g8 N -> the Nth pool GPU, wrapping
