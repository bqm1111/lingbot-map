#!/bin/bash
# Move rarely accessed directories from the NVMe disks to MINH1 (cold tier), leaving a symlink at the original path.
# Per directory: refuse if in use or already a link -> rsync copy -> checksum verify (dry-run -c must list nothing) -> remove original -> symlink.
# Every move is appended to artifacts/semantic_field_scaling/prep/cold_moves.jsonl.
set -u
DEST=/media/minh/MINH1/lingbot_b4_scaling/relocated
REC=/home/minh/workspace/lingbot-map_fork/artifacts/semantic_field_scaling/prep/cold_moves.jsonl
mkdir -p $DEST
for SRC in "$@"; do
  NAME=$(basename $SRC); PARENT=$(basename $(dirname $SRC)); TGT=$DEST/${PARENT}__$NAME
  if [ -L "$SRC" ]; then echo "skip $SRC: already a link"; continue; fi
  if [ ! -d "$SRC" ]; then echo "skip $SRC: missing"; continue; fi
  if lsof +D "$SRC" >/dev/null 2>&1; then echo "REFUSE $SRC: files open"; exit 2; fi
  BYTES=$(du -sb "$SRC" | cut -f1); FILES=$(find "$SRC" -type f | wc -l); T0=$(date +%s)
  rsync -rt "$SRC/" "$TGT/" || { echo "rsync failed for $SRC"; exit 3; }
  DIFF=$(rsync -rcn --out-format='%n' "$SRC/" "$TGT/" | grep -v '/$' | wc -l)
  NT=$(find "$TGT" -type f | wc -l); BT=$(du -sb "$TGT" | cut -f1)
  if [ "$DIFF" -ne 0 ] || [ "$NT" -ne "$FILES" ]; then echo "VERIFY FAILED $SRC: diff=$DIFF files $FILES vs $NT"; exit 4; fi
  rm -rf "$SRC" && ln -s "$TGT" "$SRC" || { echo "swap failed for $SRC"; exit 5; }
  echo "{\"source\": \"$SRC\", \"target\": \"$TGT\", \"bytes\": $BYTES, \"target_bytes\": $BT, \"files\": $FILES, \"checksum_diff_files\": $DIFF, \"seconds\": $(( $(date +%s) - T0 )), \"at\": \"$(date -Is)\"}" >> $REC
  echo "moved $SRC -> $TGT ($FILES files, $BYTES bytes)"
done
