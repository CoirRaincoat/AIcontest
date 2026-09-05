#!/bin/sh
# R2-B resumable 8-way ranged download of the official torch cu128 wheel.
# Rationale: runner kills background tasks after ~1h; pip --no-cache-dir cannot resume.
# Same official source host (download.pytorch.org/whl/cu128) as the authorized pip index;
# fetch method only deviates, recorded in the retry ledger + R2B_STATE.md.
# Idempotent: complete parts are skipped; incomplete parts restart (each part ~4-8 min).
set -u
URL='https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl'
SIZE=3461420395
CHUNK=432677550
N=8
DIR=/c/fire_envs/dl_torch
mkdir -p "$DIR"
pids=""
i=0
while [ "$i" -lt "$N" ]; do
  START=$((i * CHUNK))
  END=$((START + CHUNK - 1))
  [ "$END" -ge "$SIZE" ] && END=$((SIZE - 1))
  TARGET=$((END - START + 1))
  OUT="$DIR/part$i"
  HAVE=0
  [ -f "$OUT" ] && HAVE=$(stat -c%s "$OUT")
  if [ "$HAVE" -eq "$TARGET" ]; then
    echo "part$i complete ($TARGET bytes), skip"
  else
    echo "part$i fetching range $START-$END ($TARGET bytes)"
    curl -sS --retry 3 --retry-delay 5 -r "$START-$END" -o "$OUT" "$URL" &
    pids="$pids $!"
  fi
  i=$((i + 1))
done
rc=0
for p in $pids; do
  wait "$p" || rc=1
done
echo "CURL_PHASE_RC=$rc"
# verify all parts
i=0
ok=1
while [ "$i" -lt "$N" ]; do
  START=$((i * CHUNK))
  END=$((START + CHUNK - 1))
  [ "$END" -ge "$SIZE" ] && END=$((SIZE - 1))
  TARGET=$((END - START + 1))
  OUT="$DIR/part$i"
  HAVE=0
  [ -f "$OUT" ] && HAVE=$(stat -c%s "$OUT")
  if [ "$HAVE" -ne "$TARGET" ]; then
    echo "PART$i_INCOMPLETE have=$HAVE want=$TARGET"
    ok=0
  fi
  i=$((i + 1))
done
if [ "$ok" -eq 1 ]; then
  cat "$DIR/part0" "$DIR/part1" "$DIR/part2" "$DIR/part3" \
      "$DIR/part4" "$DIR/part5" "$DIR/part6" "$DIR/part7" > "$DIR/torch-2.8.0+cu128-cp311-cp311-win_amd64.whl"
  GOT=$(stat -c%s "$DIR/torch-2.8.0+cu128-cp311-cp311-win_amd64.whl")
  echo "ASSEMBLED=$GOT expect=$SIZE"
  [ "$GOT" -eq "$SIZE" ] && echo "ASSEMBLY_OK" || echo "ASSEMBLY_SIZE_MISMATCH"
  sha256sum "$DIR/torch-2.8.0+cu128-cp311-cp311-win_amd64.whl"
else
  echo "PARTS_INCOMPLETE"
fi
