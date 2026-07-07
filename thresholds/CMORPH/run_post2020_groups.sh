#!/usr/bin/env bash
# Build the post-2020 re-chunked eras as separate zarr groups in cmorph-s3-nc-v2,
# AFTER the root (501,1506) rebuild finishes. Single writer -> must be sequential.
# Zombie-safe wait (kill -0 succeeds on <defunct>; gate on process STATE instead).
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$D"
export GOOGLE_APPLICATION_CREDENTIALS="$D/coiled-data-e4drr_202505.json"
STORE="gs://cpc_awc/icechunk/cmorph-s3-nc-v2"

proc_live(){ local p="${1:-}" st; [[ -n "$p" ]] || return 1
  st="$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')" || return 1
  [[ -n "$st" && "$st" != Z* ]]; }

RB="${REBUILD_PID:-$(pgrep -f 'backfill_cmorph_icechunk.py.*cmorph-s3-nc-v2' | head -1)}"
if [[ -n "$RB" ]] && proc_live "$RB"; then
  echo "[chain] waiting for root rebuild PID $RB to finish ..."
  while proc_live "$RB"; do sleep 60; done
fi
while pgrep -f 'build_cmorph_icechunk.py.*cmorph-s3-nc-v2' >/dev/null 2>&1; do sleep 20; done
echo "[chain] root rebuild finished; building post-2020 groups into $STORE"

run(){ echo "[chain] == backfill $* =="; uv run backfill_cmorph_icechunk.py --store "$STORE" "$@" --batch-days 60; }

# cmorph_832 windows: 2020-07..2022-08 and 2023-06..2023-07
run --group cmorph_832 --start 20200701 --end 20220831
run --group cmorph_832 --start 20230601 --end 20230731
# cmorph_825 windows: 2022-09..2023-05 and 2023-08..present
run --group cmorph_825 --start 20220901 --end 20230531
run --group cmorph_825 --start 20230801 --end 20241231

echo "[chain] all post-2020 groups done. Store $STORE now has root + cmorph_832 + cmorph_825."
