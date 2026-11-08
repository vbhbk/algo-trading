#!/usr/bin/env bash
# Cloud setup: install deps, unpack data/*.csv.gz next to the archives (keeps the .gz).
set -e
cd "$(dirname "$0")/.."
uv sync
for f in data/*.csv.gz; do
  [ "$f" = data/sample.csv.gz ] && [ -f data/sample.csv ] && continue
  out="${f%.gz}"
  [ -f "$out" ] || uv run python -c "import gzip,shutil,sys; shutil.copyfileobj(gzip.open(sys.argv[1],'rb'), open(sys.argv[2],'wb'))" "$f" "$out"
done
ls -la data/
