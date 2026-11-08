"""Split a round tape into time halves: <name>_a (0..T/2) and <name>_b (T/2..T, shifted so its first row is at 0).

Cancels in the second half for orders inserted before T/2 are dropped (the order is not in the book).
Usage: uv run python research-tools/split_round.py SRC.csv OUT_DIR NAME [--half SECONDS]
Unpacks SRC.csv.gz first if given a .gz file.
"""

from __future__ import annotations

import argparse
import gzip
import math
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out_dir')
    ap.add_argument('name')
    ap.add_argument('--half', type=float, default=None, help='split time in seconds (default: ceil(T_last)/2)')
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out_dir)
    if src.suffix == '.gz':
        dst = out / f'{a.name}_public.csv'
        if dst.exists():
            raise SystemExit(f'{dst} exists, refusing to overwrite')
        with gzip.open(src, 'rb') as fi, dst.open('wb') as fo:
            shutil.copyfileobj(fi, fo)
        src = dst

    with src.open(encoding='utf-8') as f:
        header = f.readline()
        last = None
        for line in f:
            if line.strip():
                last = line
    t_last = float(last.split(',', 1)[0])
    half = a.half if a.half is not None else math.ceil(t_last) / 2
    print(f'{src}: last time {t_last:.6f}, split at {half}')

    pa, pb = out / f'{a.name}_a.csv', out / f'{a.name}_b.csv'
    for p in (pa, pb):
        if p.exists():
            raise SystemExit(f'{p} exists, refusing to overwrite')

    na = nb = dropped = 0
    live_b: set[str] = set()
    t0 = None  # second half is shifted so that its first row (orphans included) is at 0
    with src.open(encoding='utf-8') as f, pa.open('w', encoding='utf-8', newline='') as fa, \
            pb.open('w', encoding='utf-8', newline='') as fb:
        f.readline()
        fa.write(header)
        fb.write(header)
        for line in f:
            if not line.strip():
                continue
            ts, rest = line.split(',', 1)
            t = float(ts)
            if t < half:
                fa.write(line)
                na += 1
                continue
            if t0 is None:
                t0 = t
            op, oid = rest.split(',', 2)[:2]
            if op == 'Insert':
                live_b.add(oid)
            elif oid not in live_b:
                dropped += 1
                continue
            fb.write(f'{t - t0:.6f},{rest}')
            nb += 1
    print(f'{pa.name}: {na} rows; {pb.name}: {nb} rows; dropped {dropped} orphan cancels')


if __name__ == '__main__':
    main()
