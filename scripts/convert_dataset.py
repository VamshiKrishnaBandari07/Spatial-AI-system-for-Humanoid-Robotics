import argparse
from pathlib import Path
from spatialmemory.datasets import convert_3rscan, convert_scannet, extract_scannet_sens

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="kind", required=True)
s = sub.add_parser("sens")
s.add_argument("source", type=Path)
s.add_argument("output", type=Path)
for name in ["3rscan", "scannet"]:
    s = sub.add_parser(name)
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--before", required=True)
    s.add_argument("--after", required=True)
    s.add_argument("--pair-metadata", type=Path, required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.kind == "sens":
    print(extract_scannet_sens(a.source, a.output))
elif a.kind == "3rscan":
    print(convert_3rscan(a.root, a.pair_metadata, a.before, a.after, a.output, a.split))
else:
    print(
        convert_scannet(a.root, a.before, a.after, a.pair_metadata, a.output, a.split)
    )
