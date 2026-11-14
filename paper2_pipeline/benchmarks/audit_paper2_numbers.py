"""Audit every numeric claim in paper2_access.tex against the results files.

WHY

Paper 3's ICRA draft was audited this way and it found a real error (0.418 printed
for 0.41749). A manuscript assembled over months from dozens of result files
accumulates stale numbers silently: a run is repeated, the JSON updates, the prose
does not. Reviewers find those; we would rather find them first.

METHOD

1. Harvest every number appearing in the .tex, with its surrounding sentence.
2. Harvest every number appearing anywhere in results/*.json, recursively.
3. For each paper number, ask whether a matching value exists in the results at a
   sensible tolerance.

A paper number with no match is NOT automatically an error - it may be a sample
size, a derived quantity, a percentage of something, or a figure from a cited work.
The output is a WORKLIST ordered by suspicion, not a verdict. Each flagged item is
then checked by hand.

    python benchmarks/audit_paper2_numbers.py
    python benchmarks/audit_paper2_numbers.py --show-matched
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

P2 = Path(__file__).resolve().parents[1]
TEX = P2 / "paper_writing/ieee_access/paper2_access.tex"
RESULTS = P2 / "results"

# Numbers that are structural rather than empirical: these are not claims about
# measurements and flagging them is noise.
IGNORE_CONTEXT = re.compile(
    r"\\(cite|ref|label|includegraphics|section|subsection|begin|end|"
    r"columnwidth|textwidth|linewidth|hspace|vspace|setlength|footnote)",
    re.I)
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")


def harvest_results() -> dict[float, list[str]]:
    """Every numeric leaf in every results JSON, mapped to where it came from."""
    found: dict[float, list[str]] = {}

    def walk(o, path, src):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k, src)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]", src)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            found.setdefault(round(float(o), 6), []).append(f"{src}:{path}")

    for f in sorted(RESULTS.glob("*.json")):
        try:
            walk(json.loads(f.read_text()), "", f.name)
        except Exception:
            continue
    return found


def matches(val: float, results: dict[float, list[str]]):
    """A paper value matches if some result value agrees once rounded to the
    paper's own precision, or as a percentage of a fraction (and vice versa)."""
    cands = []
    for rv, srcs in results.items():
        for scale in (1.0, 100.0, 0.01):
            r = rv * scale
            if r == 0:
                continue
            # agree to the number of decimals the paper actually printed
            dec = len(str(val).split(".")[1]) if "." in str(val) else 0
            if round(r, dec) == val and abs(r) > 1e-9:
                cands.append((srcs[0], rv, scale))
                break
        if len(cands) >= 3:
            break
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-matched", action="store_true")
    ap.add_argument("--min", type=float, default=0.01,
                    help="ignore values below this (section numbers, small ints)")
    a = ap.parse_args()

    results = harvest_results()
    print(f"harvested {len(results)} distinct numeric values from "
          f"{len(list(RESULTS.glob('*.json')))} results files\n")

    text = TEX.read_text()
    lines = text.split("\n")
    flagged, matched = [], 0

    for ln, line in enumerate(lines, 1):
        if line.lstrip().startswith("%") or IGNORE_CONTEXT.search(line):
            continue
        for m in NUM.finditer(line):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if val < a.min or val.is_integer() and val < 10:
                continue
            got = matches(val, results)
            if got:
                matched += 1
                if a.show_matched:
                    print(f"  OK   {raw:>10s}  L{ln}  <- {got[0][0]}")
            else:
                ctx = line.strip()
                flagged.append((ln, raw, ctx[:150]))

    print(f"{matched} values matched a results file")
    print(f"{len(flagged)} values NOT found in any results file "
          f"(worklist - verify each by hand):\n")
    for ln, raw, ctx in flagged:
        print(f"  L{ln:<5} {raw:>10s}   {ctx}")

    out = P2 / "results/audit_paper2_numbers.json"
    out.write_text(json.dumps(
        {"matched": matched,
         "unmatched": [{"line": l, "value": v, "context": c} for l, v, c in flagged]},
        indent=1))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
