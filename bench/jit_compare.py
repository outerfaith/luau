#!/usr/bin/env python3
# This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details

# Compares the interpreter, CodeGen and the LLVM JIT on the bench/tests suite. Every test reports its own run times
# (bench_support.lua drops the slowest runs, which is where JIT compilation lands), and the summary is the geometric
# mean speedup of each mode over the interpreter.

import argparse
import glob
import json
import math
import os
import re
import statistics
import subprocess

MODES = {
    "interp": [],
    "codegen": ["--codegen"],
    "jit": ["--jit"],
    "jit-cold": ["--jit-cold"],
}

RESULT = re.compile(r"\|><\|(.*?)((?:\|><\|[0-9.e+-]+)+)\|\|_\|\|")

benchdir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--vm", required=True, help="luau executable (build it with -c opt)")
parser.add_argument("--modes", default="interp,codegen,jit-cold", help="comma separated: " + ",".join(MODES))
parser.add_argument("--filter", default="", help="regex that test paths have to match")
parser.add_argument("--flags", default="", help="extra arguments for every run, e.g. --fflags=...")
parser.add_argument("--json", help="write raw medians to this file")
args = parser.parse_args()

modes = args.modes.split(",")
vm = os.path.abspath(args.vm)

tests = sorted(glob.glob("tests/*.lua", root_dir=benchdir) + glob.glob("tests/shootout/*.lua", root_dir=benchdir) + glob.glob("tests/sunspider/*.lua", root_dir=benchdir))
tests = [t for t in tests if re.search(args.filter, t)]


def run(test, mode):
    cmd = [vm, "-O2", *MODES[mode], *args.flags.split(), test]

    try:
        p = subprocess.run(cmd, cwd=benchdir, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None

    m = RESULT.search(p.stdout)
    if p.returncode != 0 or not m:
        return None

    return statistics.median(float(x) for x in m.group(2).split("|><|") if x)


results = {}

print(f"{'test':40}" + "".join(f"{m:>10}" for m in modes) + "".join(f"{m + '/x':>12}" for m in modes[1:]))

for test in tests:
    results[test] = {mode: run(test, mode) for mode in modes}
    row = results[test]
    line = f"{test.removeprefix('tests/'):40}" + "".join(f"{row[m]:10.2f}" if row[m] else f"{'fail':>10}" for m in modes)

    base = row[modes[0]]
    for m in modes[1:]:
        line += f"{base / row[m]:12.2f}" if base and row[m] else f"{'-':>12}"

    print(line, flush=True)

print()

for m in modes[1:]:
    speedups = [r[modes[0]] / r[m] for r in results.values() if r[modes[0]] and r[m]]
    if speedups:
        geomean = math.exp(sum(map(math.log, speedups)) / len(speedups))
        print(f"{m}: geomean speedup over {modes[0]} {geomean:.3f} across {len(speedups)} tests")

if args.json:
    with open(args.json, "w") as f:
        json.dump(results, f, indent=1)
