#!/usr/bin/env python3
"""Parse a Beagle run log and extract, at the generation-G cut (default 100):
  wall_s   elapsed wall-clock when generation G completed (their own timestamp)
  gcl      the 'most accurate ever genome' as of generation G
Usage: beagle_gen100.py run.log [G]   -> prints "wall_s<TAB>gcl" (or FAIL:...)
"""
import re, sys

GEN = re.compile(r"^(\d+):(\d+):(\d+(?:\.\d+)?):\s+Generation\s+([\d,]+)\s+took")
BIRTHS = re.compile(r"Total births:\s+([\d,]+)")


def parse(path, cut=100):
    wall = None
    gcl = None
    births = 0
    pending_gcl = False
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if pending_gcl:
                if line.strip():
                    gcl = line.strip()
                    pending_gcl = False
                continue
            m = GEN.match(line)
            if m:
                g = int(m.group(4).replace(",", ""))
                if g == cut:
                    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    wall = h * 3600 + mn * 60 + s
                if g > cut:
                    break
                continue
            if line.startswith("Most accurate ever genome"):
                pending_gcl = True
            b = BIRTHS.search(line)
            if b:
                births = int(b.group(1).replace(",", ""))
    return wall, gcl, births


if __name__ == "__main__":
    path = sys.argv[1]
    cut = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    wall, gcl, births = parse(path, cut)
    if wall is None:
        print("FAIL:gen%d-not-reached" % cut)
    else:
        print(f"{wall:.3f}\t{births}\t{gcl or 'NO-GENOME'}")
