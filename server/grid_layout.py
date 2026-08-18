"""
grid_layout.py — Stardew-style gapless zone layout.

Claude designs *which* zones exist (types, labels, needs, agents); this module decides
*where* they go, partitioning the whole world rectangle into abutting rooms with ZERO gaps
and zero overlaps (a BSP / recursive-slice partition). The zone JSON contract is unchanged —
we just overwrite each zone's x/y/w/h with grid-aligned, tiling coordinates, so Unity renders
one continuous top-down map instead of scattered floating boxes.

World rectangle matches the existing contract: X in [-8, 8], Y in [-5, 5].
Pure stdlib; safe to import from the server or test standalone (`python grid_layout.py`).
"""

import random

# world rect as (x, y, w, h) with x,y = bottom-left corner (matches the zone contract)
WORLD = (-8.0, -5.0, 16.0, 10.0)
SNAP = 0.5           # snap split lines to this grid so rooms line up tidily
MIN_FRAC = 0.32      # a split never carves off less than this fraction of a side

# Relative space appetite by zone-type keyword — bigger zones get the bigger rooms,
# so seating/activity areas dominate and toilets/entrances stay small (Stardew-ish).
SIZE_PREF = {
    "table": 3.0, "seating": 3.0, "dining": 3.0, "lounge": 2.5, "activity": 2.5,
    "stage": 2.5, "field": 2.5, "balcony": 2.0, "counter": 1.7, "bar": 1.7,
    "vip": 1.5, "shelter": 1.5, "work": 1.7, "waiting": 1.5,
    "water": 1.0, "entrance": 1.0, "exit": 1.0, "toilet": 1.0, "restroom": 1.0,
}


def _snap(v):
    return round(v / SNAP) * SNAP


def _partition(rect, items, rng):
    """Recursively slice `rect` into abutting rooms, one per item, with room AREA proportional to each
    item's weight (its space appetite). `items` = list of (zone_index, weight). Splitting the longer side
    by the weight ratio keeps aspect ratios sane and makes big-appetite zones clearly bigger than small
    ones (a lounge dwarfs a toilet) instead of a uniform grid. Returns [(zone_index, rect), ...]."""
    if len(items) == 1:
        return [(items[0][0], rect)]
    x, y, w, h = rect
    total = sum(wt for _, wt in items)

    # split the item list into two contiguous groups near a weight-balanced point (with jitter)
    target = total * rng.uniform(0.42, 0.58)
    acc, k = 0.0, 1
    for i in range(len(items) - 1):
        acc += items[i][1]
        k = i + 1
        if acc >= target:
            break
    k = max(1, min(len(items) - 1, k))
    frac = sum(wt for _, wt in items[:k]) / total
    frac = min(1.0 - MIN_FRAC, max(MIN_FRAC, frac))

    vertical = w >= h                       # split the longer side -> avoids thin slivers
    if vertical and w < 2.0:
        vertical = False
    if (not vertical) and h < 2.0:
        vertical = True

    if vertical:
        sw = _snap(w * frac)
        sw = min(w - 1.0, max(1.0, sw)) if w >= 2.0 else w * frac
        return (_partition((x, y, sw, h), items[:k], rng) +
                _partition((x + sw, y, w - sw, h), items[k:], rng))
    else:
        sh = _snap(h * frac)
        sh = min(h - 1.0, max(1.0, sh)) if h >= 2.0 else h * frac
        return (_partition((x, y, w, sh), items[:k], rng) +
                _partition((x, y + sh, w, h - sh), items[k:], rng))


def _pref(zone):
    t = (zone.get("zone_type", "") + " " + zone.get("id", "") + " " + zone.get("label", "")).lower()
    best = 1.0
    for k, v in SIZE_PREF.items():
        if k in t:
            best = max(best, v)
    return best


def assign_grid_layout(zones, seed=None, world=None):
    """Overwrite each zone's x/y/w/h so the zones tile `world` (default: the module WORLD constant — the
    real Unity floor-plan extent) with no gaps/overlaps, with each room's AREA proportional to its space
    appetite (varied, design-like sizes rather than a uniform grid). Returns the same list.

    `world` (scale-gate item 5, "fixed density with larger environments"): an explicit (x, y, w, h) override
    lets a training-data generator lay the SAME zone count out over a LARGER rect, so a bigger crowd gets
    proportionally more floor space (density held roughly constant) instead of only ever getting packed
    denser into the fixed live-Unity extent. Every existing caller (live production, v1/v2/v21 dataset gen
    that doesn't pass `world`) is completely unaffected — this is additive."""
    if not zones:
        return zones
    rect = world if world is not None else WORLD
    rng = random.Random(seed)
    # Order by appetite so the biggest zones are carved off as large contiguous blocks first; this both
    # sizes rooms by appetite AND breaks the uniform-grid look.
    items = sorted(((i, _pref(zones[i])) for i in range(len(zones))),
                   key=lambda it: it[1], reverse=True)
    for zi, (x, y, w, h) in _partition(rect, items, rng):
        z = zones[zi]
        z["x"], z["y"], z["w"], z["h"] = round(x, 3), round(y, 3), round(w, 3), round(h, 3)
    return zones


def _selftest():
    """Verify the partition tiles the world with no gaps or overlaps (area + coverage)."""
    for n in range(1, 12):
        zones = [{"id": f"z{i}", "zone_type": t} for i, t in
                 enumerate(["table", "counter", "toilet", "entrance", "lounge", "water",
                            "activity", "vip", "shelter", "stage", "work"][:n])]
        assign_grid_layout(zones, seed=n)
        total = sum(z["w"] * z["h"] for z in zones)
        world_area = WORLD[2] * WORLD[3]
        # pairwise overlap check
        overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = zones[i], zones[j]
                if (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and
                        a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]):
                    overlap = True
        print(f"n={n:2d}  area={total:6.1f}/{world_area:.0f}  overlap={overlap}  "
              f"{'OK' if abs(total - world_area) < 0.01 and not overlap else 'FAIL'}")


if __name__ == "__main__":
    _selftest()
