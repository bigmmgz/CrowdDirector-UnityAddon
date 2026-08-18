"""
prop_grammar.py — a SCENE-INDEPENDENT relational prop grammar.

Instead of café/restroom-specific rules, scenes are furnished by a generic hierarchy:

    Scene → Zone → FunctionalCluster → StructuralSupport → FunctionalObject
          → AssociatedObject → Consumable/MovableItem → InteractionSlot

driven entirely by DATA the catalog assets declare (roles, env_tags, support, requires,
optional_with, kind, capacity). A zone's required FUNCTIONS come from its affordances; each
function maps to a generic CLUSTER template that names the ROLES it needs. Roles are filled by
whichever asset matches the role AND the scene ENVIRONMENT — so a `drink` requirement instantiates
a coffee counter indoors, a water fountain / drinks stall outdoors, etc., with no scene-name logic.

Relations emitted (projected into the existing SceneSpec/ECGP schema via parent_id + relation, so the
GNN checkpoint and action vocabulary are untouched):
    PART_OF (zone), SUPPORTS/ON (item on a surface), AT (seat at a table),
    ATTACHED_TO (wall-mounted), WITH (fixture with a counter), HAS_SLOT (interaction slots).

Generation order (per the spec): supports before dependent objects; then consumables ON supports;
then wall-mounted; interaction-slot counts come from each asset's capacity.

Pure stdlib. `python prop_grammar.py` self-tests across café / restroom / park / market / campsite /
hospital / office.
"""
from __future__ import annotations
from collections import defaultdict


def nav_blocks(prop_class: str | None, support: str | None, kind: str | None) -> bool | None:
    """CATALOG metadata: does this object occupy floor an agent must route AROUND? Derived from the
    controlled prop_class / support / kind vocabulary (NOT sprite/name string-matching), so the renderer
    consults data instead of guessing. None = "no opinion" -> the client falls back to its own heuristic.
      - rests-on / hangs (TABLETOP, COUNTERTOP, WALL_MOUNTED)      -> False (never a floor obstacle)
      - SEAT / seating (agents mount it) and decor (walk-through)  -> False
      - SURFACE / FLOOR_STANDING / structural furniture            -> True  (solid floor body)
    """
    s = (support or "").upper()
    pc = (prop_class or "").lower()
    if s in ("TABLETOP", "COUNTERTOP", "WALL_MOUNTED"):
        return False
    if s == "SEAT" or pc == "seating" or pc == "decor":
        return False
    if s in ("SURFACE", "FLOOR_STANDING") or (kind or "").lower() == "structural":
        return True
    return None

# ── FUNCTION → cluster template (environment-INDEPENDENT) ────────────────────────────────────────
# Each scene function names the ROLES that satisfy it. `provider` = the object(s) that PROVIDE the
# function (one env-appropriate pick). `consumable` = a movable item the function produces (rests ON a
# surface — CONSUMES/PRODUCES). `needs_surface` = the function needs a SURFACE support in the zone.
# `optional` = OPTIONAL_WITH companions (added only if an env-appropriate asset exists). `seat_at_surface`
# = this function's seat associates AT a surface (dining/work).
FUNCTION_CLUSTERS: dict[str, dict] = {
    "relieve":  dict(provider=["toilet"],                          optional=["sink"]),
    "wash":     dict(provider=["sink"],                            optional=["mirror"]),
    "drink":    dict(provider=["drink_source"], consumable="drink_item", needs_surface=True),
    "eat":      dict(provider=["food_source", "surface"], consumable="food_item", needs_surface=True),
    "sit":      dict(provider=["seat"],                            optional=["table"]),
    "rest":     dict(provider=["rest_spot", "seat"]),
    "sleep":    dict(provider=["rest_spot", "bed", "seat"]),
    "work":     dict(provider=["desk", "surface"], seat_at_surface=True, consumable="work_item"),
    "read":     dict(provider=["exhibit", "display", "surface"],  optional=["seat"]),
    "observe":  dict(provider=["exhibit", "display"]),
    "buy":      dict(provider=["retail", "counter"]),
    "browse":   dict(provider=["retail", "display"]),
    "pay":      dict(provider=["counter", "retail", "surface"]),
    "receive":  dict(provider=["counter", "surface"],             optional=["seat"]),
    "exercise": dict(provider=["exercise"]),
    "refill":   dict(provider=["drink_source"]),
    "watch":    dict(provider=["exhibit", "display"],             optional=["seat"]),
}

# consumable role -> the concrete item role to retrieve for it (work_item maps to a tabletop device)
_CONSUMABLE_ROLE = {"drink_item": "drink_item", "food_item": "food_item", "work_item": "workstation_item"}

# Scene-category synonyms -> the catalog THEME they should prefer when several assets fit a role. This is a
# retrieval HINT (environment compatibility), not scene-name branching: it only influences WHICH themed asset
# fills a role, never WHETHER a role/cluster is created.
_CATEGORY_THEME = {
    "coffee": "cafe", "restaurant": "cafe", "diner": "cafe", "bistro": "cafe", "bar": "cafe", "pub": "cafe",
    "clinic": "hospital", "ward": "hospital", "store": "shop", "market": "shop", "supermarket": "shop",
    "gallery": "museum", "library": "museum", "exhibit": "museum", "school": "office", "classroom": "office",
    "workplace": "office", "hotel": "living_room", "house": "living_room", "home": "living_room",
    "lounge": "living_room", "fitness": "gym", "workout": "gym",
}
_OUTDOOR_HINTS = ("park", "beach", "pool", "garden", "plaza", "playground", "market", "fair", "festival",
                  "campsite", "camp", "outdoor", "street", "picnic", "stall", "yard", "field", "trail")


def scene_env_tags(scene_theme_text: str, scene_family: str = "indoor") -> set[str]:
    """Derive the scene's environment tags (indoor/outdoor/public + category themes) from its description.
    Used to pick environment-APPROPRIATE assets for each role — the only place scene text is consulted, and
    only to rank assets, never to branch cluster logic."""
    txt = (scene_theme_text or "").lower()
    tags: set[str] = set()
    if scene_family == "outdoor" or any(k in txt for k in _OUTDOOR_HINTS):
        tags |= {"outdoor", "public"}
    else:
        tags.add("indoor")
    for kw, theme in _CATEGORY_THEME.items():
        if kw in txt:
            tags.add(theme)
    for theme in ("cafe", "office", "hospital", "shop", "museum", "gym", "living_room", "bedroom", "bathroom"):
        if theme in txt:
            tags.add(theme)
    return tags


class _Builder:
    def __init__(self, objects: list[dict], env: set[str]):
        self.objects = objects
        self.env = env
        self.by_role: dict[str, list[dict]] = defaultdict(list)
        for o in objects:
            for r in (o.get("roles") or []):
                self.by_role[r].append(o)
        self.out: list[dict] = []
        self.used: set[str] = set()                 # prop_ids used anywhere (mild cross-zone variety)

    def pick(self, roles: list[str], allow_reuse: bool = False):
        """Best asset whose roles intersect `roles`, ranked by environment fit then novelty."""
        best, best_key = None, None
        for role in roles:
            for a in self.by_role.get(role, []):
                et = set(a.get("env_tags") or [])
                env_fit = len(self.env & et)
                # an asset with NO overlap with an outdoor scene is a poor fit but still allowed (last resort)
                penalty = 0 if (allow_reuse or a["prop_id"] not in self.used) else -5
                # earlier roles in the preference list win ties
                role_rank = -roles.index(role)
                key = (env_fit + penalty, role_rank)
                if best is None or key > best_key:
                    best, best_key = a, key
        return best

    def emit(self, asset: dict, zone_id: str, affordance: str, parent_id=None, relation=None) -> dict:
        o = asset
        furniture = o.get("kind") == "structural"
        e = {
            "smart_object_id": f"so_{len(self.out)}",
            "zone_id": zone_id,
            "prop_id": o["prop_id"],
            "affordance": affordance,
            "prop_class": o.get("prop_class"),
            "support": o.get("support"),
            "roles": o.get("roles", []),
            "kind": o.get("kind"),
            # a pure structural support isn't itself an interaction target; keep its affordances empty
            "affordances": [] if furniture else o.get("affordances", []),
            "need_effects": {} if furniture else o.get("need_effects", {}),
            "capacity": o.get("capacity", 1),
            "interaction_slots": o.get("capacity", 1),          # HAS_SLOT: agents that can use it at once
            "interaction_duration": o.get("interaction_duration", 3),
            "availability": o.get("availability_default", "available"),
            "sprite_path": o.get("sprite_path"),
            # NAV metadata (item 6): whether this body blocks navigation + an optional explicit footprint side
            # (world units). Derived from the controlled vocabulary; the renderer prefers this over heuristics.
            "blocks_navigation": nav_blocks(o.get("prop_class"), o.get("support"), o.get("kind")),
            "navigation_footprint": float(o["navigation_footprint"]) if o.get("navigation_footprint") else 0.0,
        }
        if parent_id:
            e["parent_id"] = parent_id
            e["relation"] = relation
        self.used.add(o["prop_id"])
        self.out.append(e)
        return e

    # A zone's structural surface, created on demand and shared by every consumable in that zone.
    def ensure_surface(self, zone_id: str, zone_surface: dict) -> dict | None:
        if zone_id in zone_surface:
            return zone_surface[zone_id]
        surf = self.pick(["counter", "table", "surface"], allow_reuse=True)
        if surf is None:
            return None
        e = self.emit(surf, zone_id, "pay")             # a structural support (placed FIRST — supports before deps)
        zone_surface[zone_id] = e
        return e

    def satisfy_requires(self, asset: dict, zone_id: str, zone_surface: dict):
        """Recursively satisfy an asset's REQUIRES relations (e.g. a consumable REQUIRES a surface)."""
        for role in (asset.get("requires") or []):
            if role == "surface":
                self.ensure_surface(zone_id, zone_surface)
            # 'wall' is always available (rooms have walls) — nothing to create.


def build_objects(plan_zones: list[dict], env: set[str], objects: list[dict]) -> list[dict]:
    """Furnish every zone by expanding its affordances into functional clusters and filling each cluster's
    roles with environment-appropriate assets. Returns the flat smart_objects list (existing schema) with
    supports emitted BEFORE the dependent items that rest on them."""
    b = _Builder(objects, env)
    zone_surface: dict[str, dict] = {}     # zone_id -> its ONE structural surface entry (shared by consumables)
    zone_seats: dict[str, list] = defaultdict(list)
    zone_have: dict[str, set] = defaultdict(set)   # provider roles already satisfied in a zone (dedup)

    def _register_surface(zid, entry):
        if zid not in zone_surface:
            zone_surface[zid] = entry

    # PASS 1 — structural supports + functional providers (fixtures/sources/seats/surfaces), per function.
    # Providers are DE-DUPLICATED per zone (a zone gets ONE counter, not one per pay/buy/receive), and all
    # structural surfaces CONSOLIDATE to a single shared zone surface (no surface pile-up).
    for z in plan_zones:
        zid = z.get("zone_id") or z.get("id")
        have = zone_have[zid]
        for aff in z.get("affordances", []):
            tmpl = FUNCTION_CLUSTERS.get(aff)
            if not tmpl:
                continue
            # this function's provider is already present in the zone -> reuse it (no duplicate)
            if not (set(tmpl["provider"]) & have):
                prov = b.pick(tmpl["provider"])
                if prov is not None:
                    if prov.get("kind") == "structural":
                        # consolidate: only ONE surface per zone; a later surface-provider just reuses it
                        if zid not in zone_surface:
                            e = b.emit(prov, zid, aff)
                            _register_surface(zid, e)
                            have |= set(prov.get("roles") or [])
                    else:
                        e = b.emit(prov, zid, aff)
                        have |= set(prov.get("roles") or [])
                        b.satisfy_requires(prov, zid, zone_surface)
                        if "seat" in (prov.get("roles") or []):
                            zone_seats[zid].append(e)
            # OPTIONAL_WITH companions (mirror by a sink, a table for a seating group) — only if novel + env-fit
            for opt_role in tmpl.get("optional", []):
                if opt_role in have:
                    continue
                comp = b.pick([opt_role])
                if comp is None:
                    continue
                if comp.get("kind") == "structural":
                    if zid in zone_surface:            # already have a surface — don't add a second
                        continue
                    ce = b.emit(comp, zid, (comp.get("affordances") or [aff])[0])
                    _register_surface(zid, ce)
                else:
                    ce = b.emit(comp, zid, (comp.get("affordances") or [aff])[0])
                have |= set(comp.get("roles") or [])

    # PASS 2 — dependent CONSUMABLES rest ON their zone's surface (create one if a support is still missing).
    for z in plan_zones:
        zid = z.get("zone_id") or z.get("id")
        for aff in z.get("affordances", []):
            tmpl = FUNCTION_CLUSTERS.get(aff)
            if not tmpl or not tmpl.get("consumable"):
                continue
            item_role = _CONSUMABLE_ROLE.get(tmpl["consumable"], tmpl["consumable"])
            item = b.pick([item_role], allow_reuse=True)
            if item is None:
                continue
            surf = b.ensure_surface(zid, zone_surface)
            parent = surf["smart_object_id"] if surf else None
            b.emit(item, zid, (item.get("affordances") or [aff])[0],
                   parent_id=parent, relation="on")

    # PASS 3 — seats ASSOCIATE AT the zone's surface (dining/work), facing it.
    for zid, seats in zone_seats.items():
        surf = zone_surface.get(zid)
        if not surf:
            continue
        for s in seats:
            s["parent_id"] = surf["smart_object_id"]
            s["relation"] = "at"

    # PASS 4 — WALL_MOUNTED items ATTACH_TO the wall (near an associated fixture if present).
    for e in b.out:
        if e.get("support") == "WALL_MOUNTED" and "parent_id" not in e:
            e["relation"] = "attached_to"

    # PASS 5 (V2.1 section 4) — an explicit, AUTHORED cluster_id shared by a zone's surface + its bound
    # seats/consumables, plus a PROVIDES edge from any real provider in that zone to the cluster. This is
    # the authoritative alternative to dsag_bridge.build_dining_clusters' runtime same-zone heuristic —
    # cluster membership is decided HERE, at scene-generation time, using the grammar's own PASS 1-3
    # bindings, not re-inferred later from object_type keywords + zone proximity. Only a zone that actually
    # got a shared surface forms a cluster (a zone with no surface has nothing to cluster); a provider is
    # identified as a non-structural, non-parented entry in the same zone (PASS 1 never parents a provider
    # to anything) that is not the surface itself.
    # Only a provider whose OWN role is genuinely a dining/consumable source counts as serving this cluster
    # — NOT every non-parented interactive object co-located in the zone. Found via a real round-trip test
    # (Gate 1): a zone requesting {sit, eat, drink} places BOTH a food AND a drink source, and a naive "any
    # provider in this zone" rule would ALSO capture something unrelated like a workstation/desk if the zone
    # ever requested "work" too — exactly the "second unrelated provider" case this gate tests for.
    _DINING_PROVIDER_ROLES = {"food_source", "drink_source", "coffee_source"}
    for zid, surf in zone_surface.items():
        cluster_id = f"cluster_{surf['smart_object_id']}"
        surf["cluster_id"] = cluster_id
        for e in b.out:
            if e.get("zone_id") != zid or e is surf:
                continue
            if e.get("parent_id") == surf["smart_object_id"]:       # bound seat ("at") or consumable ("on")
                e["cluster_id"] = cluster_id
            elif (e.get("kind") != "structural" and "parent_id" not in e
                  and set(e.get("roles") or []) & _DINING_PROVIDER_ROLES):
                # a real dining provider serving this zone's cluster (coffee machine/food source) —
                # PROVIDES the cluster, distinct from ASSOCIATED_WITH/ON/AT which describe furniture layout.
                e["provides_cluster_id"] = cluster_id
    return b.out


if __name__ == "__main__":
    import scene_spec as S
    scenes = {
        "cozy café":       [("Front Counter", "counter"), ("Seating", "lounge"), ("Restroom", "toilet")],
        "public restroom": [("Restroom", "toilet")],
        "sunny park":      [("Lawn", "activity"), ("Picnic Area", "lounge"), ("Snack Kiosk", "counter")],
        "outdoor market":  [("Food Stalls", "counter"), ("Seating", "lounge"), ("Restrooms", "toilet")],
        "campsite":        [("Campfire", "activity"), ("Picnic Tables", "lounge"), ("Water Point", "counter")],
        "hospital ward":   [("Reception", "counter"), ("Waiting Area", "lounge"), ("Ward", "activity"), ("Restroom", "toilet")],
        "open office":     [("Desks", "activity"), ("Kitchenette", "counter"), ("Restroom", "toilet")],
    }
    for name, zdefs in scenes.items():
        zones = [{"id": lbl.lower().replace(" ", "_"), "label": lbl, "zone_type": zt} for lbl, zt in zdefs]
        plan_zones = [{"zone_id": z["id"], "label": z["label"],
                       "affordances": S.zone_affordances(z["label"], z["zone_type"])} for z in zones]
        env = scene_env_tags(name)
        objs = build_objects(plan_zones, env, S._OBJECTS)
        print(f"\n=== {name}   env={sorted(env)} ===")
        by = defaultdict(list)
        for o in objs:
            tag = f"{o['prop_id']}[{o.get('support')}]"
            if o.get("parent_id"):
                tag += f"-{o['relation']}->{o['parent_id']}"
            by[o["zone_id"]].append(tag)
        for z in plan_zones:
            print(f"  {z['label']:16s} affs={z['affordances']}")
            for t in by.get(z["zone_id"], []):
                print(f"       {t}")
