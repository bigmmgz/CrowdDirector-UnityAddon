"""
scene_spec.py — OPEN-VOCABULARY scene composition (content side only; no behavior).

The renderer must handle ANY user prompt ("swimming pool", "library", "hotel lobby"), not a fixed list
of categories. This module turns a parsed SceneSpec into a concrete plan by:

  1. Composing reusable ThemeKits (asset/style modules), combining several for unknown scenes.
  2. Resolving each zone's required smart objects by AFFORDANCE (sit/eat/drink/wash/work/exercise/...),
     retrieving from the LimeZu catalog regardless of the asset's original theme — so a pool scene can
     pull showers (BathroomKit) + benches (SeatingKit) + snacks (FoodDrinkKit) with no dedicated PoolKit.

Behavior (LLM director -> DSAG/EventPatch -> rule/GNN/hybrid) is SEPARATE and unchanged: this only
produces zones + placed smart objects + affordances + nav hints that the behavior pipeline consumes.

Templates (artist floor plans) are an OPTIONAL shortcut for a kit, never the method: when no template
matches, the renderer generates a procedural layout from the same SceneSpec.

Pure stdlib; `python scene_spec.py` self-tests open-vocab prompts.
"""
from __future__ import annotations
import json, os, re
from collections import defaultdict
import prop_grammar   # generic relational prop grammar (functional clusters; replaces per-scene station rules)

# Which prop set to draw smart objects from — SWAPPABLE without replacing the other (A/B test LPC vs LimeZu):
#   ECGP_PROP_SET=limezu (default)  -> smart_object_catalog_limezu.json
#   ECGP_PROP_SET=lpc               -> smart_object_catalog_lpc.json   (built by assetgen.lpc_objects)
_ASSETS = os.environ.get("CROWDDIRECTOR_ASSETS",
                         os.path.join(os.path.dirname(__file__), "assets"))
_PROP_SET = os.environ.get("ECGP_PROP_SET", "limezu").strip().lower()   # LimeZu default (LPC set too limited —
                                                                        # 102 sliced props; switch back with =lpc)
_CATALOG = os.path.join(_ASSETS, f"smart_object_catalog_{_PROP_SET}.json")


# ── Catalog-derived affordance index (the open-vocab retrieval backbone) ─────────────
def _load_catalog():
    path = _CATALOG
    if not os.path.exists(path):                              # unknown/absent set -> fall back to LimeZu
        path = os.path.join(_ASSETS, "smart_object_catalog_limezu.json")
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    return raw.get("objects", raw) if isinstance(raw, dict) else raw

_OBJECTS = _load_catalog()

# FURNITURE-ONLY props: surfaces (tables/desks) are things items sit ON and people sit AT — not behavior
# targets themselves. Agents interact with the CHAIR (sit), the FOOD/CUP (eat/drink), the TOILET (relieve);
# the table is scenery + a mounting surface. Excluded from affordance retrieval; still placed as furniture.
def _furniture_only(o: dict) -> bool:
    pid = (o.get("prop_id") or "").lower()
    return "table" in pid and "vegetable" not in pid

_BY_AFF: dict[str, list[dict]] = defaultdict(list)
_BY_THEME: dict[str, list[dict]] = defaultdict(list)
_BY_ID: dict[str, dict] = {}
for _o in _OBJECTS:
    if not _furniture_only(_o):
        for _a in (_o.get("affordances") or []):
            _BY_AFF[_a].append(_o)
    _BY_THEME[_o.get("theme", "")].append(_o)
    _BY_ID[_o.get("prop_id", "")] = _o


# ── ThemeKits: reusable asset/style modules, NOT fixed scene categories ──────────────
# Each kit lists the catalog asset-themes it draws props/floors from and the affordances it typically
# provides. Cross-cutting kits (Seating/FoodDrink) draw by affordance across every theme.
KITS: dict[str, dict] = {
    "GymKit":       {"themes": ["gym"],                    "affordances": ["exercise"],            "template": "gym"},
    # cafe/shop use PROCEDURAL room layout (the ice-cream-shop template's tall back wall wasted ~40% of the
    # map and made agents look like they stood on it) — proper multi-room zones read better and block agents.
    "CafeKit":      {"themes": ["cafe"],                   "affordances": ["drink", "eat", "pay"], "template": None},
    "OfficeKit":    {"themes": ["office"],                 "affordances": ["work"],                "template": None},
    "HospitalKit":  {"themes": ["hospital"],               "affordances": ["rest", "receive"],     "template": None},
    "HomeKit":      {"themes": ["living_room", "bedroom"], "affordances": ["sit", "sleep", "watch", "rest"], "template": "home"},
    "ShopKit":      {"themes": ["shop"],                   "affordances": ["buy", "browse", "pay"],"template": None},
    "MuseumKit":    {"themes": ["museum"],                 "affordances": ["observe", "read"],     "template": "museum"},
    "BathroomKit":  {"themes": ["bathroom"],               "affordances": ["wash", "relieve"],     "template": None},
    "SeatingKit":   {"themes": [],                         "affordances": ["sit"],                 "template": None},
    "FoodDrinkKit": {"themes": [],                         "affordances": ["eat", "drink", "refill"], "template": None},
    "EmergencyKit": {"themes": [],                         "affordances": [],                      "template": None},  # exits/signage (event context)
    "ExteriorKit":  {"themes": [],                         "affordances": [],                      "template": None},  # outdoor style hint (no assets yet)
    "LeisureKit":   {"themes": [],                         "affordances": ["sit", "drink", "eat"], "template": None},
}

# Open-vocab keyword -> kit set. Used ONLY as a fallback when the LLM parser doesn't name kits, so the
# system degrades gracefully on prompts with no exact match by COMBINING kits.
_SCENE_KIT_HINTS: list[tuple[str, list[str]]] = [
    ("pool",       ["ExteriorKit", "BathroomKit", "SeatingKit", "FoodDrinkKit", "LeisureKit"]),
    ("swim",       ["ExteriorKit", "BathroomKit", "SeatingKit", "FoodDrinkKit", "LeisureKit"]),
    ("beach",      ["ExteriorKit", "SeatingKit", "FoodDrinkKit", "LeisureKit"]),
    ("library",    ["MuseumKit", "OfficeKit", "SeatingKit"]),
    ("hotel",      ["HomeKit", "OfficeKit", "SeatingKit", "FoodDrinkKit"]),
    ("lobby",      ["OfficeKit", "SeatingKit", "FoodDrinkKit"]),
    ("airport",    ["ShopKit", "SeatingKit", "FoodDrinkKit", "EmergencyKit"]),
    ("terminal",   ["ShopKit", "SeatingKit", "FoodDrinkKit", "EmergencyKit"]),
    ("mall",       ["ShopKit", "SeatingKit", "FoodDrinkKit"]),
    ("shopping",   ["ShopKit", "SeatingKit", "FoodDrinkKit"]),
    ("classroom",  ["OfficeKit", "SeatingKit"]),
    ("school",     ["OfficeKit", "SeatingKit"]),
    ("university", ["OfficeKit", "SeatingKit", "FoodDrinkKit"]),
    ("museum",     ["MuseumKit", "SeatingKit"]),
    ("gallery",    ["MuseumKit", "SeatingKit"]),
    ("gym",        ["GymKit", "FoodDrinkKit"]),
    ("fitness",    ["GymKit", "FoodDrinkKit"]),
    ("restaurant", ["CafeKit", "FoodDrinkKit", "SeatingKit"]),
    ("diner",      ["CafeKit", "FoodDrinkKit", "SeatingKit"]),
    ("cafe",       ["CafeKit", "SeatingKit"]),
    ("coffee",     ["CafeKit", "SeatingKit"]),
    ("pub",        ["CafeKit", "SeatingKit"]),
    ("bar",        ["CafeKit", "SeatingKit"]),
    ("nightclub",  ["CafeKit", "SeatingKit"]),
    ("club",       ["CafeKit", "SeatingKit"]),
    ("hospital",   ["HospitalKit", "SeatingKit"]),
    ("clinic",     ["HospitalKit", "SeatingKit"]),
    ("office",     ["OfficeKit", "FoodDrinkKit", "SeatingKit"]),
    ("workplace",  ["OfficeKit", "FoodDrinkKit", "SeatingKit"]),
    ("home",       ["HomeKit", "BathroomKit"]),
    ("house",      ["HomeKit", "BathroomKit"]),
    ("apartment",  ["HomeKit", "BathroomKit"]),
    ("park",       ["ExteriorKit", "SeatingKit", "FoodDrinkKit", "LeisureKit"]),
]

# Common zone-word -> the affordances that zone should offer (so required_smart_objects can be inferred
# even when the LLM only gives zone labels). Open-vocab: matched by keyword, not exhaustive.
_ZONE_AFFORDANCES: list[tuple[str, list[str]]] = [
    ("entrance", []), ("exit", []), ("lobby", ["sit"]), ("reception", ["receive", "sit"]),
    ("seating", ["sit"]), ("lounge", ["sit"]), ("waiting", ["sit"]), ("cafeteria", ["eat", "sit"]),
    ("counter", ["pay", "buy"]), ("bar", ["drink", "sit"]), ("snack", ["eat", "drink"]),
    ("kitchen", ["eat", "drink"]), ("dining", ["eat", "sit"]), ("cafe", ["drink", "eat"]),
    ("barista", ["drink", "eat", "pay"]), ("pastry", ["eat", "buy"]), ("bakery", ["eat", "buy"]),
    ("deli", ["eat", "buy"]), ("kiosk", ["buy", "eat", "drink"]), ("stall", ["buy", "eat", "drink"]),
    ("food", ["eat"]), ("coffee", ["drink"]), ("picnic", ["sit", "eat"]), ("canteen", ["eat", "drink", "sit"]),
    ("changing", ["wash"]), ("locker", ["wash"]), ("shower", ["wash"]), ("restroom", ["relieve", "wash"]),
    ("toilet", ["relieve", "wash"]), ("bath", ["wash", "relieve"]), ("wash", ["wash"]),
    ("desk", ["work"]), ("office", ["work"]), ("study", ["work", "read"]), ("computer", ["work"]),
    ("cardio", ["exercise"]), ("weights", ["exercise"]), ("workout", ["exercise"]), ("studio", ["exercise"]),
    ("gallery", ["observe", "read"]), ("exhibit", ["observe"]), ("reading", ["read", "sit"]),
    ("aisle", ["browse", "buy"]), ("shop", ["browse", "buy"]), ("store", ["browse", "buy"]),
    ("ward", ["rest"]), ("bed", ["sleep", "rest"]), ("living", ["sit", "watch"]), ("pool", ["sit"]),
]


def kits_for(scene_theme_text: str, style_tags=None, named_kits=None) -> list[str]:
    """Resolve the kit set for a scene. Prefer kits the LLM named; else combine kits by keyword hint;
    else fall back to a generic indoor kit set. Always returns a de-duplicated, non-empty list."""
    kits: list[str] = []
    for k in (named_kits or []):
        if k in KITS and k not in kits:
            kits.append(k)
    if not kits:
        hay = (scene_theme_text or "").lower()
        for kw, ks in _SCENE_KIT_HINTS:
            if kw in hay:
                for k in ks:
                    if k not in kits:
                        kits.append(k)
                break
    if not kits:
        kits = ["SeatingKit", "FoodDrinkKit", "OfficeKit"]      # generic indoor fallback
    return kits


def _style_theme_pref(kits: list[str]) -> list[str]:
    """Ordered asset-theme preference implied by the chosen kits (for style-consistent retrieval)."""
    pref: list[str] = []
    for k in kits:
        for th in KITS.get(k, {}).get("themes", []):
            if th not in pref:
                pref.append(th)
    return pref


def resolve_affordance(affordance: str, theme_pref: list[str], limit: int = 4) -> list[str]:
    """Prop_ids in the catalog offering `affordance`, ordered so style-matching themes come first.
    This is the open-vocab retrieval: ANY prop with the affordance qualifies, theme is only a tiebreak."""
    cands = _BY_AFF.get(affordance, [])
    if not cands:
        return []
    def rank(o):
        th = o.get("theme", "")
        return theme_pref.index(th) if th in theme_pref else len(theme_pref)
    ordered = sorted(cands, key=rank)
    return [o["prop_id"] for o in ordered[:limit]]


def zone_affordances(zone_label: str, zone_type: str = "") -> list[str]:
    hay = f"{zone_label} {zone_type}".lower()
    out: list[str] = []
    for kw, affs in _ZONE_AFFORDANCES:
        if kw in hay:
            for a in affs:
                if a not in out:
                    out.append(a)
    return out


def compose(scene_theme_text: str, zones: list[dict], style_tags=None,
            named_kits=None, required_smart_objects=None) -> dict:
    """Turn a parsed scene into a concrete content plan: kits used, and per-zone resolved smart objects
    (prop_ids) chosen BY AFFORDANCE. `zones` is the list of zone dicts (need id/label/zone_type)."""
    kits = kits_for(scene_theme_text, style_tags, named_kits)
    theme_pref = _style_theme_pref(kits)

    # Union of affordances this scene wants: explicit (from the LLM) + inferred per zone.
    plan_zones = []
    all_affs: set[str] = set(required_smart_objects or [])
    for z in zones:
        affs = zone_affordances(z.get("label", ""), z.get("zone_type", ""))
        for a in affs:
            all_affs.add(a)
        resolved = {a: resolve_affordance(a, theme_pref) for a in affs}
        plan_zones.append({
            "zone_id": z.get("id"), "label": z.get("label"),
            "affordances": affs,
            "smart_objects": {a: ids for a, ids in resolved.items() if ids},
            "missing": [a for a, ids in resolved.items() if not ids],
        })

    # Scene-level smart-object budget: ensure >=8 by back-filling generic affordances if sparse.
    placed = sum(len(v) for z in plan_zones for v in z["smart_objects"].values())
    return {
        "scene_theme_text": scene_theme_text,
        "kits": kits,
        "style_theme_pref": theme_pref,
        "zones": plan_zones,
        "total_smart_objects": placed,
        "missing_affordances": sorted(a for a in all_affs if not _BY_AFF.get(a)),
    }


def build_scene_objects(plan: dict, per_affordance: int = 1, scene_min: int = 8) -> list[dict]:
    """Flatten a compose() plan into a concrete PLACEMENT list Unity renders: one entry per smart object
    with its prop_id + affordances + need_effects + capacity, tagged with the zone to place it in. Back-fills
    extra candidates until at least `scene_min` objects exist (when the catalog allows)."""
    out: list[dict] = []
    seen: dict[str, set] = defaultdict(set)
    used_scene: set = set()          # prop_ids already placed ANYWHERE this scene (for cross-zone variety)

    def add(zid, pid, aff):
        if not pid or pid in seen[zid]:
            return
        seen[zid].add(pid)
        used_scene.add(pid)
        o = _BY_ID.get(pid, {})
        furniture = _furniture_only(o)                    # tables: placed + used as surfaces, never interacted with
        out.append({
            "smart_object_id": f"so_{len(out)}",
            "zone_id": zid, "prop_id": pid, "affordance": aff,
            "prop_class": o.get("prop_class"),
            "support": o.get("support"),        # FLOOR_STANDING/TABLETOP/COUNTERTOP/WALL_MOUNTED/SEAT/SURFACE
            "affordances": [] if furniture else o.get("affordances", []),
            "need_effects": {} if furniture else o.get("need_effects", {}),
            "capacity": o.get("capacity", 1),
            "interaction_duration": o.get("interaction_duration", 3),
            "availability": o.get("availability_default", "available"),
            "sprite_path": o.get("sprite_path"),
        })

    for z in plan["zones"]:
        for aff, ids in z["smart_objects"].items():
            # VARIETY: only place candidates not yet used ANYWHERE this scene, so a distinctive prop (e.g. a
            # ticket counter) is never cloned into every counter zone. A zone that can't get a UNIQUE prop
            # for an affordance simply skips it (its other affordances / the back-fill / decor fill the gap).
            for pid in [p for p in ids if p not in used_scene][:per_affordance]:
                add(z["zone_id"], pid, aff)
    # back-fill more DISTINCT candidates to reach the budget — only genuinely new prop_ids (never a repeat).
    for _ in range(3):
        if len(out) >= scene_min:
            break
        for z in plan["zones"]:
            for aff, ids in z["smart_objects"].items():
                for pid in [p for p in ids if p not in used_scene][:1]:
                    add(z["zone_id"], pid, aff)
    # SURFACE GUARANTEE: any zone receiving a TABLETOP item (cup/food/small drink — Unity rests these ON a
    # surface) must also contain a surface (table/desk/counter), else the item lands on the bare floor.
    tabletop_keys = ("food", "drink", "cup", "mug", "plate", "bread", "cheese", "dessert", "butter", "egg",
                     "fruit", "meal", "pie", "tray", "computer", "laptop", "monitor", "lamp", "tv")
    surfaces = [o for o in _OBJECTS
                if o.get("prop_class") == "surface" or any(k in (o.get("prop_id") or "")
                                                           for k in ("table", "desk", "counter"))]
    if surfaces:
        by_zone: dict[str, list] = defaultdict(list)
        for e in out:
            by_zone[e["zone_id"]].append(e)
        for zid, entries in by_zone.items():
            has_top = any(any(k in e["prop_id"].lower() for k in tabletop_keys) for e in entries)
            has_surface = any(e.get("prop_class") == "surface" or any(k in e["prop_id"].lower()
                              for k in ("table", "desk", "counter")) for e in entries)
            if has_top and not has_surface:
                s = next((o for o in surfaces if o["prop_id"] not in used_scene), surfaces[0])
                add(zid, s["prop_id"], (s.get("affordances") or ["sit"])[0])

    # ── STATION TEMPLATES: mandatory members per functional zone (never leave a station half-furnished) ──
    # A station is only complete with its required members; any missing one is ADDED (retrieved from the catalog
    # by id, then by affordance). Food / coffee / water / cups are VISIBLE consumable smart objects that reuse the
    # existing eat/drink actions + need_effects (no new behaviour, no retrain). PHASE A below then binds the
    # tabletop consumables ON their counter and the chairs AT their table.
    def _zent(zid): return [e for e in out if e["zone_id"] == zid]
    def _has(entries, *subs): return any(any(s in e["prop_id"].lower() for s in subs) for e in entries)
    def _first_id(*ids):
        for i in ids:
            if i in _BY_ID:
                return i
        return None
    def _first_aff(aff, exclude=()):
        for o in _BY_AFF.get(aff, []):
            if not any(x in o["prop_id"].lower() for x in exclude):
                return o["prop_id"]
        return None

    for z in plan["zones"]:
        zid = z["zone_id"]; label = (z.get("label") or "").lower(); affs = set(z["affordances"])
        ent = _zent(zid)
        is_restroom = bool({"relieve", "wash"} & affs) or any(k in label for k in ("restroom", "toilet", "bathroom", "washroom", "lavatory"))
        is_service  = bool({"drink", "eat"} & affs) or any(k in label for k in
                        ("cafe", "coffee", "bar", "counter", "kitchen", "diner", "snack", "cafeteria", "canteen", "bistro", "barista", "pastry"))

        # RESTROOM STATION — require a toilet (relieve) AND a sink fixture; a hygiene zone is incomplete without both.
        if is_restroom:
            if not _has(ent, "toilet", "urinal"):
                pid = _first_id("bathroom__toilet") or _first_aff("relieve")
                if pid: add(zid, pid, "relieve")
            if not _has(ent, "sink", "basin"):
                pid = _first_id("bathroom__sink", "cafe__sink") or _first_aff("wash")
                if pid: add(zid, pid, "wash")
            # optional mirror ATTACHED_TO the wall above the sink — placed only if the asset exists (audit: none yet)
            if not _has(ent, "mirror"):
                pid = _first_id("bathroom__mirror", "cafe__mirror")
                if pid: add(zid, pid, "observe")

        # CAFE SERVICE STATION — counter + coffee source + water source + visible food + cups (where drink is offered).
        if is_service:
            if not _has(ent, "counter", "table", "desk", "bar"):
                pid = _first_id("cafe__counter", "shop__checkout_counter", "museum__ticket_counter")
                if pid: add(zid, pid, "pay")
            ent = _zent(zid)
            if not _has(ent, "coffee_machine", "vending"):                       # coffee source
                pid = _first_id("cafe__coffee_machine", "office__coffee_machine")
                if pid: add(zid, pid, "drink")
            if not _has(ent, "water"):                                           # water source
                pid = _first_id("gym__water_station")
                if pid: add(zid, pid, "drink")
            if not _has(ent, "food", "snack", "pastry", "donut"):                # visible food
                pid = _first_id("cafe__food", "shop__food")
                if pid: add(zid, pid, "eat")
            if not _has(ent, "drink", "cup", "mug", "glass"):                    # cups/glasses
                pid = _first_id("cafe__drink", "shop__drink")
                if pid: add(zid, pid, "drink")

    # ── PHASE A: relational hierarchy (scene grammar) ────────────────────────────────────────────
    # Every child object gets an explicit parent link, per zone:
    #   ON    tabletop item -> a surface/counter it rests on        (food ON table)
    #   AT    seat          -> the table it belongs to              (chair AT table, faces it)
    #   WITH  service fixture (machine/register/urn) -> its counter (machine WITH counter station)
    # Unity places children RELATIVE to their parent (not independently); behavior faces the parent and the
    # eat/drink chains use the links (source counter -> seat -> table).
    def _is(e, *kws): return any(k in e["prop_id"].lower() for k in kws)
    rel_by_zone: dict[str, list] = defaultdict(list)
    for e in out:
        rel_by_zone[e["zone_id"]].append(e)
    for zid, entries in rel_by_zone.items():
        surf = [e for e in entries if _is(e, "table", "desk") or e.get("prop_class") == "surface"]
        ctr  = [e for e in entries if _is(e, "counter", "bar", "register")]
        seats = [e for e in entries if e.get("prop_class") == "seating" and not _is(e, "table")]
        hosts = surf + ctr
        ci = {}                                            # per-host round-robin so children spread out
        def _bind(child, host_list, relation):
            if not host_list:
                return
            i = ci.get(relation, 0); ci[relation] = i + 1
            h = host_list[i % len(host_list)]
            child["parent_id"] = h["smart_object_id"]; child["relation"] = relation
        for e in entries:
            if any(k in e["prop_id"].lower() for k in tabletop_keys):
                _bind(e, hosts, "on")
            elif e in seats:
                _bind(e, surf, "at")
            elif _is(e, "machine", "register", "urn", "dispenser", "fountain", "grinder") and ctr:
                _bind(e, ctr, "with")
    return out


def _distribute_required_functions(plan_zones: list[dict], kits: list[str]) -> None:
    """SCENE-LEVEL requirement satisfaction (algorithm step 1). A scene's selected kits imply required FUNCTIONS
    (e.g. CafeKit => drink/eat/pay). Any required function not already offered by SOME zone is assigned to the
    best-matching zone — a food/service zone for eat/drink/pay/buy, a seating zone for sit — so the grammar then
    instantiates it there. Generic: keyed on affordances + zone role words, never on the scene name."""
    required: set[str] = set()
    for k in kits:
        required |= set(KITS.get(k, {}).get("affordances", []))
    if not required:
        return
    present = {a for z in plan_zones for a in z["affordances"]}
    _SERVICE = ("counter", "bar", "cafe", "coffee", "kitchen", "snack", "food", "kiosk", "stall", "pastry",
                "bakery", "deli", "reception", "service", "canteen", "cafeteria", "diner", "bistro", "barista")
    _SEATING = ("seat", "lounge", "waiting", "rest", "dining", "picnic", "bench", "table", "sofa")
    for aff in sorted(required):
        if aff in present:
            continue
        target = None
        for z in plan_zones:
            lab = (z.get("label") or "").lower()
            if aff in ("eat", "drink", "pay", "buy", "refill") and any(k in lab for k in _SERVICE):
                target = z; break
            if aff in ("sit", "rest") and any(k in lab for k in _SEATING):
                target = z; break
        if target is None:                       # fallback: any furnished, non-entrance zone
            cand = [z for z in plan_zones
                    if z["affordances"] and not any(k in (z.get("label") or "").lower() for k in ("entrance", "exit", "lobby", "hall"))]
            target = cand[0] if cand else (plan_zones[0] if plan_zones else None)
        if target is not None and aff not in target["affordances"]:
            target["affordances"].append(aff)
            present.add(aff)


def build_scenespec(scene_theme_text: str, zones: list[dict], style_tags=None, named_kits=None,
                    scene_family: str = "indoor", agent_roles=None, event_context: str = "") -> dict:
    """Assemble the full open-vocabulary SceneSpec the renderer + behavior pipeline consume. Composes kits,
    resolves smart objects by affordance, and picks the primary kit's artist TEMPLATE if one exists (else
    the renderer builds a procedural layout). Behavior-agnostic — no agent decisions here."""
    plan = compose(scene_theme_text, zones, style_tags, named_kits)
    # Furnish via the SCENE-INDEPENDENT relational grammar: each zone's affordances expand into functional
    # clusters whose roles are filled by environment-appropriate assets (supports before dependents, ON/AT/
    # ATTACHED_TO relations). Projected straight into the existing smart_objects schema (GNN + action vocab
    # untouched). `build_scene_objects` remains as the legacy affordance-only path / fallback.
    _distribute_required_functions(plan["zones"], plan["kits"])
    env = prop_grammar.scene_env_tags(scene_theme_text, scene_family)
    smart_objects = prop_grammar.build_objects(plan["zones"], env, _OBJECTS)
    if not smart_objects:                          # safety net: never ship an unfurnished scene
        smart_objects = build_scene_objects(plan)
    kits = plan["kits"]
    # primary kit's optional artist template (shortcut); None -> procedural layout in Unity
    template = None
    for k in kits:
        t = KITS.get(k, {}).get("template")
        if t:
            template = t
            break
    return {
        "scene_theme_text": scene_theme_text,
        "scene_family": scene_family,
        "style_tags": list(style_tags or []),
        "selected_kits": kits,
        "style_theme_pref": plan["style_theme_pref"],
        "template": template,                       # optional shortcut, not required
        "zones": [{"id": z.get("id"), "label": z.get("label"), "zone_type": z.get("zone_type"),
                   "affordances": zone_affordances(z.get("label", ""), z.get("zone_type", ""))}
                  for z in zones],
        "required_affordances": sorted({a for z in plan["zones"] for a in z["affordances"]}),
        "smart_objects": smart_objects,
        "agent_roles": list(agent_roles or []),
        "event_context": event_context,
        "nav_requirements": {"connected": True, "min_door_width": 1},
    }


def _adjacent(a: dict, b: dict, eps: float = 0.05) -> bool:
    """Two gapless zone rects (corner x/y + w/h) share a walkable border."""
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    # vertical shared edge (touch in x, overlap in y)
    if (abs(ax1 - bx0) < eps or abs(bx1 - ax0) < eps) and min(ay1, by1) - max(ay0, by0) > eps:
        return True
    # horizontal shared edge (touch in y, overlap in x)
    if (abs(ay1 - by0) < eps or abs(by1 - ay0) < eps) and min(ax1, bx1) - max(ax0, bx0) > eps:
        return True
    return False


def attach_layout(spec: dict, laid_zones: list[dict]) -> dict:
    """Merge the grid layout engine's coordinates into the SceneSpec and derive zone_connections (which
    rooms share a walkable border). Called AFTER grid_layout.assign_grid_layout so coords are final."""
    by_id = {z.get("id"): z for z in laid_zones}
    for sz in spec["zones"]:
        z = by_id.get(sz["id"])
        if z:
            sz.update({k: z.get(k) for k in ("x", "y", "w", "h")})
    zs = [z for z in laid_zones if all(k in z for k in ("x", "y", "w", "h"))]
    conns = []
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            if _adjacent(zs[i], zs[j]):
                conns.append([zs[i]["id"], zs[j]["id"]])
    spec["zone_connections"] = conns
    return spec


if __name__ == "__main__":
    # OPEN-VOCAB acceptance test: prompts with NO dedicated category must still compose sensibly.
    tests = {
        "swimming pool": [
            {"id": "ent", "label": "Entrance", "zone_type": "entrance"},
            {"id": "chg", "label": "Changing Room", "zone_type": "service"},
            {"id": "shw", "label": "Showers", "zone_type": "service"},
            {"id": "pool", "label": "Pool Area", "zone_type": "activity"},
            {"id": "seat", "label": "Poolside Seating", "zone_type": "lounge"},
            {"id": "snk", "label": "Snack Bar", "zone_type": "counter"},
        ],
        "library": [
            {"id": "ent", "label": "Entrance", "zone_type": "entrance"},
            {"id": "read", "label": "Reading Room", "zone_type": "activity"},
            {"id": "desk", "label": "Study Desks", "zone_type": "activity"},
            {"id": "gal", "label": "Archive Gallery", "zone_type": "activity"},
            {"id": "wc", "label": "Restroom", "zone_type": "toilet"},
        ],
        "busy hotel lobby": [
            {"id": "rec", "label": "Reception", "zone_type": "counter"},
            {"id": "lng", "label": "Lounge Seating", "zone_type": "lounge"},
            {"id": "caf", "label": "Lobby Cafe", "zone_type": "counter"},
            {"id": "wc", "label": "Restroom", "zone_type": "toilet"},
        ],
        "students in a classroom": [
            {"id": "desks", "label": "Student Desks", "zone_type": "activity"},
            {"id": "front", "label": "Teacher Desk", "zone_type": "activity"},
            {"id": "wc", "label": "Restroom", "zone_type": "toilet"},
        ],
        "office workers during a fire alarm": [
            {"id": "desks", "label": "Open Office Desks", "zone_type": "activity"},
            {"id": "meet", "label": "Meeting Room", "zone_type": "activity"},
            {"id": "kit", "label": "Kitchenette", "zone_type": "counter"},
            {"id": "exit", "label": "Emergency Exit", "zone_type": "exit"},
        ],
    }
    for prompt, zones in tests.items():
        spec = build_scenespec(prompt, zones)
        print(f"\n=== '{prompt}' ===")
        print("  selected_kits:", spec["selected_kits"], " template:", spec["template"] or "PROCEDURAL")
        print("  required_affordances:", spec["required_affordances"])
        by_zone = defaultdict(list)
        for so in spec["smart_objects"]:
            by_zone[so["zone_id"]].append(f"{so['prop_id']}({so['affordance']})")
        for z in spec["zones"]:
            objs = by_zone.get(z["id"], [])
            print(f"    zone {z['label']:22s} affs={z['affordances']} objs={objs}")
        print(f"  total smart objects: {len(spec['smart_objects'])}"
              + ("  (>=8 OK)" if len(spec['smart_objects']) >= 8 else "  (<8!)"))
