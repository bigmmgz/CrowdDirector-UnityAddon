"""
Built-in smart-object templates, keyed by zone type.

For the first runnable milestone we instantiate smart objects deterministically from the
generated zones (robust, no fragile LLM policy JSON yet). Swapping this for LLM-generated
objects/policies later is a drop-in change — the SceneModel doesn't care where they come from.
"""

from .smart_object import SmartObject, Affordance, PolicyRule


def _cup(oid, zone):
    return SmartObject(
        id=oid, object_type="coffee_cup", zone_id=zone,
        states=["full", "empty"], state="full",
        affordances=[Affordance("drink", requires_state="full", changes_state_to="empty",
                                need_effects={"thirst": -30, "energy": 5})],
        policy=[PolicyRule(when={"state": "empty"}, emit="needs_clearing")])


def _table(oid, zone):
    # A table is a SURFACE, not a seat — it must never advertise "sit" (a table you sit AT is a chair; the
    # table itself is furniture). This idealized Stage-1 template used to conflate the two before the real
    # placement pipeline (scene_spec._furniture_only) drew that line; fixed here to match.
    return SmartObject(
        id=oid, object_type="table", zone_id=zone,
        states=["clean", "dirty"], state="clean",
        affordances=[],
        policy=[PolicyRule(when={"usage_count_gte": 3}, set_state="dirty", emit="needs_cleaning")])


def _chair(oid, zone):
    return SmartObject(
        id=oid, object_type="chair", zone_id=zone,
        states=["clean"], state="clean",
        affordances=[Affordance("sit", need_effects={"energy": 4, "stress": -5})])


def _dispenser(oid, zone):
    return SmartObject(
        id=oid, object_type="water_dispenser", zone_id=zone,
        states=["available"], state="available",
        affordances=[Affordance("drink", need_effects={"thirst": -22})])


def _toilet(oid, zone):
    return SmartObject(
        id=oid, object_type="toilet", zone_id=zone,
        states=["available"], state="available",
        affordances=[Affordance("relieve", need_effects={"bladder": -70})])


def _board(oid, zone):
    return SmartObject(
        id=oid, object_type="noticeboard", zone_id=zone,
        states=["available"], state="available",
        affordances=[Affordance("observe", need_effects={"curiosity": -25})])


# keyword in zone_type/id -> factory + how many
_RULES = [
    (("counter", "bar", "cafe", "kitchen"),           _cup,       2),
    (("water", "fountain"),                            _dispenser, 1),
    (("table", "seating", "lounge", "booth", "dining"),_table,     1),
    (("seating", "lounge", "booth", "dining", "chair"), _chair,     2),
    (("toilet", "restroom", "bathroom", "wc"),         _toilet,    1),
    (("activity", "notice", "stage", "board"),         _board,     1),
]


def build_objects(zones: list[dict]) -> list[SmartObject]:
    """Instantiate smart objects for a list of zone dicts ({id, zone_type, label, ...}). A zone can match
    MULTIPLE families (a seating zone needs both a table AND chairs — sit must resolve to the chair, never
    the table; see _table/_chair above), so every matching rule fires rather than stopping at the first."""
    objs: list[SmartObject] = []
    for z in zones:
        zid = z.get("id", "")
        key = f"{z.get('zone_type', '')} {zid} {z.get('label', '')}".lower()
        for keywords, factory, n in _RULES:
            if any(k in key for k in keywords):
                for i in range(n):
                    objs.append(factory(f"{zid}_{factory.__name__.strip('_')}_{i+1}", zid))
    return objs
