"""
capability_compiler.py — the MINIMAL capability-constrained EventPatch compiler (frozen scope: office,
cafe, restaurant, party/lounge).

Pipeline stages (the LLM only ever PROPOSES; every stage after that is deterministic):

    instruction (+ optionally LLM-proposed ops)
        → schema validation of proposed ops
        → live-scene capability lookup (CapabilityIndex)
        → entity grounding / reference resolution
        → support-status decision
        → grounded primitive operations + affected scope

Statuses: EXECUTABLE / PARTIALLY_EXECUTABLE / UNSUPPORTED / AMBIGUOUS / INVALID_REFERENCE.

Grounding uses ONLY live scene capabilities: zones + functions, object roles/affordances, object
state/capacity, reachability, exits, event anchors. No open-world operations, no named-event branches —
the op vocabulary is exactly policy.vocab.EVENT_OP_KINDS_V1.

`propose_ops_from_text` is a small DETERMINISTIC pattern layer used by unit tests and as a no-LLM
fallback; the E1 compiler benchmark swaps in the LLM proposal while reusing every stage below it.
"""
import re
from dataclasses import dataclass, field

from .. import vocab as V


@dataclass
class CapabilityIndex:
    """Deterministic snapshot of what THIS scene can actually do (built from the live world)."""
    zones: dict = field(default_factory=dict)          # zid -> zone_function
    exits: list = field(default_factory=list)
    actions: set = field(default_factory=set)          # union of live affordance actions
    providers: dict = field(default_factory=dict)      # action -> [object_id]
    objects: dict = field(default_factory=dict)        # oid -> {"zone", "actions", "disabled", "full"}
    agents: dict = field(default_factory=dict)         # aid -> {"role", "zone"}
    groups: set = field(default_factory=set)           # role/group names targetable ("staff","visitors",...)
    reachable: dict = field(default_factory=dict)      # zid -> reachable-from-any-exit bool

    @classmethod
    def from_world(cls, world):
        ix = cls()
        for zid, z in world.zones.items():
            ix.zones[zid] = z.zone_function
        ix.exits = list(world.exit_zone_ids())
        ref = ix.exits[0] if ix.exits else next(iter(world.zones), None)
        for zid in world.zones:
            ix.reachable[zid] = world.zone_path(ref, zid)[1] if ref else False
        for oid, o in world.objects.items():
            acts = [a.action for a in getattr(o, "affordances", [])]
            disabled = {a for a in acts if world.overlays.is_disabled(oid, a)}
            ix.objects[oid] = {"zone": o.zone_id, "actions": acts,
                               "disabled": sorted(disabled),
                               "full": not o.capacity_ok()}
            for a in acts:
                ix.actions.add(a)
                if a not in disabled and o.capacity_ok():
                    ix.providers.setdefault(a, []).append(oid)
        for aid, a in world.agents.items():
            role = getattr(a, "social_role", "visitor")
            ix.agents[aid] = {"role": role, "zone": a.zone}
            ix.groups.add(role if role == "staff" else "visitors")
        ix.groups.add("all")
        return ix

    def alternatives(self, action, excluding=()):
        return [o for o in self.providers.get(action, []) if o not in excluding]


@dataclass
class CompileResult:
    support_status: str
    operations: list = field(default_factory=list)      # grounded primitive op dicts
    affected_scope: dict = field(default_factory=dict)  # {"kind": "role"|"all"|"agents", "value": ...}
    resolved_targets: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)      # missing capabilities / unresolvable references
    reason: str = ""

    def as_dict(self):
        return {"support_status": self.support_status, "operations": self.operations,
                "affected_scope": self.affected_scope, "resolved_targets": self.resolved_targets,
                "unresolved": self.unresolved, "reason": self.reason}


_OP_REQUIRED = {"op_kind", "target_type", "target_id"}


def validate_op_schema(op):
    """Stage 1: is the proposed op structurally valid against the closed vocabulary?"""
    if not isinstance(op, dict) or not _OP_REQUIRED.issubset(op):
        return False, "missing required op fields"
    if op["op_kind"] not in V.EVENT_OP_KINDS_V1:
        return False, f"unknown op_kind {op['op_kind']!r} (closed vocabulary)"
    if op["target_type"] not in ("zone", "object", "agent", "group"):
        return False, f"invalid target_type {op['target_type']!r}"
    return True, ""


def _resolve_zone(ref, ix):
    """Resolve a zone reference by id, then by unique name/function fuzzy match."""
    if ref in ix.zones:
        return ref, None
    toks = re.sub(r"[^a-z0-9 ]", " ", str(ref).lower()).split()
    cands = [z for z in ix.zones
             if all(t in z.lower().replace("_", " ") for t in toks)] if toks else []
    if not cands:
        cands = [z for z, fn in ix.zones.items() if str(ref).lower() == fn]
    if len(cands) == 1:
        return cands[0], None
    return None, ("ambiguous" if len(cands) > 1 else "unknown")


def _resolve_object(ref, ix):
    if ref in ix.objects:
        return ref, None
    toks = re.sub(r"[^a-z0-9 ]", " ", str(ref).lower()).split()
    cands = [o for o in ix.objects if all(t in o.lower().replace("_", " ") for t in toks)] if toks else []
    if len(cands) == 1:
        return cands[0], None
    return None, ("ambiguous" if len(cands) > 1 else "unknown")


def compile_ops(proposed_ops, world, scope_hint=None):
    """Stages 2-5 for a list of PROPOSED operations (from the LLM or the pattern layer): validate,
    ground every reference against the live CapabilityIndex, decide the support status, and emit only
    the grounded subset. Never invents operations; never silently substitutes a different entity for an
    unresolvable reference."""
    ix = CapabilityIndex.from_world(world)
    grounded, unresolved, reasons = [], [], []
    ambiguous = False
    for op in proposed_ops or []:
        ok, why = validate_op_schema(op)
        if not ok:
            unresolved.append({"op": op, "why": why})
            continue
        g = dict(op)
        if op["target_type"] == "zone":
            zid, err = _resolve_zone(op["target_id"], ix)
            if err:
                unresolved.append({"op": op, "why": f"zone reference {op['target_id']!r} {err}"})
                ambiguous |= err == "ambiguous"
                continue
            g["target_id"] = zid
        elif op["target_type"] == "object":
            oid, err = _resolve_object(op["target_id"], ix)
            if err:
                unresolved.append({"op": op, "why": f"object reference {op['target_id']!r} {err}"})
                ambiguous |= err == "ambiguous"
                continue
            g["target_id"] = oid
            act = op.get("action")
            if act and act not in ix.objects[oid]["actions"]:
                unresolved.append({"op": op, "why": f"{oid} does not afford {act!r}"})
                continue
        elif op["target_type"] == "group":
            if op["target_id"] not in ix.groups:
                unresolved.append({"op": op, "why": f"unknown group/role {op['target_id']!r}"})
                continue
        elif op["target_type"] == "agent" and op["target_id"] not in ix.agents:
            unresolved.append({"op": op, "why": f"unknown agent {op['target_id']!r}"})
            continue
        # capability-level checks for ops that require a live provider/route
        if g["op_kind"] == "object_attraction" and g["target_id"] not in ix.objects:
            unresolved.append({"op": op, "why": "object_attraction on non-object"})
            continue
        grounded.append(g)

    scope = scope_hint or {"kind": "all", "value": "all"}
    if not grounded and unresolved:
        has_unknown_ref = any("unknown" in u["why"] for u in unresolved)
        status = ("AMBIGUOUS" if ambiguous and not has_unknown_ref else
                  "INVALID_REFERENCE" if has_unknown_ref else "UNSUPPORTED")
        return CompileResult(status, [], scope, [], unresolved,
                             "; ".join(u["why"] for u in unresolved))
    if grounded and unresolved:
        return CompileResult("PARTIALLY_EXECUTABLE", grounded, scope,
                             [g["target_id"] for g in grounded], unresolved,
                             "supported subset grounded; missing: " +
                             "; ".join(u["why"] for u in unresolved))
    if grounded:
        return CompileResult("EXECUTABLE", grounded, scope,
                             [g["target_id"] for g in grounded], [], "")
    return CompileResult("AMBIGUOUS", [], scope, [], [], "no operations proposed")


# ── deterministic pattern layer (tests + no-LLM fallback; NOT the E1 LLM condition) ──────────────────
_ACTION_WORDS = {"drink": ("drink", "coffee", "water"), "eat": ("eat", "food", "snack", "cake", "pastry"),
                 "swim": ("swim", "pool"), "dance": ("dance", "dancing"),
                 "observe": ("show", "watch"), "rest": ("rest",)}


def propose_ops_from_text(instruction, world):
    """Small deterministic proposal layer for the benchmark's instruction patterns. Returns
    (proposed_ops, scope_hint, ambiguous_reason|None, unsupported_actions)."""
    ix = CapabilityIndex.from_world(world)
    t = instruction.lower()
    ops, unsupported = [], []
    scope = None

    if re.search(r"\bsend them\b|\bover there\b|\bthat one\b", t):
        return [], None, "unresolvable subject/target reference", []

    m = re.search(r"(gather|gathering|party|event|attention|head over|draw the crowd|center of)", t)
    if m:
        zone_ref = None
        for zid in ix.zones:
            if zid.replace("_", " ") in t:
                zone_ref = zid
                break
        ops.append({"op_kind": "zone_attraction", "target_type": "zone",
                    "target_id": zone_ref or "UNRESOLVED_ZONE", "magnitude": 60})
    if re.search(r"(broken|out of service|offline|maintenance|no more|closed the)", t):
        obj_ref = None
        for oid in ix.objects:
            stem = oid.rsplit("_", 1)[0].replace("_", " ")
            if stem in t:
                obj_ref = oid
                break
        act = next((a for a, words in _ACTION_WORDS.items() if any(w in t for w in words)
                    and a in ix.objects.get(obj_ref, {}).get("actions", [a])), None)
        ops.append({"op_kind": "disable_affordance", "target_type": "object",
                    "target_id": obj_ref or "UNRESOLVED_OBJECT", "action": act or "drink"})
    if re.search(r"(hazard|smoke|spill|emergency|evacuate)", t):
        zone_ref = next((z for z in ix.zones if z.replace("_", " ") in t), None)
        ops.append({"op_kind": "zone_hazard", "target_type": "zone",
                    "target_id": zone_ref or (next(iter(ix.zones)) if ix.zones else "?"),
                    "magnitude": 90})
    m = re.search(r"(staff|visitors|everyone|team)", t)
    if re.search(r"(leave|head home|to the exit|gather at the entrance|out first|go home)", t):
        who = "staff" if (m and m.group(1) in ("staff", "team")) else \
              "visitors" if (m and m.group(1) == "visitors") else "all"
        ops.append({"op_kind": "group_directive", "target_type": "group", "target_id": who,
                    "directive_target": ix.exits[0] if ix.exits else "exit"})
        scope = {"kind": "role", "value": who}

    for act, words in _ACTION_WORDS.items():
        if any(re.search(rf"\b{w}\w*\b", t) for w in words) and act not in ix.actions:
            unsupported.append(act)
    for word in ("fireworks", "cake"):
        if word in t and not any(word in o for o in ix.objects):
            unsupported.append(word)
    return ops, scope, None, sorted(set(unsupported))


def compile_instruction(instruction, world, proposed_ops=None, scope_hint=None):
    """Full pipeline. When proposed_ops is None the deterministic pattern layer proposes (tests /
    fallback); an LLM condition passes its own proposal in and reuses everything else."""
    unsupported = []
    if proposed_ops is None:
        proposed_ops, scope_hint2, ambiguous_reason, unsupported = propose_ops_from_text(instruction, world)
        scope_hint = scope_hint or scope_hint2
        if ambiguous_reason:
            return CompileResult("AMBIGUOUS", [], scope_hint or {"kind": "all", "value": "all"},
                                 [], [{"why": ambiguous_reason}], ambiguous_reason)
        if not proposed_ops and unsupported:
            return CompileResult("UNSUPPORTED", [], {"kind": "all", "value": "all"}, [],
                                 [{"why": f"no scene capability for: {', '.join(unsupported)}"}],
                                 f"missing capabilities: {', '.join(unsupported)}")
    res = compile_ops(proposed_ops, world, scope_hint=scope_hint)
    if unsupported and res.support_status == "EXECUTABLE":
        res.support_status = "PARTIALLY_EXECUTABLE"
        res.unresolved.append({"why": f"no scene capability for: {', '.join(unsupported)}"})
        res.reason = (res.reason + "; " if res.reason else "") + \
            f"missing capabilities: {', '.join(unsupported)}"
    return res
