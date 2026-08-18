"""
patch.py — the runtime GRAPH PATCH: a structured, typed edit that redirects an ongoing crowd.

This is the paper's core mechanism. A user's free-text interruption ("there's a party at the pool")
is turned by the LLM into a `ScenePatch` — a small list of typed ops drawn from a controlled
vocabulary that reshape the live scene: which zones pull agents, which affordances zones offer, how
needs drift, and role-conditioned priorities. Applying it changes the WORLD; agents then redirect by
reading the patched world through whatever policy is in play (rule engine or the learned GNN) — no
per-agent LLM. One LLM call → one patch → an arbitrarily large crowd redirects.

Pure/stdlib (the `dsag` engine stays dependency-free); the LLM that *generates* patches lives in the
server. Patches persist for `ttl` ticks (a party fades) and are auditable (typed + provenance).
"""

import os
import difflib
import logging
from dataclasses import dataclass, field

log = logging.getLogger("Director")   # same logger name/stream as crowd_director_server.py

# STEP 8 validation gate: while verifying CIVILIAN evacuation, suppress responder spawning (firefighters/
# medics/security) AND their role-directive routing, so no responder walks INTO the hazard and confuses the
# civilian-only assertions. Set ECGP_DISABLE_RESPONDERS=1. Off in normal play (responders spawn as usual).
_DISABLE_RESPONDERS = os.environ.get("ECGP_DISABLE_RESPONDERS", "0") == "1"
_RESPONDER_ROLES = ("firefighter", "medic", "paramedic", "security", "police", "responder", "guard")

# controlled patch-op vocabulary (what a user event may do to the world)
PATCH_OP_KINDS = (
    "zone_attraction",     # pull agents toward zone(s):        {zone|zone_function, delta}
    "enable_affordance",   # a zone now offers an action:       {zone|zone_function, action}
    "disable_affordance",  # a zone/global stops offering it:   {zone|zone_function?, action}
    "need_rate",           # change per-tick need DRIFT:        {need, delta, role?}
    "need_shift",          # one-shot need nudge on apply:      {need, delta, role?}
    "role_priority",       # bias a role toward zone(s):        {role, zone|zone_function, weight}
    "agent_directive",     # send ONE named agent to a zone or  {agent, zone} or {agent, target_agent}
                           #   to another agent (social/"X talk to Y", "Drew go to the bar")
    "agent_leave",         # send ONE named agent OUT of the building {agent} — "X goes home", they leave
    # ── graph-EDIT ops: structural scene mutations (the LLM composes; applied ONCE on patch creation) ──
    "remove_object",       # despawn a smart object:            {object} — "a glass broke/was cleared"
    "spawn_agent",         # NEW agents enter the scene:        {role, count, zone|zone_function}
                           #   — "firefighters arrive", "a medic comes in", "more customers show up"
    "role_directive",      # send EVERY agent of a role to a    {role, zone|zone_function, action?}
                           #   zone (responders → the hazard), stronger than role_priority
    # ── AMBIENT world ops (see policy.runtime.ambient) — these reach the policy through the graph only ──
    "object_attraction",   # point the crowd at ONE object:     {object, delta}
                           #   the free-sample tray, the busy counter — scoped tighter than a whole zone
    "zone_hazard",         # non-evacuation KEEPAWAY on zone(s): {zone|zone_function, delta, severity}
                           #   the trained "spill/keepaway" shape. NOT an evacuation — that is is_emergency.
)


@dataclass
class PatchOp:
    op: str
    zone: str = None            # zone id OR a keyword matched against id/type/label ("pool")
    zone_function: str = None   # match zones by function instead
    action: str = None          # affordance action (enable/disable)
    need: str = None
    delta: float = 0.0
    role: str = None            # role/personality keyword; None/"all" = everyone
    weight: float = 0.0
    agent: str = None           # agent_directive: the SUBJECT agent (name or id)
    target_agent: str = None    # agent_directive: go to this agent's location (name or id)
    object: str = None          # remove_object / object_attraction / object-scoped disable: object id or keyword
    count: int = 1              # spawn_agent: how many agents to spawn
    severity: float = 0.0       # zone_hazard: 0..1 keepaway strength (0 => the emitter's default)

    @classmethod
    def from_dict(cls, d: dict) -> "PatchOp":
        return cls(op=d.get("op", ""), zone=d.get("zone"), zone_function=d.get("zone_function"),
                   action=d.get("action"), need=d.get("need"),
                   delta=float(d.get("delta", d.get("weight", 0.0)) or 0.0),
                   role=d.get("role"), weight=float(d.get("weight", 0.0) or 0.0),
                   agent=d.get("agent"), target_agent=d.get("target_agent"),
                   object=d.get("object"), count=int(d.get("count", 1) or 1),
                   severity=float(d.get("severity", 0.0) or 0.0))


def validate_op(op: "PatchOp") -> str:
    """Structural validation for the ops that name a SPECIFIC subject — a directive with no subject silently
    matched nobody and did nothing (the reported bug: an 'agent_directive' with no `agent` and no recognizable
    target). Returns a human-readable rejection reason, or '' if the op is structurally sound. Loose/global
    ops (zone_attraction, need_rate, …) are unaffected — they are valid with no subject by design."""
    if op.op == "agent_directive":
        if not op.agent:
            return "agent_directive missing 'agent' (a directive must name WHO it's for)"
        if not (op.zone or op.target_agent or op.action):
            return f"agent_directive for '{op.agent}' has no zone, target_agent, or action — nothing to do"
    elif op.op == "agent_leave":
        if not (op.agent or op.role):
            return "agent_leave missing both 'agent' and 'role' (must name who leaves)"
    elif op.op == "role_directive":
        if not op.role:
            return "role_directive missing 'role'"
        if not (op.zone or op.zone_function):
            return f"role_directive for role '{op.role}' has no destination zone"
    elif op.op == "spawn_agent":
        if not op.role:
            return "spawn_agent missing 'role'"
    elif op.op == "remove_object":
        if not op.object:
            return "remove_object missing 'object'"
    return ""


# TINY deterministic safety net ONLY — unmistakable danger words, a last resort for when the upstream
# semantic intent resolver (event_intent.py) is absent. Do NOT grow this into a synonym list: end-of-day
# / go-home / closing / everyone-out are resolved SEMANTICALLY (LLM `intent` + MiniLM), not by keywords.
_EVAC_KEYWORDS = ("fire", "gas leak", "smoke", "bomb", "explosion", "evacuate", "active shooter")


@dataclass
class ScenePatch:
    event_type: str
    display_name: str = ""
    ops: list = field(default_factory=list)
    ttl: int = 25                       # ticks the patch stays active (then the world relaxes)
    behavior_hint: str = ""
    # ── independent intent fields (resolved by event_intent.py; NOT synonyms of each other) ──
    is_emergency: bool = False          # hard danger (fire/gas) — panic, immediate; NOT every global leave
    global_directive: str = None        # "leave" (end_of_day OR evacuation) | "avoid" (hazard) | "resume" (all_clear)
    priority: str = "normal"
    evacuate: bool = False              # every agent leaves the building — True for is_emergency OR safety net
    # ── leave-lifecycle cache (item C): agent_leave/personal-action targets are resolved ONCE, here, at
    # patch-activation time — never re-matched against the shrinking/changing live agent list on later ticks
    # (that was the "resolved to NOBODY" bug: a role like 'family' stopped matching once members had already
    # left). `leavers_locked` guards the one-time resolution; `resolved_leavers` is the frozen id list.
    leavers_locked: bool = False
    resolved_leavers: list = field(default_factory=list)
    # WHO CREATED THIS PATCH. "" / "user" = a human director typed it; "ambient" = the world simulation
    # (policy.runtime.ambient) produced it on a timer. The distinction is load-bearing, not cosmetic: the
    # deterministic party/gather branch seizes every eligible agent for a patch's whole ttl, which is
    # correct for a typed command and wrong for background world texture. See live_bridge._attracted_zones.
    origin: str = ""

    # populated by from_dict: [(op_dict, reason)] for ops that named PATCH_OP_KINDS but failed structural
    # validation (e.g. an agent_directive with no agent) — REJECTED, never silently downgraded to a vague
    # zone-only directive. The caller (crowd_director_server) logs these so a malformed command is visible.
    rejected_ops: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenePatch":
        et = d.get("event_type", "event")
        dn = d.get("display_name", et)
        is_emerg = bool(d.get("is_emergency"))
        gdir = d.get("global_directive")
        evac = bool(d.get("evacuate") or is_emerg)
        if not evac:                                     # tiny safety net only (semantics resolved upstream)
            blob = f"{et} {dn}".lower()
            evac = any(k in blob for k in _EVAC_KEYWORDS)
        if evac:                                         # a detected danger IS an emergency global-leave
            is_emerg, gdir = True, (gdir or "leave")
        ops, rejected = [], []
        for raw in d.get("ops", []):
            if raw.get("op") not in PATCH_OP_KINDS:
                continue
            op = PatchOp.from_dict(raw)
            reason = validate_op(op)
            if reason:
                rejected.append((raw, reason))
            else:
                ops.append(op)
        patch = cls(event_type=et, display_name=dn, ops=ops,
                   ttl=int(d.get("ttl", 25)), behavior_hint=d.get("behavior_hint", ""),
                   is_emergency=is_emerg, global_directive=gdir, priority=d.get("priority", "normal"),
                   evacuate=evac)
        patch.rejected_ops = rejected
        return patch

    def is_valid(self) -> bool:
        return all(o.op in PATCH_OP_KINDS for o in self.ops)


def is_evacuation(patches) -> bool:
    """True if any active patch is an EMERGENCY evacuation (hard danger — panic, immediate)."""
    return any(getattr(p, "evacuate", False) for p in patches)


def is_global_leave(patches) -> bool:
    """True if any active patch tells the WHOLE crowd to leave the building — evacuation (emergency)
    OR end_of_day (calm closing). Both empty the building; only the former is an emergency."""
    return any(getattr(p, "global_directive", None) == "leave" or getattr(p, "evacuate", False)
               for p in patches)


# ── matching helpers ─────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    """Normalise for lenient matching: lowercase, and treat _ - and spaces the same, so a patch that
    says 'dance_floor' still matches a zone labelled 'Dance Floor'."""
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


# Scene-independent venue-vocabulary synonyms — NOT zone IDs (those are per-scene and dynamic). Bidirectional:
# either side appearing in the query, matched against the zone's id/label containing the other side.
_ZONE_ALIASES = {
    "restroom": ("toilet", "bathroom", "wc", "washroom"), "bathroom": ("toilet", "wc"),
    "dining area": ("table", "dining", "seating"), "dining": ("table", "dine", "seating"),
    "bar": ("counter", "bar"), "lobby": ("entrance", "lobby", "reception"),
    "lounge": ("lounge", "seating", "sofa", "couch"),
}


def resolve_zone_reference(raw_reference: str, zones) -> tuple:
    """THE single zone-reference resolver (item D) — every id/label-scoped zone match routes through this,
    replacing ad hoc per-call substring checks. Scored, ordered tiers; zone_type/zone_function is DELIBERATELY
    excluded from the id/label tiers below — two zones can legitimately share a type (a real bug this fixes:
    a party aimed at the zone id 'lounge' was also matching a DIFFERENT zone 'window_seats' purely because
    that zone's zone_TYPE happened to also be 'lounge'). `zones` is any iterable of objects with .id/.label
    (and optionally .zone_function for the last-resort semantic tier) — works for any dynamically generated
    scene, no hard-coded names. Returns (best_zone_id_or_None, candidates) where candidates is
    [(zone_id, score, method), ...] sorted by score descending."""
    q = _norm(raw_reference)
    if not q:
        return None, []
    scored = []
    for z in zones:
        zid, label = _norm(getattr(z, "id", "")), _norm(getattr(z, "label", "") or "")
        if not zid:
            continue
        score, method = 0.0, ""
        if q == zid or q == label:                                          # 1. exact
            score, method = 1.0, "exact"
        else:
            if q in zid or q in label or (zid and zid in q) or (label and label in q):
                score, method = 0.85, "subphrase"                           # 2. token/subphrase
            else:
                id_tok, label_tok = set(zid.split()), set(label.split())
                q_tok = {t for t in q.split() if len(t) >= 4}                # distinctive words only
                if q_tok & (id_tok | label_tok):
                    score, method = 0.7, "token"
                else:
                    for word, alts in _ZONE_ALIASES.items():                 # 3. alias
                        blob = f"{zid} {label}"
                        if (word in q and any(a in blob for a in alts)) or \
                           (any(a in q for a in alts) and word in blob):
                            score, method = 0.5, "alias"
                            break
                    if not method:                                          # 4. fuzzy/typo
                        ratio = max(difflib.SequenceMatcher(None, q, zid).ratio(),
                                   difflib.SequenceMatcher(None, q, label).ratio() if label else 0.0)
                        if ratio >= 0.72:
                            score, method = ratio, "fuzzy"
                        else:                                                # 5. semantic function (last resort)
                            fn = " ".join(getattr(z, "zone_function", None) or [])
                            if q_tok & set(_norm(fn).split()):
                                score, method = 0.25, "semantic-function"
        if score > 0:
            scored.append((getattr(z, "id"), score, method))
    scored.sort(key=lambda t: -t[1])
    best = scored[0][0] if scored and scored[0][1] >= 0.25 else None
    return best, scored


def _log_zone_grounding(raw_reference, candidates, selected):
    top = [(zid, round(sc, 2)) for zid, sc, _ in candidates[:5]]
    method = next((m for zid, sc, m in candidates if zid == selected), "none")
    log.info(f"[ZoneGrounding] query={raw_reference!r} candidates={top} selected={selected!r} method={method}")


def _resolve_op_zone(op, all_zones) -> str:
    """Resolve op.zone against `all_zones` ONCE (cached on the op instance) and log [ZoneGrounding] on first
    resolution. `all_zones` must be the FULL zone set — resolving one op against every candidate together is
    what prevents the same op from matching two different zones that merely share a type/keyword.
    Cache key is the SORTED TUPLE OF ZONE IDS, not id(all_zones): callers rebuild `list(scene.zones.values())`
    fresh on every call (e.g. `_attracted_zones`/`_is_party_zone` run every tick), so an identity-based key
    never hit — the resolver (and its [ZoneGrounding] log) re-ran and re-logged on EVERY zone_matches call,
    once per zone per tick, spamming 20-30 identical lines/tick instead of once per op per tick."""
    if not op.zone:
        return None
    cache_key = tuple(sorted(getattr(z, "id", "") for z in all_zones))
    if getattr(op, "_zres_key", None) == cache_key:
        return op._zres_id
    best, candidates = resolve_zone_reference(op.zone, all_zones)
    op._zres_key, op._zres_id = cache_key, best
    _log_zone_grounding(op.zone, candidates, best)
    return best


def zone_matches(op: PatchOp, zone, all_zones=None) -> bool:
    if op.zone_function and op.zone_function in zone.zone_function:
        return True
    if op.zone:
        if all_zones is not None:
            # THE resolver (item D): one best zone for this op, never a type-keyword leak into a sibling zone.
            return _resolve_op_zone(op, all_zones) == zone.id
        # Legacy path (no zone list available at this call site) — UNCHANGED from before this rewrite
        # (includes zone_type), so any caller not yet threading all_zones keeps its original behavior. The
        # cross-zone-type-leak fix lives in the all_zones tiered resolver above; pass all_zones to get it.
        blob = _norm(f"{zone.id} {zone.zone_type} {zone.label}")
        z = _norm(op.zone)
        if z and z in blob:
            return True
        words = set(blob.split())
        return any(len(t) >= 5 and t in words for t in z.split())
    return op.zone is None and op.zone_function is None      # unscoped op -> all zones


def agent_matches(name: str, agent) -> bool:
    """True if `name` (from an op) refers to this agent, by id or (sub)name, case/format-insensitive."""
    if not name:
        return False
    n = _norm(name)
    return n == _norm(agent.id) or n == _norm(getattr(agent, "name", "")) or n in _norm(getattr(agent, "name", ""))


def agent_directive_target(patches, agent, scene) -> str:
    """Where a named agent has been directed (zone id), or None. Resolves 'go to <other agent>' to that
    agent's current zone so 'Morgan talk to Sam' moves Morgan to wherever Sam is. Newest patch wins."""
    all_zones = list(scene.zones.values())
    for p in reversed(list(patches)):
        for o in p.ops:
            if o.op != "agent_directive" or not agent_matches(o.agent, agent):
                continue
            if o.zone:
                for z in all_zones:
                    if zone_matches(o, z, all_zones):
                        return z.id
            if o.target_agent:
                for other in scene.agents.values():
                    if other.id != agent.id and agent_matches(o.target_agent, other):
                        return other.current_zone
    return None


def personal_action_directives(patches, scene) -> dict:
    """{agent_id: action} for a named-person 'do this specific thing' directive — e.g. 'Alex wants to sit
    down' -> {agent:'Alex', action:'sit'}. Distinct from a plain zone directive (agent_directive_target,
    {agent,zone}) and a meet directive (meet_pairs, {agent,target_agent}): this one has NO zone/target_agent,
    just a semantic action, and gets resolved to a concrete reachable smart object at tick time (live_bridge),
    not here (this module has no notion of nav-reachability). Newest patch wins per agent."""
    out = {}
    for p in reversed(list(patches)):
        for o in p.ops:
            if o.op != "agent_directive" or not o.agent or o.zone or o.target_agent or not o.action:
                continue
            subj = next((a for a in scene.agents.values() if agent_matches(o.agent, a)), None)
            if subj and subj.id not in out:
                out[subj.id] = o.action
    return out


def meet_pairs(patches, scene) -> list:
    """(subject_id, target_id) for each active 'X talk to Y' directive, resolved to real agent ids — so the
    server can tell Unity to make X WALK to Y and converse (stop + face + talk), not just share a zone."""
    out, seen = [], set()
    for p in reversed(list(patches)):
        for o in p.ops:
            if o.op != "agent_directive" or not o.target_agent:
                continue
            subj = next((a for a in scene.agents.values() if agent_matches(o.agent, a)), None)
            tgt  = next((a for a in scene.agents.values() if agent_matches(o.target_agent, a)), None)
            if subj and tgt and subj.id != tgt.id and subj.id not in seen:
                seen.add(subj.id); out.append((subj.id, tgt.id))
    return out


def meet_patches_complete(patches, scene, met_state, radius=1.5, dwell=3):
    """Meet directives ('X talk to Y') whose subject+target have CONVERGED (within `radius`) and stayed
    together for `dwell` ticks — long enough to actually have the conversation. Return them so the caller
    drops them IMMEDIATELY (not wait out the full ttl), so 'X talk to Y' is ONE interaction (walk over, chat,
    resume life) instead of the pair being locked meeting for ~25 ticks. `met_state` is a caller-owned dict
    {pair_key: first_close_tick} tracking convergence across ticks (cleared here on scene reset). Uses the live
    positions synced from Unity update_state; if positions are missing it never completes early (the patch ttl
    is the backstop, e.g. if the two can never physically reach each other). Mirrors leave_patches_complete."""
    tick = getattr(scene, "tick_no", 0)
    if tick <= 1:
        met_state.clear()
    done = []
    for p in patches:
        meet_ops = [o for o in p.ops if o.op == "agent_directive" and o.target_agent]
        if not meet_ops:
            continue
        key = tuple(sorted((o.agent or "", o.target_agent or "") for o in meet_ops))
        close = True
        for o in meet_ops:
            subj = next((a for a in scene.agents.values() if agent_matches(o.agent, a)), None)
            tgt  = next((a for a in scene.agents.values() if agent_matches(o.target_agent, a)), None)
            sp = getattr(subj, "pos", None) if subj else None
            tp = getattr(tgt, "pos", None) if tgt else None
            if sp is None or tp is None or (sp[0] - tp[0]) ** 2 + (sp[1] - tp[1]) ** 2 > radius * radius:
                close = False
                break
        if close:
            met_state.setdefault(key, tick)
            if tick - met_state[key] >= dwell:
                done.append(p)
                met_state.pop(key, None)
        else:
            met_state.pop(key, None)   # drifted apart before dwelling -> reset the convergence clock
    return done


def _resolve_leave_targets(patch, scene) -> list:
    """One-shot resolution of a SINGLE patch's agent_leave ops against the CURRENT scene.agents — the raw
    matching logic (by name or by group/role/type), run exactly once at patch-activation time by
    `lock_leavers`. Never call this per-tick; that re-match-against-a-shrinking-list is what caused the
    'resolved to NOBODY' bug once some members had already left."""
    gtype = {}
    for gid, gd in (getattr(scene, "social_groups", {}) or {}).items():
        for m in gd.get("members", []):
            gtype[m] = (gd.get("type") or "").lower()
    out, seen = [], set()
    for o in patch.ops:
        if o.op != "agent_leave":
            continue
        if o.agent:                                            # a named person
            subj = next((a for a in scene.agents.values() if agent_matches(o.agent, a)), None)
            if subj and subj.id not in seen:
                seen.add(subj.id); out.append(subj.id)
        elif o.role:                                           # a group / role / type ('family', 'tourists')
            r = _norm(o.role)
            # candidate forms so 'friends'->'friend', 'family members'/'the family group'->'family' all hit:
            # the raw role, its singular (drop trailing s), and its first word + that word's singular.
            cands = {r, r.rstrip("s")}
            if r:
                head = r.split()[0]
                cands |= {head, head.rstrip("s")}
            cands.discard("")
            for a in scene.agents.values():
                if a.id in seen:
                    continue
                g = gtype.get(a.id, "")
                gset = {g, g.rstrip("s")} if g else set()
                blob = _norm(f"{getattr(a, 'role', '')} {getattr(a, 'agent_type', '')} {g}")
                if any(c and (c in gset or c in blob) for c in cands):
                    seen.add(a.id); out.append(a.id)
    return out


def lock_leavers(patch, scene) -> list:
    """Resolve a patch's agent_leave targets ONCE and freeze them on the patch. Idempotent — later calls are
    no-ops. Call this the moment a leave patch activates (patch-creation time), so the frozen id list survives
    agents departing/being removed and role/group churn, instead of re-matching text against a live list that
    is actively shrinking because of the very directive being resolved."""
    if not patch.leavers_locked:
        patch.resolved_leavers = _resolve_leave_targets(patch, scene)
        patch.leavers_locked = True
    return patch.resolved_leavers


def agents_leaving(patches, scene) -> list:
    """agent ids CURRENTLY still in the scene that are locked to leave (routed out like a personal evacuation:
    walk to the exit, then off the floor plan and disappear). Auto-locks any not-yet-resolved patch (so a
    caller never needs to remember to call lock_leavers explicitly), then reads the FROZEN id list — filtered
    to ids still present — rather than re-matching name/role/group against the live (shrinking) agent set."""
    out, seen = [], set()
    for p in reversed(list(patches)):
        if any(o.op == "agent_leave" for o in p.ops):
            for aid in lock_leavers(p, scene):
                if aid in scene.agents and aid not in seen:
                    seen.add(aid); out.append(aid)
    return out


def leave_patches_complete(patches, scene) -> list:
    """Leave patches whose ENTIRE frozen target list has already left the scene (none remain in scene.agents).
    These are done — the caller should drop them from active_patches immediately (not wait for ttl) so the
    patch stops being reported/logged as active and its slot is free for a new directive."""
    done = []
    for p in patches:
        if not any(o.op == "agent_leave" for o in p.ops):
            continue
        targets = lock_leavers(p, scene)
        if targets and not any(aid in scene.agents for aid in targets):
            done.append(p)
    return done


def role_matches(op: PatchOp, agent) -> bool:
    if not op.role or op.role in ("all", "*", "everyone"):
        return True
    r = op.role.lower()
    return r == agent.role.lower() or r == getattr(agent, "social_status", "").lower()


# ── effect queries over the active patch set (read by the policy each tick) ──
def zone_attraction(patches, zone, all_zones=None) -> float:
    return sum(o.delta for p in patches for o in p.ops
               if o.op == "zone_attraction" and zone_matches(o, zone, all_zones))


def role_zone_bias(patches, agent, zone) -> float:
    return sum(o.weight for p in patches for o in p.ops
               if o.op == "role_priority" and zone_matches(o, zone) and role_matches(o, agent))


def zone_bias(patches, agent, zone) -> float:
    """Total additive pull toward this zone for this agent (attraction + role priority)."""
    return zone_attraction(patches, zone) + role_zone_bias(patches, agent, zone)


# ── graph-EDIT ops: structural scene mutations (applied ONCE when a patch is created) ──
SPAWN_CAP = 12          # never spawn more than this many agents from one op (safety)


# Generic words the LLM reaches for when it means "a small consumable item" rather than a specific object
# id ("a glass broke", "clear the tray", "the cup is empty"). These bind to the smart-object CATEGORY
# (food_drink / container class) instead of failing to string-match a prop_id like `cafe__drink`.
_CONSUMABLE_WORDS = frozenset({
    "glass", "glasses", "cup", "cups", "mug", "mugs", "bottle", "bottles", "tray", "trays",
    "plate", "plates", "dish", "dishes", "bowl", "bowls", "can", "cans", "napkin", "napkins",
    "drink", "drinks", "food", "snack", "snacks", "meal", "meals", "pastry", "cutlery", "tableware",
})
# …but the item words above must NOT resolve to a refillable FIXTURE that merely affords drink/eat
# (a coffee machine / counter / table stays put — you don't "break" it). Mirrors live_bridge._FIXTURE_KEYS.
_FIXTURE_WORDS = frozenset({
    "machine", "counter", "bar", "table", "chair", "sofa", "couch", "bench", "sink", "fountain",
    "dispenser", "fridge", "cooler", "stall", "toilet", "urinal", "shelf", "desk", "stand", "station", "exhibit",
})


def _consumable_words(op) -> set:
    return {w for w in _norm(op.object).split() if w in _CONSUMABLE_WORDS} if op.object else set()


def object_is_consumable(obj) -> bool:
    """True if `obj` is a small removable/consumable item (food_drink/container) rather than a fixture —
    the smart-object CATEGORY a 'a glass broke' / drink-finished event should be able to target."""
    t = _norm(f"{getattr(obj, 'object_type', '')} {getattr(obj, 'display_name', '')} "
              f"{getattr(obj, 'label', '')} {getattr(obj, 'id', '')}")
    if any(f in t for f in _FIXTURE_WORDS):
        return False
    if any(k in t for k in _CONSUMABLE_WORDS):
        return True
    affs = {getattr(a, "action", None) for a in getattr(obj, "affordances", [])}
    return bool(affs & {"drink", "eat"})


def _obj_blob(obj) -> str:
    # display_name is in here deliberately: it is the name the UI SHOWS the user ("Coffee Machine"), so it is
    # the name they will type at the director. Without it, a hand-built level whose object_type is an asset id
    # ("Kitchen_Singles_48x48_186") could be seen but never named.
    return _norm(f"{getattr(obj, 'id', '')} {getattr(obj, 'object_type', '')} "
                 f"{getattr(obj, 'display_name', '')} {getattr(obj, 'label', '')}")


def _object_substr_match(op, obj) -> bool:
    """Tier 1 — the op names this object exactly (id/type/label substring): 'so_5', 'cafe__drink'."""
    if not op.object:
        return False
    o = _norm(op.object)
    return bool(o) and o in _obj_blob(obj)


def _object_word_match(op, obj) -> bool:
    """Tier 2 — a distinctive (>=4 char) word of the reference appears in this object's id/type/label."""
    if not op.object:
        return False
    words = set(_obj_blob(obj).split())
    return any(len(t) >= 4 and t in words for t in _norm(op.object).split())


def _object_matches(op, obj) -> bool:
    """Full boolean: substring OR word OR class/affordance-aware semantic match (a generic consumable word —
    'glass'/'tray'/'drink' — binds to any consumable-category object). resolve_remove_targets applies these
    with strict precedence; this stays for callers that just want a yes/no."""
    return (_object_substr_match(op, obj) or _object_word_match(op, obj)
            or (bool(_consumable_words(op)) and object_is_consumable(obj)))


def _obj_in_op_zone(op, obj, scene) -> bool:
    """True if `obj` sits in the zone the op names (or the op is zone-unscoped)."""
    if not op.zone and not op.zone_function:
        return True
    z = getattr(scene, "zones", {}).get(getattr(obj, "zone_id", ""))
    return z is not None and zone_matches(op, z, list(getattr(scene, "zones", {}).values()))


def resolve_remove_targets(op, scene) -> list:
    """Resolve a remove_object op to concrete PLACED-object ids, by STRICT precedence (each tier only used
    if the previous found nothing, so a specific reference never spills into a broad match):
      (1) exact id/type/label substring — 'so_5', 'cafe__drink' → that object;
      (2) distinctive word overlap — 'coffee machine' → the object whose id/type contains those words;
      (3) semantic — a generic consumable word ('glass'/'tray'/'drink') → a consumable-category object,
          preferring the named zone, and only ONE item (a single dropped glass, not the whole cafe).
    A generic consumable word that word-matches several objects (tier 2) is likewise capped to one, zone-first.
    Already-removed objects are skipped. Returns [] if nothing resolves (caller logs the miss)."""
    objs = getattr(scene, "objects", {})
    live = [(oid, ob) for oid, ob in objs.items() if not getattr(ob, "removed", False)]

    substr = [oid for oid, ob in live if _object_substr_match(op, ob)]
    if substr:
        return substr

    def _one_zone_first(ids):
        zoned = [oid for oid in ids if _obj_in_op_zone(op, objs[oid], scene)]
        return (zoned or ids)[:1]

    word = [oid for oid, ob in live if _object_word_match(op, ob)]
    if word:
        return _one_zone_first(word) if (_consumable_words(op) and len(word) > 1) else word

    if _consumable_words(op):
        cands = [oid for oid, ob in live if object_is_consumable(ob)]
        if cands:
            return _one_zone_first(cands)
    return []


def _next_agent_id(scene) -> str:
    n = 0
    while f"spawn_{n}" in scene.agents:
        n += 1
    return f"spawn_{n}"


def _entry_zone_id(scene, op) -> str:
    """The zone new arrivals enter through — the access/entrance zone (so they come in the door), else the
    zone the op named, else the first zone."""
    for z in scene.zones.values():
        if "access" in (getattr(z, "zone_function", None) or []):
            return z.id
    for z in scene.zones.values():
        blob = _norm(f"{z.id} {z.zone_type} {z.label}")
        if any(k in blob for k in ("entrance", "exit", "door", "lobby", "foyer", "reception")):
            return z.id
    named = [z.id for z in scene.zones.values() if zone_matches(op, z)]
    return named[0] if named else (next(iter(scene.zones)) if scene.zones else "")


def _spawn_outside_point(scene, zid):
    """Where a `spawn_agent` arrival (firefighters, medics, extra patrons) appears.

    PREFERS THE REAL DOOR — the egress-portal gate Unity exports in the scene graph. The old behaviour
    guessed a point "just outside the world boundary" from a HARDCODED rect (X∈[-8,8], Y∈[-5,5]) that no
    longer matches any real level: CafeDemo is x∈[-11.4,34.2], y∈[-5.5,5.5], so for the reception zone it
    returned y=-6.2 — which is INSIDE the south wall band (-6.69..-5.31), not outside the building. That
    was masked for a long time because Unity re-rolled every arrival to the entrance zone centre and threw
    the server's coordinates away; once Unity started honouring them (so arrivals enter at the door), the
    bad coordinate surfaced as responders spawning in a wall and wandering off instead of attending the
    hazard. Fall back to the old guess only when no graph has been exported yet."""
    try:
        from scene_bridge import entry_point          # late import: scene_bridge imports this module
        p = entry_point(scene, near_zone=zid)
        if p is not None:
            return (float(p[0]), float(p[1]))
    except Exception:
        pass
    z = scene.zones.get(zid)
    if z is None:
        return (0.0, -6.0)
    cx, cy = z.center
    wx, wy = 8.0, 5.0
    dist = {"right": wx - cx, "left": cx + wx, "top": wy - cy, "bottom": cy + wy}
    side = min(dist, key=dist.get)
    if side == "right":
        return (wx + 1.2, cy)
    if side == "left":
        return (-wx - 1.2, cy)
    if side == "top":
        return (cx, wy + 1.2)
    return (cx, -wy - 1.2)


def apply_structural_ops(scene, patch):
    """Apply the graph-EDIT ops ONCE on patch creation: `remove_object` takes matching smart objects OUT OF
    SERVICE reversibly (marks them removed/unavailable — id + provenance preserved, affordances disabled,
    reservations released, excluded from options — NOT deleted); `spawn_agent` adds new agents
    (firefighters/medics/arrivals) in a zone. Mutates the scene (so the GNN's next-tick graph reflects it)
    and RETURNS {removed_objects:[ids], spawned_agents:[...]}. Spawns are mirrored to Unity via scene_mutation;
    removals are conveyed via object_states.available (reversible hide). role_directive is read per tick."""
    from .world import AgentInstance
    from .needs import Needs
    removed, spawned = [], []
    src = getattr(patch, "display_name", None) or getattr(patch, "event_type", "remove_object")
    for op in patch.ops:
        if op.op == "remove_object":
            # REVERSIBLE: mark the resolved object(s) UNAVAILABLE (id + provenance preserved), never delete —
            # restore_expired_removals() puts them back when this patch clears.
            for oid in resolve_remove_targets(op, scene):
                scene.objects[oid].mark_removed(source=src)
                removed.append(oid)
        elif op.op == "spawn_agent":
            if _DISABLE_RESPONDERS and (op.role or "responder").lower() in _RESPONDER_ROLES:
                continue                                       # Step 8: no responders while validating civilians
            # New arrivals enter from OUTSIDE — spawn them just past the world boundary near the entrance and
            # place them logically in the entrance zone, so Unity walks them IN through the door (not popping
            # into the middle of the room). Prefer the access/entrance zone regardless of what op names.
            entry = _entry_zone_id(scene, op)
            ox, oy = _spawn_outside_point(scene, entry)
            role = (op.role or "responder").lower()
            for i in range(max(1, min(int(op.count or 1), SPAWN_CAP))):
                aid = _next_agent_id(scene)
                name = f"{role.title()} {len(spawned) + 1}"
                scene.agents[aid] = AgentInstance(id=aid, name=name, role=role,
                                                  needs=Needs(), current_zone=entry)
                jitter = (i - 1) * 0.5                          # fan multiple arrivals so they don't overlap
                spawned.append({"id": aid, "name": name, "role": role, "agent_type": role,
                                "zone_id": entry, "x": float(ox + jitter), "y": float(oy)})
    return {"removed_objects": removed, "spawned_agents": spawned}


def role_directive_targets(patches, scene) -> dict:
    """{role: zone_id} for active role_directive ops — EVERY agent of that role is routed there (responders →
    the hazard zone). Newest patch wins per role."""
    out = {}
    if _DISABLE_RESPONDERS:
        return out                                            # Step 8: no responder routing while validating
    for p in reversed(list(patches)):
        for o in p.ops:
            if o.op == "role_directive" and o.role:
                for z in scene.zones.values():
                    if zone_matches(o, z):
                        out.setdefault(o.role.lower(), z.id)
                        break
    return out


def restore_expired_removals(scene) -> list:
    """Restore any object whose OWNING patch is no longer active (its removal was reversible). Matched by
    the provenance recorded in `removed_by`; returns the ids restored so the caller can re-show them."""
    active = {getattr(p, "display_name", None) or getattr(p, "event_type", None)
              for p in getattr(scene, "active_patches", [])}
    restored = []
    for oid, o in getattr(scene, "objects", {}).items():
        if getattr(o, "removed", False) and getattr(o, "removed_by", None) not in active:
            o.restore()
            restored.append(oid)
    return restored


def action_disabled(patches, action: str, zone, all_zones=None) -> bool:
    return any(o.op == "disable_affordance" and o.action == action and
               (o.zone is None and o.zone_function is None or zone_matches(o, zone, all_zones))
               for p in patches for o in p.ops)


def zone_enabled_actions(patches, zone, all_zones=None) -> set:
    return {o.action for p in patches for o in p.ops
            if o.op == "enable_affordance" and o.action and zone_matches(o, zone, all_zones)}


def zone_disabled_actions(patches, zone, all_zones=None) -> set:
    return {o.action for p in patches for o in p.ops
            if o.op == "disable_affordance" and o.action
            and (o.zone is None and o.zone_function is None or zone_matches(o, zone, all_zones))}


def active_event_types(patches) -> list:
    return [p.event_type for p in patches]


def need_rate_delta(patches, agent, need: str) -> float:
    return sum(o.delta for p in patches for o in p.ops
               if o.op == "need_rate" and o.need == need and role_matches(o, agent))
