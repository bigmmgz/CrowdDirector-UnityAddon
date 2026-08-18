"""
live_bridge.py — drive the LIVE CrowdSim demo with the trained ECGP policy.

Builds an EcgpWorld from the server's grounded dsag SceneModel each tick (agents + needs, zones +
adjacency from the grid layout, grounded smart objects, active EventPatches → overlays), runs the frozen
64x3 ECGP model, and returns per-agent actions in the server's exact wire format
(`move_to_zone`/`rest`/`meet`/`idle`, with smart_object_id + interaction point + evacuate). Drops into
`crowd_director_server.request_tick` as a new `behavior_engine == "ecgp"`. Unity stays the visual layer;
the meet/leave overlays the server already applies on top still work. Falls back cleanly on any error.
"""
import json
import logging
import math
import os
import random
import re
import time

from ..graph.world import EcgpWorld, EAgent, EZone, EEvent, EOperation, EGroup
from ..graph.chain_state import ChainState, ChainStage
from .. import social as cs
from ..graph.encoder import encode
from ..teacher import need_pressure as NP
from .. import vocab as V
from .inference import ECGPPolicy
from . import behavior_profile as BP
from . import spill as _spill

import dsag_bridge  # for _outside_point (evacuation off-map target)

log = logging.getLogger("ecgp.live")
_TRACE = bool(os.environ.get("ECGP_TRACE"))          # set ECGP_TRACE=1 to log per-tick decision diagnostics
_TRACE_PATH = os.environ.get("ECGP_TRACE_PATH", "outputs/ecgp_trace.jsonl")
_LAST_TARGET = {}                                    # agent_id -> (target_type, target_id) for switch detection
_COMMIT = {}                                         # agent_id -> committed action {target_type,target_id,action,ttl}
_QUEUE_LOCK = {}                                     # agent_id -> {oid, pt, action}: held place in a capacity queue

# ── TWO EXPLICIT RUNTIME MODES (default: research_learned, for paper experiments) ──────────────────────
#   research_learned  — ONLY hard feasibility masks, navigation/execution, capacity/reservations, explicit
#                        single-agent/group directives, and the critical emergency safety override are
#                        active. Ordinary needs and non-critical (ambient) events are resolved ENTIRELY by
#                        ECGP's own joint-option scoring — no deterministic attraction/party/gather routing.
#                        This is the mode any paper claim about EventPatch-conditioned graph inference must
#                        be measured in.
#   safe_demo_hybrid   — additionally enables deterministic AMBIENT event routing (party/gather zone
#                        attraction) as an explicitly optional engineering fallback for live demos. Actions
#                        produced this way carry origin="hybrid_fallback" (see the origin taxonomy in
#                        ecgp_tick) and MUST NOT be cited as evidence of learned event-response.
# The two modes must never silently share results in a report — always state ECGP_MODE alongside a trace.
_MODE = os.environ.get("ECGP_MODE", "research_learned")
if _MODE not in ("research_learned", "safe_demo_hybrid"):
    logging.getLogger("ecgp.live").warning(f"[ecgp] unknown ECGP_MODE={_MODE!r} — using research_learned")
    _MODE = "research_learned"
_RESEARCH_MODE = _MODE == "research_learned"

# AMBIENT hybrid routing (party/gather zone-attraction) ONLY — the learned policy is only weakly
# event-conditioned, so this deterministic fallback exists purely for safe_demo_hybrid and is explicitly
# flagged/logged as "hybrid" so it is NEVER counted as evidence that ECGP learned event redirection. It is
# HARD-DISABLED in research_learned regardless of the legacy env var; in safe_demo_hybrid, the legacy var
# still allows turning it off per-run (ECGP_HYBRID_DIRECTIVE=0) to observe the PURE model response.
# Explicit single-agent/group directives (role dispatch, personal action, directed leave — section 6) are
# NOT ambient hybrid behaviour and are always active in BOTH modes; they do not use this flag.
_HYBRID_DIRECTIVE = (not _RESEARCH_MODE) and os.environ.get("ECGP_HYBRID_DIRECTIVE", "1") == "1"

# Deterministic Tier-1 NEED-RELIEF routing — safe_demo_hybrid ONLY, the SAME mode-split as party/gather.
# The frozen V2 GNN is only weakly need-conditioned (Round-5 diagnostics: a bladder=95 agent picks relieve
# only ~10% of the time), so in DEMO mode an urgent agent is reliably sent to a toilet/food/drink/seat; in
# research_learned this is OFF and needs stay the model's job (honestly measured for the paper). Per-run
# override ECGP_NEED_RELIEF=0 disables it within demo mode to observe the pure model response.
_HYBRID_NEED_RELIEF = (not _RESEARCH_MODE) and os.environ.get("ECGP_NEED_RELIEF", "1") == "1"

# Dining-cluster macro-option, off by default for the same reason as ROLE_MASK_V2: a checkpoint that
# never saw a cluster_root object during training would be handed one in its live candidate set, which is
# the train/live skew the other flags exist to avoid. A model trained with clusters turns this on
# alongside ECGP_ROLE_MASK_V2.
_CLUSTER_MASK_V2 = os.environ.get("ECGP_CLUSTER_MASK_V2", "0") == "1"
_FOCAL = os.environ.get("ECGP_TRACE_AGENT", "agent_0")   # agent whose needs/options are logged under ECGP_TRACE

# Staggered decision timing (user idea): agents do NOT all re-decide on the same tick — an idle/uncommitted
# agent only re-picks on its OWN phase tick, so the crowd doesn't move in lockstep (more natural, calmer).
# Events (attraction/leave) and an urgent Tier-1 need bypass the stagger so a called/urgent agent acts NOW.
_STAGGER = os.environ.get("ECGP_STAGGER", "1") == "1"
_DECIDE_PERIOD = max(1, int(os.environ.get("ECGP_DECIDE_PERIOD", "3")))

# Trait-conditioned Action Prior (behavior_profile.py): reweight the frozen GNN's option distribution by a
# per-agent personality prior (product-of-experts). OFF by default; BETA is the ablation gain (0 == frozen
# ECGP exactly). NOT the learned policy — a disposition LAYER on top of it; never overrides feasibility/needs.
_BEHAVIOR_PRIOR = os.environ.get("ECGP_BEHAVIOR_PRIOR", "0") == "1"
_PRIOR_BETA = float(os.environ.get("ECGP_PRIOR_BETA", "1.0"))
_PRIOR_SAMPLE = os.environ.get("ECGP_PRIOR_SAMPLE", "1") == "1"   # sample re-pick (anti-herding) vs argmax
_TALK_CD = int(os.environ.get("ECGP_TALK_COOLDOWN", "4"))         # ticks a fresh talk/help is damped (refractory)
_LAST_TALK = {}                                                   # agent_id -> tick of last social interaction


def _load_calibrated_theta():
    """Calibrated per-PROFILE theta table {profile_name: {feature: w}} from ECGP_PRIOR_THETA (a fit produced
    by ecgp.calibration.fit_profiles). Absent -> {} -> the untuned OCEAN/role init is used instead. This is
    the ONLY difference between condition (2) OCEAN-init and (3) calibrated at runtime."""
    path = os.environ.get("ECGP_PRIOR_THETA")
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"[ecgp/prior] could not load calibrated theta {path}: {e}")
        return {}


_CALIBRATED_THETA = _load_calibrated_theta()


def scene_profiles(scene):
    """Per-agent BehaviorProfile for this scene, cached. theta precedence: an explicit calibrated table
    (ECGP_PRIOR_THETA, keyed by profile name) -> else the untuned interpretable role prior. Profile NAME and
    group come from the agent's role / group_id (set by scene-gen, the LLM 'casting' step)."""
    cached = getattr(scene, "_behavior_profiles", None)
    if cached is not None:
        return cached
    # the SOCIAL UNIT is the strongest signal for the social profile — a family-group member is a 'family'
    # profile even if its role string ('family'->casual_young) lost the keyword. Group type wins; else role.
    gtype = {}
    for gid, gd in (getattr(scene, "social_groups", {}) or {}).items():
        for m in gd.get("members", []):
            gtype[m] = gd.get("type")
    profs = {}
    for aid, a in scene.agents.items():
        gt = gtype.get(aid)
        if gt == "family":
            name = "family"
        elif gt in ("friend", "acquaintance"):
            name = "socializer"
        else:
            name = BP.profile_name_for(getattr(a, "role", ""))
        theta = _CALIBRATED_THETA.get(name) or BP.role_prior_theta(name)
        profs[aid] = BP.BehaviorProfile(name=name, theta=dict(theta),
                                        group_id=getattr(a, "group_id", None))
    scene._behavior_profiles = profs
    return profs


def _group_ctx(scene, agent):
    """Groupmates + the zones they currently occupy, so the 'group' option-feature can fire (family cohesion)."""
    gid = getattr(agent, "group_id", None)
    if not gid:
        return {}
    members, zones = set(), set()
    for oid, o in scene.agents.items():
        if oid != agent.id and getattr(o, "group_id", None) == gid:
            members.add(oid)
            zones.add(getattr(o, "current_zone", ""))
    return {"group_members": members, "group_zones": zones}


def _pick_option(opts):
    """Re-select the chosen option AFTER prior reweighting — sample (anti-herding) or argmax."""
    if _PRIOR_SAMPLE and len(opts) > 1:
        z = sum(o["p"] for o in opts)
        if z > 0:
            r = random.random() * z
            for o in opts:
                r -= o["p"]
                if r <= 0:
                    return o
    return max(opts, key=lambda o: o["p"])


def apply_behavior_prior(decisions, scene):
    """Reweight each agent's GNN option distribution by its trait prior (product-of-experts) and re-pick the
    chosen option. A pure LAYER on the frozen policy: feasibility/needs already shaped d['options']; this only
    tilts them. beta==0 (or the flag off) is a strict no-op == frozen ECGP."""
    if not _BEHAVIOR_PRIOR or not _PRIOR_BETA:
        return
    profs = scene_profiles(scene)
    tick = getattr(scene, "tick_no", 0)
    for aid, d in decisions.items():
        agent = scene.agents.get(aid)
        opts = d.get("options")
        prof = profs.get(aid)
        if agent is None or not opts or prof is None:
            continue
        # MASLOW GATE: an urgent Tier-1 physiological need (bladder/thirst/hunger/energy) is NON-negotiable —
        # the prior is DISPOSITION, it must never pull a bursting agent away from relief. Leave the GNN's
        # need-driven choice untouched for urgent agents (same gate the attraction overlay already respects).
        if _urgent_tier1(agent):
            continue
        theta = BP.agent_theta(agent, prof)
        # INTERACTION REFRACTORY: just after a talk/help, damp social/group so families interact repeatedly
        # but not CONTINUOUSLY (they chat, then do something else, then chat again). With commitment/dwell
        # this prevents the degenerate "talk every tick to the same person" loop.
        if tick - _LAST_TALK.get(aid, -10**9) < _TALK_CD:
            theta = dict(theta)
            theta["social"] = theta.get("social", 0.0) - 3.0   # damp CHATTER only; cohesion (group) stays
        BP.reweight(opts, theta, agent, _group_ctx(scene, agent), _PRIOR_BETA)
        d["chosen"] = _pick_option(opts)
        if d["chosen"].get("action") in ("talk", "help"):
            _LAST_TALK[aid] = tick


def _my_turn(aid, tick):
    return (not _STAGGER) or (tick % _DECIDE_PERIOD == (hash(aid) % _DECIDE_PERIOD))


def _dwell(action):
    """How many ticks an agent COMMITS to an action once chosen (walk there + do the thing) before the
    policy is allowed to re-pick — stops per-tick target-switching (the 'darting/unreal' motion)."""
    if action in ("sit", "rest"):
        return 3                                      # people linger when seated
    if action in ("drink", "eat", "relieve", "wash", "observe", "pay", "buy", "work", "talk", "help"):
        return 2
    return 1                                          # walk/idle: re-decide sooner


def _com_cur(com):
    """The CURRENT step of a commitment — a chain's active step, or the commitment itself (legacy single)."""
    return com["steps"][com["i"]] if "steps" in com else com


def _queue_follow_on(scene, aid, finished):
    """After an agent FINISHES using an object, send it straight to that object's `follow_on` (wash your
    hands after the toilet). A habit, not a decision: it is committed here rather than left to the policy,
    which would otherwise have to learn it and would skip it whenever the need was already satisfied — the
    whole point is that it happens EVERY time, regardless of need.

    Committed like any other action so it inherits arrival gating, dwell, reservations and the unreachable
    give-up path. Skipped when the follow-on is gone, full, or disabled — never leaves the agent stuck."""
    if finished.get("target_type") != "object":
        return
    src = scene.objects.get(finished.get("target_id"))
    nxt_id = getattr(src, "follow_on", None) if src is not None else None
    if not nxt_id:
        return
    nxt = scene.objects.get(nxt_id)
    if nxt is None or getattr(nxt, "removed", False) or not nxt.capacity_ok():
        return
    action = next((a.action for a in getattr(nxt, "affordances", [])), None)
    if not action:
        return
    _COMMIT[aid] = {"target_type": "object", "target_id": nxt.id, "action": action,
                    "ttl": _dwell(action), "source": "interaction", "priority": 2,
                    "semantic_origin": "deterministic_execution"}
    log.info(f"[ecgp/followon] {aid} finished {src.id} -> {nxt.id} ({action})")


def _target_alive(scene, com):
    cur = _com_cur(com)
    tt, tid = cur["target_type"], cur["target_id"]
    if tt == "object":
        return tid in scene.objects
    if tt == "zone":
        return tid in scene.zones
    return True


def _chain_cleanup(scene, aid, com):
    """Release a chain's seat reservation/occupancy when the chain is aborted or completed."""
    if "steps" not in com:
        return
    seat = scene.objects.get(com.get("seat", ""))
    if seat is None:
        return
    tok = com.get("token")
    if tok:
        seat.cancel(tok)
    else:
        seat.release(aid)


def _drop_commit(scene, aid):
    """ABANDON an agent's commitment and release whatever it had reserved.

    A bare `_COMMIT.pop(aid, None)` leaks: if the agent was mid-chain it still holds a seat reservation on a
    smart object, and nothing ever gives it back — occupancy drifts up until the object looks permanently
    full and no one can use it again. The deliberate `_chain_cleanup` + `pop` pairs elsewhere in this file
    are the correct discipline; the override branches (spill, directed leave, role directive, frustration,
    party attraction) each skipped it. Every ambient attraction event fires that party branch, so the leak
    compounds once Tier-A events are added — which is why this exists before them.

    Only for ABANDONMENT. The normal completion path at the end of the commit-honor block already calls
    `seat.release(aid)` itself, and `SmartObject.release` is NOT idempotent (it decrements occupancy
    whenever occupancy > 0), so calling this there too would free a slot another agent is holding.
    """
    com = _COMMIT.pop(aid, None)
    if com is not None:
        _chain_cleanup(scene, aid, com)


_DIRECTED_ACTION_OBJ = {}   # agent_id -> smart_object_id: the STABLE object a personal action directive
                            # resolved to, so re-affirming the same directive every tick targets the SAME
                            # object/coordinates (no re-pick -> stable intent -> no path reset).


_SURFACE_KEYS = ("table", "desk", "counter", "bar", "shelf", "bookcase", "bookshelf", "workbench", "island")


def _is_seat_category(o) -> bool:
    """DEFENSIVE category check: 'sit' must resolve to an actual seat (chair/stool/sofa/bench/couch), NEVER a
    surface — even if an object's affordance list is wrong (the exact bug found in dsag/templates.py's
    idealized _table archetype, which used to advertise 'sit' directly; fixed there too, but this check is
    the hard guarantee the resolver itself enforces regardless of what upstream data claims)."""
    blob = f"{getattr(o, 'object_type', '') or ''} {getattr(o, 'id', '') or ''}".lower()
    return not any(k in blob for k in _SURFACE_KEYS)


def _find_free_object_for_action(scene, action, near_zone=None):
    """A free, zone-gate-legal object offering `action` — generalizes _find_free_seat to any affordance, for
    a personal action directive ('Alex wants to sit down'). Prefers one in the agent's current zone."""
    best = None
    for oid, o in scene.objects.items():
        if getattr(o, "removed", False) or not o.capacity_ok():
            continue
        if not any(a.action == action for a in o.affordances):
            continue
        if action == "sit" and not _is_seat_category(o):     # never a table/desk/counter, even if mislabeled
            continue
        if not _action_ok(scene, o.zone_id, action):
            continue
        score = (0 if near_zone and o.zone_id == near_zone else 1, oid)
        if best is None or score < best[0]:
            best = (score, o)
    return best[1] if best else None


def _find_free_seat(scene, act):
    """A free SEAT (sit-affording, capacity open, actual seat CATEGORY — never a table/desk/counter) in a
    zone where `act` is allowed — prefers chairs bound to a table (scene-grammar parent) so agents dine AT
    tables, not on stray stools."""
    best = None
    for oid, o in scene.objects.items():
        if getattr(o, "removed", False) or not o.capacity_ok():
            continue
        if not any(a.action == "sit" for a in o.affordances):
            continue
        if not _is_seat_category(o):
            continue
        if not _action_ok(scene, o.zone_id, act):
            continue
        score = (0 if getattr(o, "parent_id", None) else 1, oid)
        if best is None or score < best[0]:
            best = (score, o)
    return best[1] if best else None

_POLICY = None
# CrowdDirect v3 is the deployed director. It is the SAME architecture as v2 (V21_MODEL_CONFIG) trained
# with options.DIRECTIVE_TARGET_V3, so a directed agent can see WHICH option is its ordered destination
# (directive-category acceptable actions 0.015 -> 0.436). The encoding flags that go with it are NOT set
# here — ECGPPolicy derives them from the checkpoint's own config, so pointing ECGP_CKPT back at an older
# checkpoint automatically reverts the encoding too (see inference.ECGPPolicy.encoding_contract).
# seed 0 is the checkpoint every crowddirect_v3/evaluation entry point uses; re-pointing at the frozen
# path is a read and violates nothing in crowddirect_v3/FREEZE_MANIFEST.json — do NOT copy the .pt to a
# deployment folder, that would create an unhashed artifact outside the frozen release.
_CKPT = os.environ.get("ECGP_CKPT", os.path.join(os.path.dirname(__file__), "..", "..",
                                                 "model", "full_graph_seed0", "best.pt"))
_STAFF_KEYS = ("staff", "barista", "waiter", "bartender", "nurse", "teacher", "security", "guard",
               "cashier", "chef", "cook", "guide", "clerk", "attendant")

# ── STAFF DUTY POST ──────────────────────────────────────────────────────────────────────────────────
# Same mode split as the other hybrid safety nets: OFF in research_learned so the paper measures the raw
# policy, ON for the demo. Per-run override ECGP_STAFF_POST=0.
_HYBRID_POST = (not _RESEARCH_MODE) and os.environ.get("ECGP_STAFF_POST", "1") == "1"
# The model outputs that mean "I have nothing for this agent to do". Only these hand over to the post rule.
_POST_IDLE_ACTIONS = ("idle", "observe", "continue")
_POST_CACHE = {}                    # role -> smart_object_id (resolved once per scene build)


# ── PERFORMER SEAT ───────────────────────────────────────────────────────────────────────────────────
# The gig event spawns a musician; this is what makes them actually sit down AT THE INSTRUMENT rather
# than take the nearest bench. Same shape as the staff duty post above.
_PERFORMER_ROLES = ("musician", "performer", "pianist", "singer", "guitarist", "band")
_PERFORMER_SEAT_CACHE = {}          # id(scene) -> smart_object_id or None


def _performer_seat(scene):
    """The one seat a performer plays from — a `sit` object whose name marks it as the instrument's
    stool (so_piano_chair / "Piano Stool"). Returns None when the venue has no such seat, in which
    case the musician just follows the ordinary role directive into the zone."""
    key = id(scene)
    if key in _PERFORMER_SEAT_CACHE:
        oid = _PERFORMER_SEAT_CACHE[key]
        return scene.objects.get(oid) if oid else None
    found = None
    for o in scene.objects.values():
        if getattr(o, "removed", False):
            continue
        if not any(getattr(a, "action", None) == "sit" for a in getattr(o, "affordances", [])):
            continue
        tag = (str(o.id) + " " + str(getattr(o, "display_name", "") or "")).lower()
        if any(k in tag for k in ("piano", "stool", "performer", "organ", "keyboard")):
            found = o
            break
    _PERFORMER_SEAT_CACHE[key] = found.id if found is not None else None
    return found


# ── LIVE SOCIAL MEMORY ───────────────────────────────────────────────────────────────────────────────
# scene.social_rel is the PERSISTENT relationship store (cs.Relationship keyed by sorted pair).
# cs.record_interaction was only ever called by the teacher, and w.relationships is rebuilt from
# authored groups each tick — so at runtime nobody could ever BECOME a friend. These make the loop
# live: completed conversations raise affinity (talk +0.05; friend at 0.60 with >=5 interactions,
# ecgp/social.py's own thresholds), and existing ties spark spontaneous chats.
_CHAT_RECORDED = {}                 # pair key -> tick a conversation was last folded into affinity
_CHAT_STARTED = {}                  # pair key -> tick a spontaneous chat was last kicked off
_CHAT_RECORD_CD = 6                 # one affinity credit per ~8 ticks of standing together
_CHAT_START_CD = 45                 # a given pair strikes up a new chat at most every ~45 ticks
_CHAT_RANGE2 = 1.6 ** 2             # "actually talking" = within conversation range
_CHAT_MIN_AFFINITY = 0.22
_TALK_RELIEF = 22.0                 # loneliness removed per completed conversation (drift is +0.8/tick,
                                    # so one chat buys ~27 ticks of company — a visible, non-permanent dip)           # near-acquaintance and up drift into spontaneous chats


class _RelStore:                    # duck-typed `world` for cs.record_interaction
    def __init__(self, rels):
        self.relationships = rels


def _social_store(scene):
    store = getattr(scene, "social_rel", None)
    if store is None:
        store = scene.social_rel = {}
    return store


def _social_tick(scene, actions):
    """Runs once per tick over the FINAL action list: (a) any meet pair standing together earns
    affinity, with story-log lines on stranger->acquaintance->friend promotions; (b) idle agents
    with an existing tie in the same zone start a chat (their idle action becomes a meet)."""
    tick = getattr(scene, "tick_no", 0)
    store = _social_store(scene)
    holder = _RelStore(store)

    # (a) fold completed conversations into affinity
    for a in actions:
        if a.get("action") != "meet":
            continue
        sa, sb = a.get("agent_id"), a.get("target_agent_id")
        A, B = scene.agents.get(sa), scene.agents.get(sb)
        pa = getattr(A, "pos", None) if A is not None else None
        pb = getattr(B, "pos", None) if B is not None else None
        if pa is None or pb is None:
            continue
        if (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 > _CHAT_RANGE2:
            continue                                   # still walking over — not a conversation yet
        k = cs._key(sa, sb)
        if tick - _CHAT_RECORDED.get(k, -10 ** 9) < _CHAT_RECORD_CD:
            continue
        before = store[k].relationship_type if k in store else "stranger"
        # user-directed introductions build the tie faster than ambient chatter (see INTERACTION_DELTA)
        itype = "talk" if a.get("reason") == "social:friends_chat" else "directed_talk"
        rel = cs.record_interaction(holder, sa, sb, itype)
        _CHAT_RECORDED[k] = tick
        if rel is not None and rel.relationship_type != before:
            _olog(scene, f"{getattr(A, 'name', sa)} and {getattr(B, 'name', sb)} "
                         f"are {rel.relationship_type}s now")

    # (b) spontaneous chat between agents who already know each other
    idle = {}
    for a in actions:
        aid = a.get("agent_id")
        ag = scene.agents.get(aid)
        if ag is None or a.get("action") != "idle":
            continue
        role = (getattr(ag, "role", "") or "").lower()
        if _social_role(role) == "staff" or role in _PERFORMER_ROLES:
            continue                                   # staff hold their post; performers perform
        idle[aid] = a
    started = 0
    ids = sorted(idle)
    for i, sa in enumerate(ids):
        if started >= 2:                               # at most two new chats a tick — keep it ambient
            break
        A = scene.agents[sa]
        for sb in ids[i + 1:]:
            B = scene.agents[sb]
            if A.current_zone != B.current_zone:
                continue
            k = cs._key(sa, sb)
            rel = store.get(k)
            if rel is None or rel.affinity < _CHAT_MIN_AFFINITY:
                continue
            if tick - _CHAT_STARTED.get(k, -10 ** 9) < _CHAT_START_CD:
                continue
            # Rebuild the action in place, but KEEP THE PROVENANCE FIELDS. `.clear()` wiped
            # semantic_origin, and _log_origin_breakdown counts an action with no (or an unknown)
            # origin as UNTAGGED — so every spontaneous chat silently subtracted itself from the
            # provenance percentages the paper cites. This rule is a deterministic hybrid rule
            # overriding a do-nothing model output, so it tags itself the same way the staff-post
            # and give-up branches do.
            idle[sa].clear()
            idle[sa].update({"agent_id": sa, "action": "meet", "target_agent_id": sb,
                             "reason": "social:friends_chat", "source": "interaction",
                             "priority": 1, "semantic_origin": "deterministic_execution"})
            _CHAT_STARTED[k] = tick
            started += 1
            break


def _post_object(scene, agent):
    """The smart object a staff member stands at when idle.

    Preference order: the role's authored `post_object`, then any `talk` object in an access/reception zone,
    then any `talk` object at all. `talk` is deliberate — `so_reception` already offers it, it is in the
    reception zone's allowed_actions, and it has no entry in the role-compatibility table, so it passes the
    option masks unchanged. `work` would need the object reclassified as a workstation AND the zone's
    allowed_actions widened, and it routes through the tight 'near' targeting that packs staff 0.3 units
    apart; `talk` uses the 8-slot ring, so posted staff spread out instead of jostling.
    """
    role = (getattr(agent, "role", "") or "").lower()
    if role in _POST_CACHE:
        oid = _POST_CACHE[role]
        return scene.objects.get(oid) if oid else None

    authored = (getattr(agent, "post_object", "") or "").strip()
    chosen = scene.objects.get(authored) if authored else None
    if chosen is None:
        def _talks(o):
            return (not getattr(o, "removed", False)
                    and any(getattr(a, "action", None) == "talk" for a in getattr(o, "affordances", [])))
        access = [z.id for z in scene.zones.values()
                  if any(f in (getattr(z, "zone_function", None) or []) for f in ("access", "waiting"))]
        chosen = next((o for o in scene.objects.values() if _talks(o) and o.zone_id in access), None) \
              or next((o for o in scene.objects.values() if _talks(o)), None)
    _POST_CACHE[role] = chosen.id if chosen is not None else None
    if chosen is None:
        log.info(f"[ecgp/post] no duty post found for role={role!r} — staff will idle where they are")
    else:
        log.info(f"[ecgp/post] role={role!r} posts at {chosen.id} @{chosen.zone_id}")
    return chosen


def get_policy():
    """The deployed director. `_POLICY` is a process-lifetime singleton and `_CKPT` is resolved at import,
    so changing ECGP_CKPT after the first tick has no effect — restart the server to change models."""
    global _POLICY
    if _POLICY is None:
        p = ECGPPolicy(_CKPT)                           # auto-detects arch + encoding contract from the ckpt
        # cd_v3_nomp_* (layers=0) is the paper's no-message-passing NEGATIVE CONTROL. It loads perfectly
        # cleanly — ECGPNet just builds an empty layer list — so a mistyped ECGP_CKPT would ship the
        # ablation as the product with no symptom whatsoever. Refuse it here rather than in ECGPPolicy,
        # which the evaluation scripts legitimately use to score that ablation.
        if p.layers == 0 and os.environ.get("ECGP_ALLOW_ABLATION") != "1":
            raise RuntimeError(f"[ecgp] refusing to deploy an ablation checkpoint (layers=0): {_CKPT}. "
                               f"Set ECGP_ALLOW_ABLATION=1 only if this is deliberate.")
        log.info(f"[ecgp/model] DEPLOYED {os.path.relpath(_CKPT, os.path.join(os.path.dirname(__file__), '..', '..'))} "
                 f"| directive_target_v3={p.directive_target_v3} role_mask_v2={p.role_mask_v2} "
                 f"tri_state_v2={p.tri_state_v2} | mode={_MODE}")
        _POLICY = p
    return _POLICY


def _social_role(role: str) -> str:
    r = (role or "").lower()
    if "security" in r or "guard" in r:
        return "security"
    if "guide" in r:
        return "guide"
    if any(k in r for k in ("tourist", "traveler", "visitor_tour")):
        return "tourist"
    if any(k in r for k in _STAFF_KEYS):
        return "staff"
    return "visitor"


def _rects_touch(a, b, eps=0.15):
    """Two gap-free grid zones share an edge → adjacent. x,y = BOTTOM-LEFT corner (the zone contract /
    grid_layout convention), NOT the center — getting this wrong makes the whole graph disconnected."""
    ax0, ax1 = a["x"], a["x"] + a["w"]
    ay0, ay1 = a["y"], a["y"] + a["h"]
    bx0, bx1 = b["x"], b["x"] + b["w"]
    by0, by1 = b["y"], b["y"] + b["h"]
    x_over = min(ax1, bx1) - max(ax0, bx0) > -eps
    y_over = min(ay1, by1) - max(ay0, by0) > -eps
    x_abut = abs(ax1 - bx0) < eps or abs(bx1 - ax0) < eps
    y_abut = abs(ay1 - by0) < eps or abs(by1 - ay0) < eps
    return (x_abut and y_over) or (y_abut and x_over)


def _adjacency(zone_dicts):
    adj = {z["id"]: [] for z in zone_dicts}
    for i, a in enumerate(zone_dicts):
        for b in zone_dicts[i + 1:]:
            if _rects_touch(a, b):
                adj[a["id"]].append(b["id"]); adj[b["id"]].append(a["id"])
    return adj


def build_world(scene, zone_dicts):
    """dsag SceneModel (+ raw zone dicts with x/y/w/h) → EcgpWorld (effective state for ECGP)."""
    w = EcgpWorld()
    w.scenario_id = "live"; w.trajectory_id = "live"
    zmeta = {z["id"]: z for z in zone_dicts}
    adj = _adjacency(zone_dicts)
    for zid, z in scene.zones.items():
        fn = (z.zone_function[0] if z.zone_function else "activity")
        if fn not in V.ZONE_FUNCTIONS_V1:
            fn = "UNK"
        zd = zmeta.get(zid)                              # x,y = bottom-left corner → true center = corner + half
        if zd is not None:
            cx, cy = zd["x"] + zd.get("w", 0) / 2, zd["y"] + zd.get("h", 0) / 2
        else:
            cx, cy = z.center
        w.add_zone(EZone(id=zid, zone_function=fn, center=(cx, cy),
                         is_exit=(fn == "access"), adjacency=adj.get(zid, [])))
    for aid, a in scene.agents.items():
        z = w.zones.get(a.current_zone)
        cx, cy = (z.center if z else (0.0, 0.0))
        jx = ((hash(aid) % 100) / 100.0 - 0.5) * 0.8    # deterministic in-zone jitter (distinct positions)
        jy = ((hash(aid + "y") % 100) / 100.0 - 0.5) * 0.8
        needs = {n: float(getattr(a.needs, n, 30.0)) for n in V.NEEDS_V1}
        w.add_agent(EAgent(id=aid, name=getattr(a, "name", aid), zone=a.current_zone,
                           pos=(cx + jx, cy + jy), needs=needs, social_role=_social_role(a.role),
                           is_staff=_social_role(a.role) == "staff",
                           group_id=getattr(a, "group_id", None),   # so co-grouped agents surface talk/group options
                           last_action=getattr(a, "last_action", "idle")))
    for oid, o in scene.objects.items():                # reuse the grounded dsag SmartObjects directly
        if getattr(o, "removed", False):
            continue                                     # removed/unavailable → excluded from candidate options
        w.add_object(o)
    if _CLUSTER_MASK_V2:                                 # V2.1 dining-cluster macro-option (graph-side only)
        for cid, cluster in dsag_bridge.build_dining_clusters(scene).items():
            w.add_object(cluster)
    # SOCIAL UNITS (item 6): authored groups + pairwise relationships, so co-grouped agents surface
    # talk/help + group-target options in the ECGP option builder (O2), enabling family cohesion.
    for gid, gd in (getattr(scene, "social_groups", {}) or {}).items():
        members = [m for m in gd.get("members", []) if m in w.agents]
        if len(members) < 2:
            continue
        w.add_group(EGroup(id=gid, group_type=gd.get("type", "friend"),
                           members=list(members), leader_id=members[0], cohesion=0.7))
        rtype = gd.get("type", "friend")
        store = _social_store(scene)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                # setdefault: an authored tie seeds the PERSISTENT store once; after that the
                # developed relationship (with its accumulated interaction_count) is the truth.
                store.setdefault(cs._key(members[i], members[j]),
                                 cs.make_relationship(members[i], members[j], rtype))
    # publish every DEVELOPED relationship into this tick's graph — same objects, so anything
    # recorded during the tick lands straight back in the store.
    w.relationships.update(_social_store(scene))
    _apply_events(w, scene)
    _apply_directives(w, scene)
    return w


def _apply_directives(w, scene):
    """Expose live director commands to the MODEL as directive overlays. Returns the number of agents.

    "Drew go to the bar" / "staff gather at the counter" arrive as dsag agent_directive / role_directive
    patches. Nothing used to put them into the ECGP world — `_apply_events` only emits zone_hazard,
    zone_attraction and disable_affordance — so `overlays.directive(agent_id)` was None for every agent on
    every tick. Two consequences, both measured:

      1. CrowdDirect v3's directive-target feature (options.py:194/:214 compare the option's target against
         `overlays.directive(agent.id)`) could never fire live. Deploying v3 without this is cosmetic: the
         one thing v3 adds over v2 would be dormant.
      2. Worse and independent of v3: a plain {agent, zone} directive was dropped ENTIRELY in ECGP mode.
         `dsag.patch.agent_directive_target` — the resolver for it — is only ever called by the symbolic
         rule engine (dsag/scene.py -> dsag/behavior.py). live_bridge resolves agents_leaving,
         personal_action_directives and role_directive_targets deterministically, but never a plain
         zone directive, so the single most common director command did nothing at all here.

    The overlay MUST be written the way every training world writes it (crowddirect_v3/cd_ext.py,
    evaluation/e1_runner.py): apply_patch(pid, [], ttl) — an empty op list — plus a direct `_directive`
    append carrying the destination. Routing it through an EOperation does NOT work and fails silently:
    `EcgpWorld._op_to_overlay` sets either the subject (`agent`) or the destination (`zone`), never both,
    so `overlays._apply_overlay` computes `tgt = op.get("zone") or ...` -> None and `directive()` still
    returns None while `_directive` looks populated. Assert on `directive(aid)`, never on `_directive`.

    Deliberately NOT included: personal-action directives ("Alex sit down" — {agent, action}, no zone, so
    the resolver returns None) and emergencies, where evacuation overrides every standing order.
    """
    if getattr(scene, "is_emergency", lambda: False)():
        return 0
    patches = getattr(scene, "active_patches", []) or []
    if not patches:
        return 0
    from dsag import patch as dp
    roles = dp.role_directive_targets(patches, scene)            # {role: zone_id}
    directed = {}
    for aid, agent in scene.agents.items():
        if aid not in w.agents:
            continue
        tgt = dp.agent_directive_target(patches, agent, scene)   # {agent,zone} and {agent,target_agent}
        if tgt is None and roles:
            role = (getattr(agent, "role", "") or "").lower()
            tgt = next((z for r, z in roles.items() if r and r in role), None)
        if tgt and tgt in w.zones:
            directed[aid] = tgt
    if not directed:
        return 0
    pid = "live_directive"
    # An empty-op patch is exactly how a directive is represented in training: the encoder emits an event
    # node with no edges, and the destination lives in the overlay rather than in an operation.
    w.events[pid] = EEvent(id=pid, is_evacuation=False, ttl=25, severity=0.2)
    w.overlays.apply_patch(pid, [], ttl=25)
    for aid, zid in sorted(directed.items()):
        w.overlays._seq += 1
        w.overlays._directive.setdefault(aid, []).append((w.overlays._seq, pid, zid))
    log.info(f"[ecgp/directive] {len(directed)} agent(s) carry an ordered destination into the graph: "
             + ", ".join(f"{a}->{z}" for a, z in sorted(directed.items())[:6])
             + (" ..." if len(directed) > 6 else ""))
    return len(directed)


def _apply_events(w, scene):
    """Map active dsag ScenePatches → ECGP overlays: emergency → hazard on interior zones (everyone
    flees to an exit); zone_attraction → attraction; disable_affordance ("toilet closed") → disable that
    action at matching zones + their objects, so options mark it infeasible and agents pick alternatives."""
    patches = getattr(scene, "active_patches", []) or []
    if getattr(scene, "is_emergency", lambda: False)():
        interior = [z.id for z in w.zones.values() if not z.is_exit]
        ops = [EOperation(id=f"haz_{zid}", event_id="live_fire", op_kind="zone_hazard",
                          magnitude=90, severity=0.9, target_type="zone", target_id=zid)
               for zid in interior]
        if ops:
            w.add_event(EEvent(id="live_fire", is_evacuation=True, ttl=30, severity=0.9), ops)
        return
    from dsag import patch as dp
    ops = []
    haz_ops = []
    for p in patches:
        for op in getattr(p, "ops", []):
            if op.op == "zone_attraction" and (op.delta or 0) > 0:
                for z in scene.zones.values():
                    if dp.zone_matches(op, z):
                        ops.append(EOperation(id=f"attr_{z.id}_{len(ops)}", event_id="live_evt",
                                              op_kind="zone_attraction", magnitude=op.delta,
                                              target_type="zone", target_id=z.id))
            # OBJECT ATTRACTION — point the crowd at ONE object (the free-sample tray, the busy counter)
            # rather than a whole room. Previously unreachable: nothing emitted it, so `object_attraction`
            # existed in the vocabulary and the overlays but never in a live graph. The encoder does not
            # feature attraction on object nodes at all; what this actually buys is `event_relevant=1` on
            # that object's options plus its insertion into the event bucket (see options._event_targets),
            # which is the only channel an object-scoped event has into the policy. MAGNITUDE IS NOT READ
            # anywhere for objects — do not try to tune behaviour with it.
            elif op.op == "object_attraction" and op.object:
                o = scene.objects.get(op.object)
                if o is not None:
                    ops.append(EOperation(id=f"oattr_{o.id}_{len(ops)}", event_id="live_evt",
                                          op_kind="object_attraction", magnitude=(op.delta or 0),
                                          target_type="object", target_id=o.id))
            # NON-EVACUATION ZONE HAZARD — the trained "keepaway" shape (cd_ext spill: magnitude 60,
            # severity 0.45, is_evacuation False). Distinct from the emergency branch above, which is a
            # whole-building evacuation. Carried on its OWN EEvent so its severity is not flattened into
            # the shared live_evt (severity is read via hazard_severity; the shared event pins it at 0.3).
            # HONEST LIMIT: for OBJECT options this is a SOFT flag, not a mask — under tri-state feasibility
            # a hazard sets temporarily_unavailable and leaves feasible=1.0. Only ZONE-walk options are hard
            # masked. Pair it with disable_affordance if you actually need the room shut.
            elif op.op == "zone_hazard":
                for z in scene.zones.values():
                    if dp.zone_matches(op, z):
                        haz_ops.append(EOperation(
                            id=f"haz_{z.id}_{len(haz_ops)}", event_id="live_hazard", op_kind="zone_hazard",
                            magnitude=(op.delta or 60), severity=float(getattr(op, "severity", None) or 0.45),
                            target_type="zone", target_id=z.id))
            elif op.op in ("disable_affordance", "enable_affordance") and op.action:
                if op.op == "enable_affordance":
                    continue                              # nothing to block for an enable
                # OBJECT-SCOPED disable: one broken machine, not the whole room. `PatchOp.object` already
                # existed (remove_object uses it) and the overlay layer already keys on it; only this
                # fan-out never read it, so "the coffee machine is broken" shut down every drink source in
                # the zone and the crowd had no alternative to move to — the opposite of the intent.
                if op.object:
                    o = scene.objects.get(op.object)
                    if o is not None:
                        ops.append(EOperation(id=f"dis_{o.id}_{op.action}_{len(ops)}", event_id="live_evt",
                                              op_kind="disable_affordance", magnitude=0.0,
                                              target_type="object", target_id=o.id, action=op.action))
                    continue
                for z in scene.zones.values():
                    scoped = (op.zone is None and op.zone_function is None) or dp.zone_matches(op, z)
                    if not scoped:
                        continue
                    ops.append(EOperation(id=f"dis_{z.id}_{op.action}_{len(ops)}", event_id="live_evt",
                                          op_kind="disable_affordance", magnitude=0.0,
                                          target_type="zone", target_id=z.id, action=op.action))
                    for o in scene.objects.values():      # also disable at the objects offering it there
                        if o.zone_id == z.id and any(a.action == op.action for a in o.affordances):
                            ops.append(EOperation(id=f"dis_{o.id}_{op.action}_{len(ops)}", event_id="live_evt",
                                                  op_kind="disable_affordance", magnitude=0.0,
                                                  target_type="object", target_id=o.id, action=op.action))
    if ops:
        w.add_event(EEvent(id="live_evt", is_evacuation=False, ttl=20, severity=0.3), ops)
    if haz_ops:
        sev = max(o.severity for o in haz_ops)
        w.add_event(EEvent(id="live_hazard", is_evacuation=False, ttl=20, severity=sev), haz_ops)


def _urgent_tier1(agent):
    """A genuinely urgent physiological need — exempt from a directive for one tick (satisfy, then rejoin),
    matching the standing-order Tier-1 override in the rule engine."""
    n = agent.needs
    return (getattr(n, "bladder", 0) > 75 or getattr(n, "thirst", 0) > 80 or
            getattr(n, "hunger", 0) > 85 or getattr(n, "energy", 100) < 15)


# Tier-1 need -> the action that relieves it. Drives the safe_demo_hybrid need-relief SAFETY-NET (_HYBRID_NEED_RELIEF).
_NEED_ACTION = {"bladder": "relieve", "thirst": "drink", "hunger": "eat", "energy": "rest"}

# Need-relief SAFETY-NET bars — deliberately HIGHER than _urgent_tier1's (75/80/85/15), so the deterministic
# override only rescues GENUINELY DESPERATE agents the weak V2 GNN failed to serve; moderate needs are left to
# the learned policy so it visibly drives most of the crowd (the user wanted the GNN to show more, not to be
# drowned out by hybrid_fallback). All tunable per-run for demo balance.
_RELIEF_BARS = {"bladder": float(os.environ.get("ECGP_RELIEF_BLADDER", "80")),
                "thirst":  float(os.environ.get("ECGP_RELIEF_THIRST",  "82")),
                "hunger":  float(os.environ.get("ECGP_RELIEF_HUNGER",  "85")),
                "energy":  float(os.environ.get("ECGP_RELIEF_ENERGY",  "18"))}   # energy is inverted (below this)
_RELIEF_GRACE = int(os.environ.get("ECGP_RELIEF_GRACE", "3"))   # ticks a desperate agent gets to self-correct first
_NEED_GRACE = {}                                                # agent_id -> consecutive desperate-but-unrescued ticks

def _need_past_relief_bar(agent, need):
    n = getattr(agent, "needs", None)
    if n is None:
        return False
    v = getattr(n, need, 100 if need == "energy" else 0)
    return v < _RELIEF_BARS[need] if need == "energy" else v > _RELIEF_BARS[need]

def _relief_need(agent):
    """The single most-DESPERATE need past its (high) relief bar, or None — ranked by how far past. Higher
    bars than _urgent_tier1 so this fires only as a genuine last-resort safety net, not for ordinary needs."""
    cand = [( (_RELIEF_BARS[need] - getattr(agent.needs, need, 100)) if need == "energy"
              else (getattr(agent.needs, need, 0) - _RELIEF_BARS[need]), need )
            for need in ("bladder", "thirst", "hunger", "energy") if _need_past_relief_bar(agent, need)]
    if not cand:
        return None
    cand.sort(reverse=True)
    return cand[0][1]

def _find_relief_object(scene, action, near_zone=None):
    """A reachable, zone-legal object offering `action` for deterministic need-relief. Prefers a FREE one in
    the agent's zone (via _find_free_object_for_action); if every one is occupied, returns the nearest anyway
    so the agent walks over and QUEUES (capacity is enforced at the object, not by hiding it from a needy
    agent). None only when the scene has NO object offering the action at all."""
    free = _find_free_object_for_action(scene, action, near_zone)
    if free is not None:
        return free
    best = None
    for oid, o in scene.objects.items():
        if getattr(o, "removed", False):
            continue
        if not any(a.action == action for a in o.affordances):
            continue
        if action == "sit" and not _is_seat_category(o):
            continue
        if not _action_ok(scene, o.zone_id, action):
            continue
        score = (0 if near_zone and o.zone_id == near_zone else 1, oid)
        if best is None or score < best[0]:
            best = (score, o)
    return best[1] if best else None


# A patch carrying this flag came from the AMBIENT world simulation (ecgp.runtime.ambient), not from a human
# director. See _attracted_zones for why the distinction has to exist.
AMBIENT_PATCH_FLAG = "ambient"


def is_ambient_patch(p) -> bool:
    return bool(getattr(p, "origin", "") == AMBIENT_PATCH_FLAG)


def _attracted_zones(scene):
    """Zones a positive zone_attraction patch points at (a USER 'gather here' / 'party here' directive).
    Empty when no attraction is active. Routes every op.zone reference through the single resolver
    (dsag.patch.resolve_zone_reference, via zone_matches(..., all_zones=...)) so a party aimed at ONE zone
    never also pulls in a DIFFERENT zone that merely shares a zone_type or keyword — without the shared
    resolver, a party aimed at 'lounge' also matches 'window_seats' when that zone's TYPE is 'lounge'.

    AMBIENT PATCHES ARE EXCLUDED. This list feeds the deterministic party/gather branch, which is a
    hybrid_fallback that seizes EVERY eligible agent for the patch's whole ttl and drops their commitments.
    That is the right response to a human typing "everyone to the stage"; it is emphatically the wrong
    response to the world simulating a rush hour, which must remain a soft pull the learned policy is free
    to weigh against needs. Without this filter every ambient attraction event would bypass CrowdDirect v3
    for its entire duration — the crowd would look scripted and the model would stop deciding.
    """
    from dsag import patch as _dp
    patches = [p for p in (getattr(scene, "active_patches", []) or []) if not is_ambient_patch(p)]
    all_zones = list(scene.zones.values())
    return [z for z in all_zones if _dp.zone_attraction(patches, z, all_zones) > 0]


# ── R2 party, R4 queueing, R5 frustration-leave helpers ──────────────────────
_FRUSTRATION = {}                                     # agent_id -> consecutive ticks with 2+ SEVERELY unmet needs
_LEAVING = {}                                         # agent_id -> ticks spent walking out (then removed from sim)
# Despawn is confirmed by Unity, not assumed on a timer. A blind fixed-tick timer breaks when the exit is
# physically blocked: the server stops driving the agent while Unity still shows it stuck there. Unity
# confirms the actual portal exit (CrowdDirector.SendAgentDespawned -> server 'agent_despawned' message ->
# this set) and the tick loop removes a _LEAVING agent as soon as that arrives, falling back to a generous
# tick timer only if the confirmation never comes (disconnect), so an agent is never driven forever.
_CONFIRMED_DESPAWNED = set()


def confirm_despawned(agent_id: str):
    """Call when Unity confirms an agent actually crossed the exit portal (server message handler)."""
    _CONFIRMED_DESPAWNED.add(agent_id)


def reset_scene_state():
    """Drop every piece of per-agent / per-scene bookkeeping. MUST be called whenever a new scene is created.

    All of these dicts are keyed by AGENT ID, and agent ids restart at `agent_0` for every scene — so without
    this a fresh scene silently inherits the previous one's state. The visible symptom is an agent walking to
    a smart object from the LAST scene: `_COMMIT` still holds `{"target_id": "so_coffee_machine", ...}` from a
    venue that no longer exists, so the commitment-honouring path re-issues it before the model is ever asked.
    Chain state, leave/despawn counters and party slots leak the same way."""
    n = (len(_COMMIT) + len(_LAST_TARGET) + len(_LAST_TALK) + len(_DIRECTED_ACTION_OBJ) + len(_NEED_GRACE)
         + len(_FRUSTRATION) + len(_LEAVING) + len(_CONFIRMED_DESPAWNED) + len(_PARTY_SLOT))
    for d in (_CHAT_RECORDED, _CHAT_STARTED,
              _COMMIT, _LAST_TARGET, _LAST_TALK, _DIRECTED_ACTION_OBJ, _NEED_GRACE,
              _FRUSTRATION, _LEAVING, _PARTY_SLOT, _QUEUE_LOCK):
        d.clear()
    _CONFIRMED_DESPAWNED.clear()
    # keyed by ROLE, not agent id, but it caches an OBJECT ID from the old scene — which the next scene
    # will not contain, so a stale entry posts staff at nothing.
    _POST_CACHE.clear()
    _PERFORMER_SEAT_CACHE.clear()
    if n:
        log.info(f"[ecgp] scene reset: dropped {n} stale per-agent entries from the previous scene")
    return n


_LEAVE_TIMEOUT_TICKS = int(os.environ.get("ECGP_LEAVE_TIMEOUT_TICKS", "40"))   # generous SAFETY NET only
                                                                                # (~2 min @ 3s/tick) — normal
                                                                                # exits are retired by the
                                                                                # confirmation below, almost
                                                                                # always long before this fires


def _maybe_retire(aid, to_remove):
    """Retire a leaving agent from the SERVER's bookkeeping only once Unity has CONFIRMED it actually crossed
    the exit portal — never on a blind tick timer alone. A blind timer assumes Unity always finishes the walk
    in time; if the door was jammed that assumption broke: the server forgot the agent (stopped driving it)
    while Unity still showed it stuck there, un-driven, forever. The tick counter remains as a generous
    fallback (_LEAVE_TIMEOUT_TICKS) so a lost confirmation message (disconnect) can't leak an agent forever —
    but under normal operation confirmation retires it almost immediately after it actually exits."""
    if aid in _CONFIRMED_DESPAWNED:
        _CONFIRMED_DESPAWNED.discard(aid)
        to_remove.append(aid)
    elif _LEAVING.get(aid, 0) >= _LEAVE_TIMEOUT_TICKS:
        log.warning(f"[ecgp] {aid} retired by SAFETY-NET timeout (no exit confirmation after "
                    f"{_LEAVE_TIMEOUT_TICKS} ticks) — was it physically able to reach the exit?")
        to_remove.append(aid)
_PARTY_SLOT = {}                                      # agent_id -> (zone_id, slot_idx): its STABLE party gathering
                                                      # slot, so the crowd spreads across the zone (freed on clear)
_PARTY_SLOT_SPACING = float(os.environ.get("ECGP_PARTY_SLOT_SPACING", "0.75"))   # grid pitch between gather slots
_PARTY_WALL_INSET   = 0.8                             # keep slots off the walls (>= _clamp_zone margin so no snap)
_FRUST_LIMIT = int(os.environ.get("ECGP_FRUSTRATION_TICKS", "20"))   # patience before giving up (raised)
_LEAVE_TICKS = int(os.environ.get("ECGP_LEAVE_TICKS", "8"))          # ticks to walk out before despawn
_QUEUE_ENABLED = os.environ.get("ECGP_QUEUE", "1") == "1"
_USE_RADIUS = float(os.environ.get("ECGP_USE_RADIUS", "1.1"))    # how close an agent must be to count as USING
_CHAINS = os.environ.get("ECGP_CHAINS", "1") == "1"              # Phase B: acquire@counter -> sit@chair -> consume
_ARRIVE_TIMEOUT = int(os.environ.get("ECGP_ARRIVE_TIMEOUT", "8"))   # walking ticks before an unreachable target is dropped
                                                                    # (short: a blocked agent re-decides quickly)


# ── per-zone ACTION GATING: an animation only happens where it makes sense ───────────────────────────
# Claude designs `allowed_actions` per zone at scene generation; zones without it fall back to defaults by
# zone function. Movement/control actions are never gated. This kills "sitting in the toilet" / "eating in
# the restroom" at the DECISION level, so the wrong clip can never play in the wrong room.
_UNGATED = {"walk", "idle", "continue", "leave", "help", "UNK"}
_FN_ALLOWED = {
    "access":   {"observe", "talk"},
    "service":  {"eat", "drink", "observe", "talk", "work"},
    "seating":  {"sit", "rest", "eat", "drink", "talk", "observe"},
    "hygiene":  {"relieve", "wash"},
    "work":     {"work", "sit", "talk", "observe"},
    "activity": {"observe", "dance", "talk", "drink", "sit"},
}


def _build_zone_allowed(scene, zone_dicts):
    allowed = {}
    for z in zone_dicts:
        acts = z.get("allowed_actions")
        if acts:                                          # Claude-designed at scene generation
            allowed[z["id"]] = {str(a).strip().lower() for a in acts if a}
        else:                                             # fallback: sensible defaults by zone function
            zi = scene.zones.get(z["id"])
            fn = (zi.zone_function[0] if zi is not None and getattr(zi, "zone_function", None) else "activity")
            allowed[z["id"]] = set(_FN_ALLOWED.get(fn, {"observe", "talk"}))
    scene._zone_allowed = allowed


# Normalize LLM-emitted affordance names that are not distinct action classes onto the real vocabulary,
# so they cannot become a silent no-op through a set-membership mismatch: zone gating compares exact
# strings, and "talk_to_staff" != "talk". (The Unity side's substring check happens to catch this, so the
# gap is server-side only.) talk_to_staff is semantically "talk" with an audience constraint; the
# constraint itself is not enforced — that would need a staff-presence check at resolution time — but the
# normalization keeps the instruction from being invisible.
_ACTION_ALIAS = {"talk_to_staff": "talk"}


def _norm_action(action):
    return _ACTION_ALIAS.get(action, action)


def _action_ok(scene, zid, action):
    action = _norm_action(action)
    if action in _UNGATED:
        return True
    al = (getattr(scene, "_zone_allowed", None) or {}).get(zid)
    return True if al is None else action in al


def _clamp_zone(scene, zid, x, y, margin=0.75):   # > nav wall band half (0.4) + body radius, so a stand point
                                                  # can never sit at the band edge and snap across the wall
    """Clamp a computed stand/queue/spread point into its zone's rect (inset from the walls) so a ring point
    never lands inside a wall band or in the neighbouring room. Zone rects are stashed per tick from the
    authoritative zone dicts (corner convention). No rect known -> point unchanged."""
    r = (getattr(scene, "_zone_rects", None) or {}).get(zid)
    if not r:
        return x, y
    zx, zy, zw, zh = r
    return (min(max(x, zx + margin), zx + zw - margin),
            min(max(y, zy + margin), zy + zh - margin))


def _critical_need(agent):
    """A need so urgent even a party can't wait. Bladder matches _urgent_tier1's bar (75), NOT a higher one —
    real bug found from a live log: an agent already QUEUEING for the toilet (bladder>75, no _COMMIT
    protecting the queue slot — see 'QUEUE (R4)') got pulled into a party at the OLD higher bar (92) and,
    since a party can run for its full ttl (often 60+ ticks), never got back to the toilet — it left the
    scene via an unrelated family-leave, having never relieved itself. A bathroom need already urgent enough
    to have started a real toilet queue must never be interrupted by an ambient event; hunger/energy stay on
    the higher bar (a party pulls the moderately hungry/tired — that's intentional and not the reported bug)."""
    n = agent.needs
    return getattr(n, "bladder", 0) > 75 or getattr(n, "energy", 100) < 8


def _unmet_need_count(agent):
    """How many needs are SEVERELY unmet — deliberately high thresholds so 'giving up and leaving' is rare
    (a normal busy agent has a couple of moderately-high needs; that must NOT count)."""
    n = agent.needs
    return sum([getattr(n, "hunger", 0) > 88, getattr(n, "thirst", 0) > 88, getattr(n, "bladder", 0) > 90,
                getattr(n, "energy", 100) < 12, getattr(n, "stress", 0) > 90, getattr(n, "loneliness", 0) > 90])


def _is_party_zone(scene, zone):
    from dsag import patch as _dp
    all_zones = list(scene.zones.values())
    return "dance" in _dp.zone_enabled_actions(getattr(scene, "active_patches", []) or [], zone, all_zones)


def _obj_capacity(obj):
    return max(1, int(getattr(obj, "capacity", 1) or 1)) if obj is not None else 1


def _committed_object_users(oid, exclude=None):
    out = []
    for a, c in _COMMIT.items():
        if a == exclude:
            continue
        cur = _com_cur(c)
        if cur.get("target_type") == "object" and cur.get("target_id") == oid:
            out.append(a)
    return out


def _queue_target(world, oid, aid):
    """A stable waiting spot BEHIND an object for an agent queueing to use it."""
    o = world.objects.get(oid)
    if o is None:
        return None
    slot = hash(f"q|{oid}|{aid}") % 6
    ang = (hash(oid) % 360) * math.pi / 180.0 + math.pi + (slot % 3 - 1) * 0.35
    r = 1.0 + (slot // 3) * 0.6
    return (o.pos[0] + r * math.cos(ang), o.pos[1] + r * math.sin(ang))


def _zone_spread(center, aid):
    """A distinct in-zone position around the centre, so a gathered crowd clusters instead of stacking."""
    ang = 2 * math.pi * (hash(aid) % 12) / 12.0
    r = 0.3 + (hash(aid + "r") % 90) / 100.0
    return (center[0] + r * math.cos(ang), center[1] + r * math.sin(ang))


def _zone_slots(rect):
    """Gathering slots tiling a zone's interior on a fixed grid, inset from the walls (corner-convention rect
    zx,zy,zw,zh). Row-major + deterministic so per-agent slot assignment is stable across ticks; the COUNT is
    the zone's party CAPACITY. Returns [] for an unknown/degenerate rect (then nobody is pulled in)."""
    if not rect:
        return []
    zx, zy, zw, zh = rect
    iw = max(0.0, zw - 2 * _PARTY_WALL_INSET)
    ih = max(0.0, zh - 2 * _PARTY_WALL_INSET)
    cols = max(1, int(iw / _PARTY_SLOT_SPACING) + 1)
    rows = max(1, int(ih / _PARTY_SLOT_SPACING) + 1)
    x0, y0 = zx + _PARTY_WALL_INSET, zy + _PARTY_WALL_INSET
    return [(x0 + iw * (c + 0.5) / cols, y0 + ih * (r + 0.5) / rows)
            for r in range(rows) for c in range(cols)]


def _role_targets(scene):
    """{role: zone_id} from active role_directive ops — every agent of that role is routed there (spawned
    firefighters/medics → the hazard zone), overriding even a global evacuation for those responders."""
    from dsag import patch as _dp
    return _dp.role_directive_targets(getattr(scene, "active_patches", []) or [], scene)


_OBJECT_LIFECYCLE = os.environ.get("ECGP_OBJECT_LIFECYCLE", "1") == "1"
# single-use consumables cleared after use (a glass/plate/cup)…
_CONSUMABLE_KEYS = ("glass", "cup", "mug", "plate", "dish", "bottle", "tray", "snack", "pastry",
                    "glassware", "utensil")
# …but NEVER the refillable fixtures agents keep returning to (a dispenser/counter/seat stays put)
_FIXTURE_KEYS = ("machine", "fountain", "dispenser", "cooler", "fridge", "tap", "sink", "station", "urn",
                 "counter", "bar", "chair", "table", "sofa", "desk", "exhibit", "toilet", "bench", "shelf")


def _is_consumable(obj):
    blob = f"{getattr(obj, 'object_type', '')} {getattr(obj, 'label', '')} {getattr(obj, 'id', '')}".lower()
    if any(k in blob for k in _FIXTURE_KEYS):
        return False
    return any(k in blob for k in _CONSUMABLE_KEYS)


_REFILL_TICKS = int(os.environ.get("ECGP_REFILL_TICKS", "3"))   # ticks a spent item stays 'empty' before restock


def _queue_removal(scene, oid):
    q = getattr(scene, "pending_mutations", None) or {"removed_objects": [], "spawned_agents": []}
    q["removed_objects"].append(oid)
    scene.pending_mutations = q


def _pretty(name) -> str:
    """A readable object/zone name for the log: 'cafe__coffee_machine' -> 'coffee machine', 'z0_toilet' -> 'toilet'."""
    s = str(name or "item")
    s = s.split("__")[-1]
    parts = s.split("_", 1)
    if len(parts) == 2 and (parts[0][:1] == "z" and parts[0][1:].isdigit() or parts[0].startswith("so")):
        s = parts[1]
    return s.replace("_", " ").strip() or "item"


# A hand-built level's props are named after the ASSET FILE they came from — 'Kitchen_Singles_48x48_207',
# 'FB_counter_2 (1)', 'Museum_Singles_48x48_36'. Prettifying those just yields 'Kitchen Singles 48x48 207',
# which is what used to appear in the log and the inspect panel. Detect that shape so we can fall back to
# something meaningful instead of showing the user a file name.
_ASSET_ID_RE = re.compile(r"(\d+\s*x\s*\d+)|(_\d+(\s*\(\d+\))?$)|(^[A-Z]{2}_)", re.I)

# Last-resort word for an object with no authored name and an asset-id object_type: say what it IS FOR.
# Derived from what the object actually offers, so it stays honest rather than inventing a specific noun.
_ROLE_WORD = {"provider": "counter", "consumable": "refreshments", "seat": "seat", "surface": "table",
              "workstation": "workstation", "sanitation": "washroom fixture", "fixture": "fixture"}
_AFFORDANCE_WORD = {"eat": "food", "drink": "drinks", "sit": "seat", "rest": "bed", "relieve": "washroom fixture",
                    "work": "workstation", "observe": "display", "talk": "desk", "help": "help point"}


def _oname(obj, fallback=None) -> str:
    """The name to SHOW A USER for a smart object. Authored `display_name` first (a prebaked level names its
    own props); else the prettified object_type when that reads like a real word; else what the object is
    for. Never leaks an asset file name into the UI."""
    if obj is None:
        return _pretty(fallback)
    name = (getattr(obj, "display_name", "") or "").strip()
    if name:
        return name
    raw = str(getattr(obj, "object_type", "") or fallback or "")
    if raw and not _ASSET_ID_RE.search(raw):
        return _pretty(raw)
    role = (getattr(obj, "functional_role", "") or "").lower()
    if role in _ROLE_WORD:
        return _ROLE_WORD[role]
    for a in getattr(obj, "affordances", []) or []:
        w = _AFFORDANCE_WORD.get((getattr(a, "action", "") or "").lower())
        if w:
            return w
    return _pretty(raw or fallback)


def _zname(scene, zid) -> str:
    """The name to SHOW A USER for a ZONE — its human label ('Music Hall'), which is what the map already
    draws over that room — not the zone id ('bedroom'). Two log lines about the same spill used to name the
    same room two different ways, because one read `.label` and the other prettified the id."""
    z = (getattr(scene, "zones", None) or {}).get(zid) if zid else None
    return (getattr(z, "label", "") or "").strip() or _pretty(zid)


def _olog(scene, msg):
    """Buffer a smart-object lifecycle message (drained by ecgp_tick, shown in the Unity log in amber)."""
    buf = getattr(scene, "_obj_log", None)
    if buf is None:
        buf = scene._obj_log = []
    buf.append(msg)


def _drain_obj_log(scene) -> list:
    buf = getattr(scene, "_obj_log", None) or []
    scene._obj_log = []
    return buf


_RESTOCK_SEC = float(os.environ.get("ECGP_RESTOCK_SEC", "10"))   # 'be back soon' duration before staff restocks


def _start_restock(scene, obj):
    """Close the zone for restocking: a restriction patch (no eat/drink, repelled) tagged restock:<zone> —
    Unity shows the 'be back soon' sign there — plus a wall-clock timer for the staff restock."""
    import time as _t
    from dsag.patch import ScenePatch, PatchOp
    zid = obj.zone_id
    rs = getattr(scene, "_restocks", None)
    if rs is None:
        rs = scene._restocks = {}
    if zid in rs:
        return                                            # already restocking this zone
    patch = ScenePatch(event_type="hazard", display_name=f"restock:{zid}", global_directive="avoid",
                       ttl=10 ** 6,
                       ops=[PatchOp(op="zone_attraction", zone=zid, delta=-45.0),
                            PatchOp(op="disable_affordance", zone=zid, action="eat"),
                            PatchOp(op="disable_affordance", zone=zid, action="drink")])
    patches = getattr(scene, "active_patches", None)
    if patches is None:
        patches = scene.active_patches = []
    patches.append(patch)
    rs[zid] = _t.time() + _RESTOCK_SEC


_REPAIR_SEC = float(os.environ.get("ECGP_REPAIR_SEC", "8"))   # downtime before a broken object is fixed


def tick_repairs(scene):
    """Advance every broken object: send a staff member to it, and once the downtime has elapsed put it
    BACK IN SERVICE. Returns {fixer_id: action_override} exactly like `tick_spills`, so the repairer is
    locked to the job for that tick and the model cannot redirect them.

    The object is taken out of service with `mark_removed()`, not with a `disable_affordance` op: that op
    is matched by action+zone (`dsag.patch.action_disabled` never reads `op.object`), so an object-scoped
    disable either did nothing or would have shut down every drink source in the room. `mark_removed` is
    the reversible mechanism the option builder already honours in every candidate loop, and Unity hides
    it via object_states.available=false and shows it again on restore."""
    reps = getattr(scene, "_repairs", None)
    if not reps:
        return {}
    overrides, done = {}, []
    for oid, st in list(reps.items()):
        obj = scene.objects.get(oid)
        if obj is None:
            done.append(oid)
            continue
        zone = st.get("zone") or getattr(obj, "zone_id", None)
        if st.get("fixer") is None or st["fixer"] not in scene.agents:
            st["fixer"] = _spill._nearest_free_staff(scene, zone, {})
        fixer = st.get("fixer")
        if fixer and fixer in scene.agents:
            # reason must contain "work" — it selects the Work/thrust clip (see ClipForReason)
            overrides[fixer] = {"agent_id": fixer, "action": "move_to_zone", "zone_id": zone,
                                "smart_object_id": oid, "target_x": obj.pos[0], "target_y": obj.pos[1],
                                "exec_action": "work", "reason": "repair:work"}
            scene.agents[fixer].current_zone = zone
        if time.monotonic() >= st["until"]:
            done.append(oid)
    for oid in done:
        st = reps.pop(oid, None) or {}
        obj = scene.objects.get(oid)
        if obj is not None and getattr(obj, "removed", False):
            obj.restore()
        for pid in st.get("props", []):
            _spill._queue(scene, "removed_objects", pid)
        _olog(scene, f"staff repaired the {_pretty_object(obj) if obj is not None else oid} — working again")
    return overrides


def _pretty_object(obj):
    return (getattr(obj, "display_name", None) or str(getattr(obj, "id", "object"))
            ).replace("so_", "").replace("_", " ")


def break_object(scene, obj, props=None, seconds=None):
    """Take `obj` out of service and register the repair. Shared by the typed outage event and any
    future breakdown source, so there is ONE place that decides what 'broken' means."""
    reps = getattr(scene, "_repairs", None)
    if reps is None:
        reps = scene._repairs = {}
    if obj.id in reps:
        return False
    obj.mark_removed(source="breakdown")
    reps[obj.id] = {"until": time.monotonic() + float(seconds or _REPAIR_SEC),
                    "zone": getattr(obj, "zone_id", None), "fixer": None,
                    "props": list(props or [])}
    _olog(scene, f"the {_pretty_object(obj)} broke down — staff are on the way")
    return True


def _check_restocks(scene):
    """Due restocks: drop the closure patch, refill every stocked source in the zone, reopen + log."""
    import time as _t
    rs = getattr(scene, "_restocks", None)
    if not rs:
        return
    for zid in [z for z, due in rs.items() if _t.time() >= due]:
        rs.pop(zid, None)
        scene.active_patches = [p for p in getattr(scene, "active_patches", [])
                                if getattr(p, "display_name", "") != f"restock:{zid}"]
        for o in scene.objects.values():
            if o.zone_id == zid and getattr(o, "stock", None) is not None:
                # back to THIS source's own full level, not the global default — an authored counter's
                # stock matches the number of food props bound to it as visible portions.
                o.stock = int(getattr(o, "_full_stock", None) or dsag_bridge.FOOD_STOCK)
                o.state = o.states[0] if getattr(o, "states", None) else "default"
                o.state_ticks = 0
        _olog(scene, "staff restocked the food — open again")


_CLEANUP_SEC = float(os.environ.get("ECGP_CLEANUP_SEC", "10"))   # 'closed' duration before staff finish cleaning
_MESS_PROB   = float(os.environ.get("ECGP_MESS_PROB", "0.20"))   # chance a toilet use / finished cup leaves a mess


def _start_cleanup(scene, zid, reason=""):
    """Close a zone for CLEANING (toilet made dirty, cup dropped): a restriction patch tagged cleanup:<zone>
    — a disable_affordance so the server reports it in closed_zones and Unity hangs the 'closed' sign — plus a
    ~10s wall-clock timer for the staff to finish, after which the zone reopens. Mirrors _start_restock."""
    import time as _t
    from dsag.patch import ScenePatch, PatchOp
    if not zid or zid not in getattr(scene, "zones", {}):
        return
    cs = getattr(scene, "_cleanups", None)
    if cs is None:
        cs = scene._cleanups = {}
    rs = getattr(scene, "_restocks", None) or {}
    if zid in cs or zid in rs:
        return                                            # already closed for cleaning / restocking
    patch = ScenePatch(event_type="hazard", display_name=f"cleanup:{zid}", global_directive="avoid",
                       ttl=10 ** 6,
                       ops=[PatchOp(op="zone_attraction", zone=zid, delta=-45.0),
                            PatchOp(op="disable_affordance", zone=zid, action="sit"),
                            PatchOp(op="disable_affordance", zone=zid, action="relieve"),
                            PatchOp(op="disable_affordance", zone=zid, action="drink"),
                            PatchOp(op="disable_affordance", zone=zid, action="eat")])
    patches = getattr(scene, "active_patches", None)
    if patches is None:
        patches = scene.active_patches = []
    patches.append(patch)
    cs[zid] = _t.time() + _CLEANUP_SEC
    # `reason` is an ALREADY-RESOLVED display name from the caller; only the zone-id fallback gets
    # prettified. Prettifying `reason` too was what turned "Coffee Machine" back into an asset string.
    _olog(scene, f"the {reason or _zname(scene, zid)} needs cleaning — closed while staff tidy up")


def _check_event_props(scene):
    """Expire runtime event props (the typed-event decorations that live in the world for a while).

    FIRE: after ~12s the blaze prop despawns, every is_emergency patch is dropped (the spawn/role ops die
    with their patch, which is the normal all-clear semantics), the spawned responders are sent home via
    agent_leave, and an ARRIVAL BURST is queued so the venue visibly refills — evacuation despawns whoever
    crossed the exit portal, and without the burst the room stayed empty for minutes.
    FOOD POP-UPS: the carts stop being targets the moment they leave scene.objects; agents mid-commitment
    to one are recovered by the existing target_removed reinvoke, same as a cleaned spill.
    """
    import time as _t
    now = _t.monotonic()
    fires = getattr(scene, "_fire_props", None)
    if fires:
        for fid, st in list(fires.items()):
            if now < st["until"]:
                continue
            fires.pop(fid, None)
            _spill._queue(scene, "removed_objects", fid)
            scene.active_patches = [p for p in getattr(scene, "active_patches", [])
                                    if not getattr(p, "is_emergency", False)]
            roles = {(getattr(a, "role", "") or "").lower() for a in scene.agents.values()}
            from dsag.patch import ScenePatch, PatchOp
            leave_ops = [PatchOp(op="agent_leave", role=r)
                         for r in ("firefighter", "medic", "police") if r in roles]
            if leave_ops:
                scene.active_patches.append(ScenePatch(event_type="group_directive",
                                                       display_name="responders head out", ttl=25,
                                                       ops=leave_ops))
            scene._arrival_burst = getattr(scene, "_arrival_burst", 0) + 3
            _olog(scene, "the fire is out — all clear, everything back to normal")
    pops = getattr(scene, "_popups", None)
    if pops:
        for oid, until in list(pops.items()):
            if now < until:
                continue
            pops.pop(oid, None)
            scene.objects.pop(oid, None)
            _spill._queue(scene, "removed_objects", oid)
            _olog(scene, "the food stall packed up and left")
    # PURE DECOR with a lifetime (party music notes): no scene.objects entry, nothing to release —
    # just tell Unity to despawn the sprite when the event that owns it ends.
    dec = getattr(scene, "_decor_props", None)
    if dec:
        for pid, until in list(dec.items()):
            if now < until:
                continue
            dec.pop(pid, None)
            _spill._queue(scene, "removed_objects", pid)
    gigs = getattr(scene, "_gigs", None)
    if gigs:
        for gid, st in list(gigs.items()):
            # Legacy records (pre-state-machine) carry only "until"; keep honouring them.
            if "state" not in st:
                if now < st["until"]:
                    continue
            elif st["state"] == "arriving":
                # Waiting for the musician to actually SIT at the instrument. Positions are live
                # (update_state mirrors Unity into scene.agents[...].pos every 1.5s).
                seat = _performer_seat(scene)
                mus = scene.agents.get(st.get("musician"))
                mp = getattr(mus, "pos", None) if mus is not None else None
                seated = (seat is not None and mp is not None
                          and (mp[0] - seat.pos[0]) ** 2 + (mp[1] - seat.pos[1]) ** 2 <= 0.9 ** 2)
                if seated:
                    zid = st["zone"]
                    z = scene.zones.get(zid)
                    gcx, gcy = z.center if z is not None else (0.0, 0.0)
                    for i, off in enumerate((-1.6, 1.6)):          # amps flanking the stage
                        pid = f"{gid}_amp_{i}"
                        _spill._queue(scene, "spawned_objects",
                                      {"id": pid, "object_type": "amp", "zone_id": zid,
                                       "x": float(gcx + off), "y": float(gcy + 0.8)})
                        st["props"].append(pid)
                    nid = f"{gid}_notes"                            # notes float over the instrument
                    _spill._queue(scene, "spawned_objects",
                                  {"id": nid, "object_type": "music_notes", "zone_id": zid,
                                   "x": float(seat.pos[0] + 1.0), "y": float(seat.pos[1] + 2.0)})
                    st["props"].append(nid)
                    st["state"] = "playing"
                    st["until"] = now + float(st.get("play_s", 10.0))
                    _olog(scene, "the musician sits down and starts to play")
                    continue
                if now < st.get("deadline", 0.0):
                    continue                                        # keep waiting for the walk-in
                # never reached the seat: give up quietly (no decor was ever shown)
            elif st["state"] == "playing":
                seat = _performer_seat(scene)
                mus = scene.agents.get(st.get("musician"))
                mp = getattr(mus, "pos", None) if mus is not None else None
                stood_up = (seat is not None and mp is not None
                            and (mp[0] - seat.pos[0]) ** 2 + (mp[1] - seat.pos[1]) ** 2 > 1.3 ** 2)
                if now < st["until"] and not stood_up:
                    continue                                        # set still going
                if stood_up:
                    _olog(scene, "the musician stands up — the set is over")
            gigs.pop(gid, None)
            for pid in st.get("props", []):
                _spill._queue(scene, "removed_objects", pid)
            from dsag.patch import ScenePatch, PatchOp
            scene.active_patches.append(ScenePatch(event_type="group_directive",
                                                   display_name="gig over", ttl=25,
                                                   ops=[PatchOp(op="agent_leave", role="musician")]))
            _olog(scene, "the show is over — the musician packs up")
    vips = getattr(scene, "_vips", None)
    if vips:
        for vid, st in list(vips.items()):
            if now < st["until"]:
                continue
            vips.pop(vid, None)
            from dsag.patch import ScenePatch, PatchOp
            scene.active_patches.append(ScenePatch(event_type="group_directive",
                                                   display_name="vip departs", ttl=25,
                                                   ops=[PatchOp(op="agent_leave", role="vip")]))
            _olog(scene, "the celebrity slipped out the front door")
    outs = getattr(scene, "_outages", None)
    if outs:
        for spid, st in list(outs.items()):
            if now < st["until"]:
                continue
            outs.pop(spid, None)
            _spill._queue(scene, "removed_objects", spid)   # the disable op dies with its patch ttl
            _olog(scene, "power is back — the machine hums to life")


def _check_cleanups(scene):
    """Due cleanups: drop the closure patch and reopen the zone (staff finished cleaning)."""
    import time as _t
    cs = getattr(scene, "_cleanups", None)
    if not cs:
        return
    for zid in [z for z, due in cs.items() if _t.time() >= due]:
        cs.pop(zid, None)
        scene.active_patches = [p for p in getattr(scene, "active_patches", [])
                                if getattr(p, "display_name", "") != f"cleanup:{zid}"]
        _olog(scene, "staff finished cleaning — open again")


def _refill_consumables(scene):
    """A consumed item ('empty') cycles back to full after a few ticks — abstracts staff clearing/restocking.
    Reversible STATE, not deletion, so the same drink station serves the next agent (fixes the old
    despawn-on-first-use that starved later agents). Streams to Unity via object_states."""
    if not _OBJECT_LIFECYCLE:
        return
    for o in scene.objects.values():
        r = getattr(o, "_refill_in", 0)
        if r > 0 and not getattr(o, "removed", False):
            o._refill_in = r - 1
            if o._refill_in == 0:
                o.state = o.states[0] if getattr(o, "states", None) else "default"
                o.state_ticks = 0
                _olog(scene, f"staff restocked the {_oname(o)}")


def _apply_live_effect(scene, aid, chosen):
    """Advance the server-side sim by APPLYING the chosen action's outcome (mirrors dsag's per-tick
    `_apply`): move the agent to its target zone, and on an object interaction apply that affordance's
    need_effects so the need is satisfied and the agent moves on next tick (fixes perpetual same-target
    chasing). Unity renders the walk; the server sim owns the needs. AUTOMATIC OBJECT LIFECYCLE: a single-use
    consumable (glass/plate/cup) is cleared from the scene after use (queued for Unity despawn)."""
    agent = scene.agents.get(aid)
    if agent is None:
        return
    tt, tid, action = chosen["target_type"], chosen["target_id"], chosen["action"]
    _log = _TRACE and aid == _FOCAL
    _before = {n: round(getattr(agent.needs, n, 0.0), 1) for n in V.NEEDS_V1} if _log else None
    # TALKING TO SOMEONE RELIEVES LONELINESS. This branch did not exist: need_effects were applied only
    # for target_type == "object", so a `meet` (a user "X talk to Y" directive, or the spontaneous
    # friends-chat rule) fell through every branch and applied NOTHING. The consequence was a dead bar —
    # loneliness drifts up +0.8/tick for every non-staff agent and the ONLY object in the live catalog
    # that lowers it is talk_to_staff, so the Social bar drained to empty and pinned there while two
    # agents chatted all day. Relief lands on BOTH participants: a conversation is not one-sided.
    if tt == "agent":
        for _p in (agent, scene.agents.get(tid)):
            if _p is not None and hasattr(_p, "needs"):
                _p.needs.loneliness = max(0.0, _p.needs.loneliness - _TALK_RELIEF)
                if hasattr(_p.needs, "groupAffinity"):
                    _p.needs.groupAffinity = max(0.0, _p.needs.groupAffinity - _TALK_RELIEF * 0.5)
        agent.current_zone = getattr(scene.agents.get(tid), "current_zone", agent.current_zone)
        return
    if tt == "object" and tid in scene.objects:
        obj = scene.objects[tid]
        agent.current_zone = obj.zone_id
        aff = next((a for a in obj.affordances if a.action == action), None)
        if aff:
            for n, dv in aff.need_effects.items():
                if hasattr(agent.needs, n):
                    setattr(agent.needs, n, max(0.0, min(100.0, getattr(agent.needs, n) + dv)))
        # STOCK: a served source holds N visible portions (Unity draws them on the counter). Each serving takes
        # one; at zero the ZONE closes for restocking ('be back soon'), staff restocks after ~10s, it reopens
        # with the portions back. Distinct from the single-glass consumable flip below.
        # DRINK counts as well as EAT: this was `action == "eat"` only, so a stocked WATER source never
        # depleted — agents drank from it forever and the bound water sprites never disappeared.
        if (_OBJECT_LIFECYCLE and action in ("eat", "drink")
                and getattr(obj, "stock", None) is not None and not obj.removed):
            had = obj.stock
            obj.stock = max(0, obj.stock - 1)
            who = getattr(agent, "name", aid)
            if obj.stock > 0:
                _olog(scene, f"{who} took some {_oname(obj)} — {obj.stock} left")
            elif had > 0:                      # announce running out ONCE, not on every later attempt
                if "empty" not in obj.states:
                    obj.states.append("empty")
                obj.state = "empty"; obj.state_ticks = 0
                _olog(scene, f"the {_oname(obj)} ran out — staff will restock soon")
                _start_restock(scene, obj)
        # a glass/plate is spent on drink/eat → flip to a reversible 'empty' state (NOT deletion), then it
        # restocks after a few ticks so the next agent can still use it. Streams via object_states.
        elif _OBJECT_LIFECYCLE and action in ("drink", "eat") and _is_consumable(obj) and not obj.removed:
            if "empty" not in obj.states:
                obj.states.append("empty")
            obj.state = "empty"; obj.state_ticks = 0
            obj._refill_in = _REFILL_TICKS
            who = getattr(agent, "name", aid); what = _oname(obj)
            _olog(scene, f"{who} finished the {what} — it's empty now" if action == "drink"
                  else f"{who} took the last {what} — running low")
            if _TRACE:
                log.info(f"[ecgp/lifecycle] {aid} consumed {tid} ({obj.object_type}) -> 'empty' "
                         f"(restock in {_REFILL_TICKS} ticks)")
        # MESS: a used TOILET occasionally gets dirty, and a finished CUP occasionally gets dropped -> the zone
        # closes for a quick clean (Unity hangs the 'closed' sign), staff finish ~10s later and it reopens.
        # Low probability so it's an occasional flavour event, not constant closures.
        if _OBJECT_LIFECYCLE and _MESS_PROB > 0 and not obj.removed:
            if action == "relieve" and random.random() < _MESS_PROB:
                _start_cleanup(scene, obj.zone_id, "toilet")
            elif action == "drink" and _is_consumable(obj) and random.random() < _MESS_PROB * 0.6:
                _start_cleanup(scene, obj.zone_id, _oname(obj))
    elif tt == "zone" and tid in scene.zones:
        agent.current_zone = tid
    elif action == "rest" and hasattr(agent.needs, "energy"):
        agent.needs.energy = min(100.0, agent.needs.energy + 6)   # resting in place restores some energy
    if _log:                                              # prove the live loop applies the effect on completion
        changed = {n: (_before[n], round(getattr(agent.needs, n, 0.0), 1)) for n in _before
                   if abs(_before[n] - getattr(agent.needs, n, 0.0)) > 0.01}
        log.info(f"[ecgp/effect] {aid} COMPLETED {action}@{tt}:{tid}  need change: {changed or 'NONE (no effect!)'}")


_SLOT_RADIUS = float(os.environ.get("ECGP_SLOT_RADIUS", "0.62"))   # how far around a prop agents stand
_SLOTS = int(os.environ.get("ECGP_SLOTS", "8"))                    # distinct standing positions per object


def _slot_target(oid, aid, pos):
    """A distinct standing SLOT around an object's centre, stable per (object, agent) — so agents cluster
    AROUND a table/counter/machine (and beside it, not on top of it) instead of stacking on the exact centre
    and playing the sit/drink clip in empty space. Deterministic + stateless; ORCA nudges any rare overlap."""
    slot = hash(f"{oid}|{aid}") % _SLOTS
    ang = 2.0 * math.pi * slot / _SLOTS + (hash(oid) % 360) * math.pi / 180.0
    r = _SLOT_RADIUS * (0.8 + (hash(f"{aid}|{oid}") % 40) / 100.0)     # slight radius jitter
    return pos[0] + r * math.cos(ang), pos[1] + r * math.sin(ang)


def _option_to_action(w, scene, agent_id, chosen):
    tt, tid, act = chosen["target_type"], chosen["target_id"], chosen["action"]
    reason = chosen.get("reason_override") or f"ecgp:{act}"   # chains relabel the acquire step ("ecgp:order")
    if tt == "object":
        o = w.objects.get(tid)
        if o is not None:
            # ECGP predicts the SEMANTIC action (act); Unity may render the catalog VARIANT (pay/buy/wash)
            # as an execution sub-action — carried as exec_action, never claimed as a model prediction.
            aff = next((a for a in getattr(o, "affordances", []) if a.action == act), None)
            variant = getattr(aff, "variant", None) if aff else None
            # TARGETING by interaction style:
            #   MOUNT (sit/rest/relieve)   -> stand ON the prop itself (prop centre): sit ON the chair/toilet
            #   NEAR  (eat/drink/work)     -> the interaction point right in FRONT of the prop (tiny spread)
            #   else  (observe/talk/…)     -> the loose ring slot around it
            mount = act in ("sit", "rest", "relieve")
            near = act in ("eat", "drink", "work")
            if mount:
                # THE EXPORTED INTERACTION POINT, not the raw prop centre. Unity lands mounted agents on the
                # AUTHORED seat (TryAuthoredMount overrides whatever we send), so measuring arrival against
                # prop_pos meant the server was watching a point up to ~0.8u away from where the agent
                # actually stood: it never saw "arrived", the dwell never started, the need was never
                # relieved, and 40s later the commitment died as route_unreachable. so_sink was the worst
                # case (server 4.35 vs authored 3.55). o.pos IS the exported stand point for these objects.
                sx, sy = o.pos
                sx, sy = _clamp_zone(scene, o.zone_id, sx, sy, margin=0.3)   # a wall-hugging prop stays legal
            elif near:
                sx, sy = o.pos                                     # grounded pos IS the stand-in-front point
                sx += ((hash(f"{o.id}|{agent_id}") % 100) / 100.0 - 0.5) * 0.3
                sx, sy = _clamp_zone(scene, o.zone_id, sx, sy, margin=0.55)  # never inside the wall band
            else:
                sx, sy = _slot_target(o.id, agent_id, o.pos)
                sx, sy = _clamp_zone(scene, o.zone_id, sx, sy)     # ring points kept out of walls
            # FACING: at a child (chair AT table) face the PARENT; a lone mount (toilet, stray seat) faces
            # DOWN so the front-facing sit sprite reads as sitting ON it; else face the prop centre.
            fx, fy = getattr(o, "prop_pos", o.pos)
            par = scene.objects.get(getattr(o, "parent_id", None) or "")
            ov = chosen.get("face_override")
            if par is not None and act in ("sit", "rest", "eat", "drink"):
                fx, fy = getattr(par, "prop_pos", par.pos)
            elif ov:
                fx, fy = ov                                        # chain consume at a lone seat: face the FOOD
            elif mount:
                fx, fy = sx, sy - 1.0
            msg = {"agent_id": agent_id, "action": "move_to_zone", "zone_id": o.zone_id,
                   "smart_object_id": o.id, "target_x": sx, "target_y": sy,
                   "face_x": fx, "face_y": fy, "reason": reason}   # face on arrival
            if variant:
                msg["exec_action"] = variant
            return msg
    if act == "leave":
        ox, oy = dsag_bridge._outside_point(scene, tid)
        return {"agent_id": agent_id, "action": "move_to_zone", "zone_id": tid,
                "target_x": ox, "target_y": oy, "evacuate": True, "reason": reason}
    if tt == "agent":
        return {"agent_id": agent_id, "action": "meet", "target_agent_id": tid, "reason": reason}
    if tt == "zone":
        z = w.zones.get(tid)
        c = z.center if z else (0.0, 0.0)
        return {"agent_id": agent_id, "action": "move_to_zone", "zone_id": tid,
                "target_x": c[0], "target_y": c[1], "reason": reason}
    # self: rest / idle / continue
    cur = w.agents[agent_id].zone if agent_id in w.agents else ""
    return {"agent_id": agent_id, "action": "rest" if act == "rest" else "idle",
            "zone_id": cur, "reason": reason}


def _trace(world, decisions):
    """Per-tick ECGP decision diagnostics (ECGP_TRACE=1): compares each choice to the graph state and
    flags the exact failure modes to watch for — thirsty-ignores-drink, missing/unreachable target,
    uniform behaviour, and target-thrashing. Logs a summary + a sample of flagged agents; appends a
    JSONL row per agent to outputs/ecgp_trace.jsonl for post-hoc analysis."""
    from collections import Counter
    active = bool(world.overlays.active_patch_ids())
    # Focal-agent probe: does an active EventPatch reach the graph (zone attraction overlays) and shift the
    # focal agent's option distribution? Prints the top-5 joint (action,target) options + probs so you can
    # compare before/after a patch and see whether the ECGP posterior is actually sensitive to the event.
    if _FOCAL in decisions:
        attr = {zid: round(world.overlays.attraction(zid), 1) for zid in world.zones
                if abs(world.overlays.attraction(zid)) > 1e-6}
        fd = decisions[_FOCAL]
        top5 = sorted(fd["options"], key=lambda o: -o["p"])[:5]
        log.info(f"[ecgp/focal] {_FOCAL} patch_active={active} zone_attractions={attr or '{}'}  top5: "
                 + " | ".join(f"{o['action']}:{o['target_type']}/{o['target_id']}={o['p']:.3f}" for o in top5))
    tgt_count = Counter()
    switches = 0
    flagged = []
    rows = []
    for aid, d in decisions.items():
        agent = world.agents.get(aid)
        if agent is None:
            continue
        ch = d["chosen"]
        top = agent.urgent_needs(2)
        key = (ch["target_type"], ch["target_id"])
        tgt_count[key] += 1
        flags = []
        # thirsty (high thirst pressure) but idling/resting instead of pursuing a reachable drink
        # (walking is fine — the agent may be heading toward the water; only inaction is flagged)
        if NP.pressure("thirst", agent.needs.get("thirst", 0)) >= 0.7:
            has_drink = any(o["action"] == "drink" for o in d["options"])
            if has_drink and ch["action"] in ("idle", "rest", "observe"):
                flags.append("thirsty_idle")
        # target must exist in the graph
        if ch["target_type"] == "object" and ch["target_id"] not in world.objects:
            flags.append("missing_object")
        if ch["target_type"] == "zone" and ch["target_id"] not in world.zones:
            flags.append("missing_zone")
        # target-switch (thrashing) vs last tick
        prev = _LAST_TARGET.get(aid)
        if prev is not None and prev != key and ch["action"] not in ("idle", "rest"):
            switches += 1
            flags.append("switched")
        _LAST_TARGET[aid] = key
        if flags:
            flagged.append((aid, flags, ch, round(top and NP.pressure(top[0], agent.needs.get(top[0], 0)) or 0, 2)))
        rows.append({"agent": aid, "top_needs": top, "chosen": ch, "prob": round(d["prob"], 3),
                     "flags": flags, "active_event": active})
    n = max(1, len(decisions))
    top_target, top_n = (tgt_count.most_common(1)[0] if tgt_count else (("", ""), 0))
    uniform = top_n / n
    log.info(f"[ecgp] tick decisions={n}  most-common-target={top_target}({top_n}/{n}={uniform:.0%})  "
             f"switches={switches}  flagged={len(flagged)}"
             + (f"  event=ACTIVE" if active else ""))
    for aid, flags, ch, p in flagged[:6]:
        log.info(f"[ecgp/FLAG] {aid} {flags} chose {ch['target_type']}:{ch['target_id']}/{ch['action']} (need_p={p})")
    try:
        os.makedirs(os.path.dirname(_TRACE_PATH) or ".", exist_ok=True)
        with open(_TRACE_PATH, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    except Exception:
        pass


# ── HONEST ACTION PROVENANCE (TWO fields, not one) ──────────────────────────────────────────────────────
# `semantic_origin` — who chose the GOAL (the action+target that MEANS something: "eat", "relieve",
# "go home"). This is the field a paper claim about learned behaviour must cite.
# `execution_origin` — who chose THIS TICK'S concrete physical subtarget while carrying that goal out
# (which queue slot, which chain stage's object). For a plain one-step action these are identical (the
# semantic target IS the execution target); they DIVERGE only inside a multi-stage chain (order->seat),
# where the goal ("eat@cluster_table_1") stays learned_ecgp for every stage while each concrete stage
# object (the provider, then the reserved chair) is deterministic_execution. The executor may pick WHICH
# sub-object satisfies an already-frozen semantic goal; it may never invent a NEW semantic goal — that
# invariant is what audit A verifies. Both fields are independent of the pre-existing `source`/`reason`/
# `priority` wire fields (unchanged — Unity's IntentCommitment sequencing depends on them).
#   learned_ecgp            — the goal IS (or directly continues) an ECGP joint-option selection, possibly
#                              narrowed by a hard feasibility mask but never replaced by an unpicked target.
#   explicit_directive       — a literal single-agent/group user command (role dispatch, "Alex sit down",
#                              directed leave, non-emergency global leave / closing time).
#   hard_safety              — the emergency evacuation override.
#   deterministic_execution  — scheduling/capacity/feasibility mechanics with no behavioral content of
#                              their own (stagger idle, queue-wait, unreachable-target give-up, spill
#                              cleanup dispatch, chain sub-target grounding) — necessary plumbing, not a
#                              claim about crowd response.
#   hybrid_fallback          — a deterministic ambient-event response standing in for what should ideally
#                              be an ECGP decision (party/gather zone routing; the queue-frustration give-up
#                              rule). Only reachable in safe_demo_hybrid mode for party/gather; the
#                              frustration rule runs in both modes as a stall safety valve, and is the one
#                              case that does not fit cleanly into the five categories.
_ORIGIN_ALL = ("learned_ecgp", "explicit_directive", "hard_safety", "deterministic_execution", "hybrid_fallback")


def _log_chain_stage(aid, semantic_goal, semantic_target, semantic_origin, chain_id, chain_stage, execution_target,
                     chain_state=None):
    """Section 1's required per-stage log line — every chain stage prints its full provenance record so a
    trace can distinguish 'ECGP chose the goal once' from 'the executor chose this stage's sub-target'.
    When a `chain_state` (ecgp.graph.chain_state.ChainState) is available, its named stage and
    failure_reason are logged alongside the integer chain_stage (a step index), so a trace shows the real
    stage name such as CONSUMING rather than an opaque 0/1."""
    extra = ""
    if chain_state is not None:
        extra = f" stage={chain_state.stage.value} failure_reason={chain_state.failure_reason}"
    log.info(f"[ecgp/chain] agent={aid} semantic_goal={semantic_goal} semantic_target={semantic_target} "
             f"semantic_origin={semantic_origin} chain_id={chain_id} chain_stage={chain_stage}{extra} "
             f"execution_target={execution_target} execution_origin=deterministic_execution")


def _log_policy_reinvoke(aid, reason, com):
    """Commitment-manager contract (section 6): every time a commitment is invalidated and the graph
    policy will be re-invoked, log WHY, what the previous commitment was, and what was released — so a
    trace can audit that the policy is only ever consulted at real semantic decision points (target
    failure / reservation loss / route invalidation / completion / emergency / clear), never as a
    per-tick re-choice while a valid commitment holds."""
    cur = _com_cur(com) if com else {}
    log.info(f"[ecgp/reinvoke] agent={aid} reason={reason} "
             f"prev_commitment={cur.get('action')}@{cur.get('target_id')} "
             f"chain_id={com.get('chain_id') if com else None} "
             f"released_seat={com.get('seat') if com else None} "
             f"released_token={bool(com.get('token')) if com else False}")


def _finalize_provenance(a):
    """Fill semantic_goal/semantic_target/execution_target/execution_origin/chain_id/chain_stage for any
    action that didn't set them explicitly (i.e. every NON-chain action) — for those, the execution target
    IS the semantic target and execution_origin mirrors semantic_origin (no separate sub-target resolution
    happened). Chain branches (CLUSTER CHAIN, PHASE B) set all of these explicitly and are left untouched.
    Called once per action right before it is appended, so every action in the wire payload — not just the
    ones this file happens to log — carries the full 7-field record."""
    a.setdefault("semantic_origin", None)
    a.setdefault("chain_id", None)
    a.setdefault("chain_stage", 0)
    a.setdefault("stage", None)
    a.setdefault("failure_reason", None)
    tgt = a.get("smart_object_id") or a.get("zone_id")
    a.setdefault("semantic_goal", a.get("reason"))
    a.setdefault("semantic_target", tgt)
    a.setdefault("execution_target", a.get("execution_target", tgt))
    a.setdefault("execution_origin", a.get("semantic_origin"))
    return a


def _log_origin_breakdown(scene, actions):
    """Section-2 compliance: count + log every final action's semantic_origin, every tick, no exceptions
    (the paper-facing claim always cites THIS field, never execution_origin). An action missing a tag is a
    BUG in this file (a branch that forgot to tag itself) — logged loudly as UNTAGGED rather than silently
    defaulting to something that could pass as legitimate. Every action missing chain fields is filled in
    via _finalize_provenance first; chain actions log their full 7-field record separately (see callers)."""
    for a in actions:
        _finalize_provenance(a)
    counts = {o: 0 for o in _ORIGIN_ALL}
    untagged = 0
    for a in actions:
        o = a.get("semantic_origin")
        if o in counts:
            counts[o] += 1
        else:
            untagged += 1
    n = max(1, len(actions))
    parts = [f"{o}={counts[o]}({counts[o] / n:.0%})" for o in _ORIGIN_ALL]
    if untagged:
        log.warning(f"[ecgp/origin] {untagged} action(s) missing a semantic_origin tag this tick — "
                    f"a branch in ecgp_tick forgot to set one")
        parts.append(f"UNTAGGED={untagged}")
    log.info(f"[ecgp/origin] mode={_MODE} tick={getattr(scene, 'tick_no', 0)} n={len(actions)}  "
             + "  ".join(parts))


def ecgp_tick(scene, zone_dicts):
    """One ECGP-driven tick → (actions, object_states, events) in the server wire format.

    Two regimes (matching the demo's hard rules): (1) EMERGENCY → deterministic full evacuation, EVERY
    agent walks to the exit and off the floor plan (never trusts the model to empty the building); (2)
    NORMAL → ECGP drives per-agent behavior, but `leave` is SUPPRESSED (agents never leave unless an
    emergency or an explicit go-home directive, which the server applies as an overlay on top)."""
    scene.tick_no = getattr(scene, "tick_no", 0) + 1
    # age out EventPatches (scene.tick() does this for the rule engine; ECGP skips scene.tick()) so
    # "toilet closed", attractions and cleared emergencies actually expire instead of lasting forever.
    for p in getattr(scene, "active_patches", []):
        p.ttl -= 1
    scene.active_patches = [p for p in getattr(scene, "active_patches", []) if p.ttl > 0]
    # a removed object whose owning patch just expired comes BACK into service (reversible remove_object)
    try:
        from dsag import patch as _dp
        restored = _dp.restore_expired_removals(scene)
        if restored:
            log.info(f"[ecgp/lifecycle] restored {restored} (owning patch cleared) -> available again")
    except Exception:
        pass
    _refill_consumables(scene)                           # spent items cycle back to full (abstracts staff restock)
    _check_restocks(scene)                               # due 'be back soon' zones reopen with food refilled
    _check_event_props(scene)                            # fire auto-clear + pop-up cart expiry
    _check_cleanups(scene)                               # due 'closed' (mess) zones reopen once staff finish
    # (repairs are advanced with the spill overrides above — they need to emit an action)
    try:
        scene._update_needs()                            # drift needs (Unity renders; sim owns needs)
    except Exception:
        pass
    # EXPLICIT directive (spawned responders -> their assigned zone) — always active in BOTH modes; this
    # is a named role command (section 6), not ambient hybrid event routing, so it does NOT gate on
    # _HYBRID_DIRECTIVE/_RESEARCH_MODE.
    role_targets = _role_targets(scene)
    global_leave = getattr(scene, "is_global_leave", lambda: False)()
    if not global_leave:
        scene._evac_cleared = False                      # arm the one-shot evac cleanup for the next emergency
    actions = []
    if global_leave:                                     # evacuation (emergency) OR end_of_day (calm)
        emergency = getattr(scene, "is_emergency", lambda: False)()
        exit_id = scene.exit_zone_id()
        ox, oy = dsag_bridge._outside_point(scene, exit_id)
        reason = "ecgp:evacuate" if emergency else "ecgp:end_of_day"
        gl_source = "emergency" if emergency else "directive"
        gl_priority = 4 if emergency else 3
        # STEP 4: on emergency activation, safely cancel every in-progress interaction chain and RELEASE all
        # held smart-object reservations/occupancy — nobody is mid-order/seated once the alarm sounds, and a
        # left-behind reservation would keep an object "in use". Idempotent (runs each evac tick harmlessly).
        if not getattr(scene, "_evac_cleared", False):
            for a2 in list(_COMMIT):
                com2 = _COMMIT[a2]
                cs2 = com2.get("chain_state")
                if cs2 is not None and not cs2.is_terminal():
                    cs2.advance(ChainStage.CANCELLED, scene.tick_no, failure_reason="emergency")
                _chain_cleanup(scene, a2, com2)
            _COMMIT.clear()
            for o in scene.objects.values():
                try:
                    o.reservations.clear(); o.occupied_by = None
                except Exception:
                    pass
            scene._evac_cleared = True
            log.info("[ecgp] emergency: chains cancelled, all smart-object reservations released")
        gl_origin = "hard_safety" if emergency else "explicit_directive"
        for aid, agent in scene.agents.items():
            rz = role_targets.get((getattr(agent, "role", "") or "").lower())
            if rz and rz in scene.zones:                 # RESPONDER: go INTO the directed zone, not out
                cz = scene.zones[rz].center
                actions.append({"agent_id": aid, "action": "move_to_zone", "zone_id": rz,
                                "target_x": float(cz[0]), "target_y": float(cz[1]), "reason": "hybrid:respond",
                                "source": "directive", "priority": 3, "semantic_origin": "explicit_directive"})
                agent.current_zone = rz
            else:                                        # everyone else evacuates
                actions.append({"agent_id": aid, "action": "move_to_zone", "zone_id": exit_id,
                                "target_x": ox, "target_y": oy, "evacuate": True, "reason": reason,
                                "source": gl_source, "priority": gl_priority, "semantic_origin": gl_origin})
                agent.current_zone = exit_id
        _log_origin_breakdown(scene, actions)
        return actions, [o.render_state() for o in scene.objects.values()], _drain_obj_log(scene)

    world = build_world(scene, zone_dicts)
    decisions = get_policy().act(world)
    if _BEHAVIOR_PRIOR:                                   # trait-conditioned prior LAYER (β=0 -> no-op)
        apply_behavior_prior(decisions, scene)
    if _TRACE:
        _trace(world, decisions)

    # USER-DIRECTIVE redirection (positive attraction) — executed DETERMINISTICALLY, exactly as emergency
    # evacuation is: the learned policy is only weakly event-conditioned (a zone_attraction overlay barely
    # moves the crowd), so a "gather in the dining area / food offering here" directive reliably PULLS the
    # crowd to the attracted zone. Agents with an urgent Tier-1 need satisfy it first (then rejoin next tick).
    attracted = _attracted_zones(scene) if _HYBRID_DIRECTIVE else []
    if attracted:
        log.info(f"[ecgp/HYBRID] deterministic directive routing to {[z.id for z in attracted]} "
                 f"(NOT learned — ECGP_HYBRID_DIRECTIVE=1)")
    if scene.tick_no <= 1:
        _COMMIT.clear(); _LEAVING.clear(); _FRUSTRATION.clear(); _PARTY_SLOT.clear()  # fresh scene → drop state
        _NEED_GRACE.clear()
    _to_remove = []                                      # agents that finished walking out this tick (R5)
    scene._zone_rects = {z["id"]: (z.get("x", 0.0), z.get("y", 0.0), z.get("w", 2.0), z.get("h", 2.0))
                         for z in zone_dicts}            # for clamping stand/queue points inside walls
    _build_zone_allowed(scene, zone_dicts)               # per-zone action gate (Claude-designed / fn defaults)

    # PARTY/GATHER SLOTS: multiple distinct gathering points tiling each attracted zone's interior (capacity =
    # slot count). Agents claim ONE stable slot each (_PARTY_SLOT) so the crowd DISTRIBUTES across the zone
    # instead of stacking on one centre; agents past capacity aren't pulled in. Rebuilt each tick from the rects.
    party_slots, party_taken = {}, {}
    if attracted:
        for z in attracted:
            party_slots[z.id] = _zone_slots(scene._zone_rects.get(z.id))
            party_taken[z.id] = set()
        for _aid, (_zid, _idx) in list(_PARTY_SLOT.items()):          # re-reserve still-valid cached slots
            if _zid in party_slots and _idx < len(party_slots[_zid]):
                party_taken[_zid].add(_idx)
            else:
                _PARTY_SLOT.pop(_aid, None)
    else:
        _PARTY_SLOT.clear()

    # STAGE 2 spill cascade — dispatch staff to clean any spill BEFORE the main loop, so an assigned cleaner
    # is EXCLUDED from re-deciding below (otherwise the GNN kept moving them off the spill and cleaning never
    # completed — the repeating GUARD:spill warning). tick_spills does reservation + navigation + completion.
    _spill.maybe_autospawn(scene)
    spill_overrides = _spill.tick_spills(scene)
    spill_overrides.update(tick_repairs(scene))   # broken objects: staff walk over + fix

    # DIRECTED LEAVE (user "X goes home" / "the family leaves"): resolved ONCE up front so it participates in the
    # decision PRECEDENCE below (top priority, step 0.0) rather than as a post-hoc patch — an agent_leave then
    # beats every normal ECGP choice before anything is serialised.
    from dsag import patch as _dp_leave
    directed_leavers = set(_dp_leave.agents_leaving(getattr(scene, "active_patches", []) or [], scene))
    if directed_leavers:
        log.info(f"[ecgp] directed leave (override): {sorted(directed_leavers)}")

    # PERSONAL ACTION DIRECTIVE: a named person told to do a specific thing now ("Alex wants to sit down"
    # -> {agent:'Alex', action:'sit'}), resolved to a real reachable object below (step 0.6).
    personal_actions = _dp_leave.personal_action_directives(getattr(scene, "active_patches", []) or [], scene)

    for aid, d in decisions.items():
        agent = scene.agents.get(aid)
        # SPILL cleaner: locked to the clean task this tick (do not let the model redirect them).
        if aid in spill_overrides:
            act = spill_overrides[aid]
            act.setdefault("semantic_origin", "deterministic_execution")   # facilities mechanic, not a crowd decision
            actions.append(act)
            _drop_commit(scene, aid)
            continue
        # 0.0) DIRECTED LEAVE — TOP precedence. A named agent / group told to go home walks to the exit and OFF
        #      the map (evacuate=True) and is LOCKED into leaving (via _LEAVING) so it never re-decides mid-exit.
        #      This is what makes "the family leaves" reliable regardless of the model's per-agent output; if the
        #      directive later expires, the _LEAVING lock (step 0.4) still carries the agent out.
        if agent is not None and aid in directed_leavers:
            _drop_commit(scene, aid)
            _LEAVING[aid] = _LEAVING.get(aid, 0) + 1
            act = dsag_bridge.build_leave_action(scene, aid, "ecgp:leave_home")   # shared builder (item 4)
            act["source"] = "directive"; act["priority"] = 3; act["semantic_origin"] = "explicit_directive"
            actions.append(act)
            agent.current_zone = act["zone_id"]
            _maybe_retire(aid, _to_remove)
            continue
        # 0) ROLE directive (spawned responders → their zone) — deterministic, overrides model + attraction.
        rz = role_targets.get((getattr(agent, "role", "") or "").lower()) if agent is not None else None
        if rz and rz in world.zones:
            # A PERFORMER sent to the stage does not stand in the middle of it — they take the
            # instrument's stool. This branch is TOP precedence and `continue`s, so without this the
            # gig's own `role_directive musician -> music_room` pinned the musician to the ZONE CENTRE
            # and the performer-seat rule at 2.55 was never reached: the visible result was a musician
            # standing in an empty room while the piano stool sat empty beside him. (The reason string
            # must contain "sit" — it selects the animation clip; see ClipForReason.)
            if (getattr(agent, "role", "") or "").lower() in _PERFORMER_ROLES:
                _pseat = _performer_seat(scene)
                if _pseat is not None and getattr(_pseat, "zone_id", None) == rz:
                    _drop_commit(scene, aid)
                    _pact = _option_to_action(world, scene, aid,
                                              {"target_type": "object", "target_id": _pseat.id,
                                               "action": "sit", "p": 1.0,
                                               "reason_override": "hybrid:performer_sit"})
                    _pact["source"] = "directive"; _pact["priority"] = 3
                    _pact["semantic_origin"] = "explicit_directive"
                    actions.append(_pact)
                    agent.current_zone = rz
                    continue
            _drop_commit(scene, aid)
            c = world.zones[rz].center
            actions.append({"agent_id": aid, "action": "move_to_zone", "zone_id": rz,
                            "target_x": c[0], "target_y": c[1], "reason": "hybrid:respond",
                            "source": "directive", "priority": 3, "semantic_origin": "explicit_directive"})
            agent.current_zone = rz
            continue
        # 0.4) ALREADY LEAVING (R5): an agent that decided to go home is LOCKED to walking out — it keeps
        #      heading for the exit every tick (no re-deciding, no re-logging) and is retired from the sim
        #      once UNITY CONFIRMS it actually crossed the exit portal (see _maybe_retire) — not on a blind
        #      timer alone (a jammed doorway used to let the timer fire while the agent was still stuck,
        #      orphaning a visible, un-driven "ghost" in Unity). The timer is now only a generous safety net.
        if aid in _LEAVING:
            _LEAVING[aid] += 1
            act = dsag_bridge.build_leave_action(scene, aid, "ecgp:gave_up")      # shared builder (item 4)
            act["source"] = "directive"; act["priority"] = 3; act["semantic_origin"] = "explicit_directive"
            actions.append(act)
            if agent is not None:
                agent.current_zone = act["zone_id"]
            _maybe_retire(aid, _to_remove)
            continue
        # 0.5) FRUSTRATION (R5): an agent that has been stuck WAITING IN LINE for too long (counted post-loop,
        #      below) gives up and goes home exactly ONCE, then the leaving lock above takes over. Tied to
        #      queue-wait (not raw needs) so only genuinely-stuck agents leave — a busy-but-served agent never
        #      does — which is what 'waited too long' actually means. Rare by design.
        if agent is not None and _FRUSTRATION.get(aid, 0) >= _FRUST_LIMIT:
            _FRUSTRATION.pop(aid, None); _drop_commit(scene, aid)
            _LEAVING[aid] = 0
            _olog(scene, f"{getattr(agent, 'name', aid)} got fed up waiting and went home")
            act = dsag_bridge.build_leave_action(scene, aid, "ecgp:gave_up")      # shared builder (item 4)
            # NOT an explicit user directive (nobody told this agent to leave) and NOT a learned choice
            # either — a deterministic queue-starvation safety valve. Labelled hybrid_fallback for honesty;
            # this is the one judgment call in the taxonomy that doesn't map cleanly onto the 5 categories
            # as literally specified — flagged here rather than silently picked.
            act["source"] = "directive"; act["priority"] = 3; act["semantic_origin"] = "hybrid_fallback"
            actions.append(act)
            agent.current_zone = act["zone_id"]
            continue
        # 0.6) PERSONAL ACTION DIRECTIVE — a named person told to do something specific now
        #      ('Alex wants to sit down'). Outranks the party/attraction pull (an explicit directive beats an
        #      ambient event) but sits below leave/role. ONLY fires on first acquisition (no existing matching
        #      commitment): once _COMMIT[aid] targets the resolved object, this step steps ASIDE (no continue)
        #      so the normal arrival-gated dwell/commit-honoring logic further down drives it — re-setting the
        #      commit every tick here would reset its dwell countdown and the agent would never "arrive".
        if agent is not None and aid in personal_actions and aid not in directed_leavers:
            action_name = personal_actions[aid]
            existing = _COMMIT.get(aid)
            already_bound = (existing is not None and "steps" not in existing
                             and existing.get("target_type") == "object"
                             and existing.get("target_id") == _DIRECTED_ACTION_OBJ.get(aid)
                             and existing.get("action") == action_name)
            if not already_bound:
                cur_oid = _DIRECTED_ACTION_OBJ.get(aid)
                obj = scene.objects.get(cur_oid) if cur_oid else None
                if obj is None or getattr(obj, "removed", False):
                    obj = _find_free_object_for_action(scene, action_name, getattr(agent, "current_zone", None))
                if obj is not None:
                    _DIRECTED_ACTION_OBJ[aid] = obj.id
                    opt = {"target_type": "object", "target_id": obj.id, "action": action_name, "p": 1.0}
                    act = _option_to_action(world, scene, aid, opt)
                    act["source"] = "directive"; act["priority"] = 3; act["semantic_origin"] = "explicit_directive"
                    actions.append(act)
                    _COMMIT[aid] = {"target_type": "object", "target_id": obj.id, "action": action_name,
                                    "ttl": _dwell(action_name), "source": "directive", "priority": 3,
                                    "semantic_origin": "explicit_directive"}
                    continue
                log.warning(f"[ecgp] personal directive '{action_name}' for {getattr(agent,'name',aid)}: "
                            f"no reachable/available {action_name} object found this tick — falling through "
                            f"to a normal decision")
            # already bound (or no object available): fall through — commit-honoring further down drives an
            # already-bound directive to completion; an unfound one gets a normal decision this tick.
        # 1) HYBRID user-directive attraction — DETERMINISTIC, explicitly flagged (reason "hybrid:*", not
        #    "ecgp:") so it is never mistaken for the learned policy. A PARTY pulls everyone but the CRITICALLY
        #    urgent (R2) and plays a celebration clip; a normal 'gather' still yields to a Tier-1 need. Agents
        #    spread within the zone so the crowd clusters visibly instead of stacking on the centre.
        if attracted and agent is not None and aid not in personal_actions:   # explicit directive beats an ambient event
            party = any(_is_party_zone(scene, z) for z in attracted)
            # A PARTY invites the crowd, not the workers: staff, security and guides keep doing their
            # jobs while non-staff join in. A plain 'gather' directive ("everyone to the centre") still
            # moves everyone, staff included.
            worker = _social_role(getattr(agent, "role", "")) in ("staff", "security", "guide")
            skip = (_critical_need(agent) if party else _urgent_tier1(agent)) or (party and worker)
            if not skip:
                tz = min(attracted, key=lambda z: _zone_dist(scene, agent.current_zone, z.id))
                slots = party_slots.get(tz.id) or []
                cached = _PARTY_SLOT.get(aid)                          # keep the SAME slot across ticks (no jitter)
                idx = cached[1] if (cached and cached[0] == tz.id and cached[1] < len(slots)) else None
                if idx is None:                                       # else claim the next FREE slot (capacity gate)
                    idx = next((i for i in range(len(slots)) if i not in party_taken[tz.id]), None)
                if idx is not None:
                    _drop_commit(scene, aid)
                    party_taken[tz.id].add(idx); _PARTY_SLOT[aid] = (tz.id, idx)
                    tx, ty = _clamp_zone(scene, tz.id, *slots[idx])
                    actions.append({"agent_id": aid, "action": "move_to_zone", "zone_id": tz.id,
                                    "target_x": tx, "target_y": ty,
                                    "reason": "hybrid:party" if party else "hybrid:gather",
                                    "source": "event", "priority": 2, "semantic_origin": "hybrid_fallback"})
                    agent.current_zone = tz.id
                    continue
                # zone at CAPACITY -> this agent isn't pulled in; it falls through to normal ECGP behaviour
        # 2) HONOR an in-progress commitment: keep heading to the SAME target and dwell there for a few
        #    ticks, instead of re-sampling a new target every tick (the fast/darting/unreal motion). The
        #    need is satisfied on completion (end of the dwell), so agents visibly walk over, do the thing,
        #    then move on — one action at a time.
        com = _COMMIT.get(aid)
        if com and not _target_alive(scene, com):        # current target gone (removed/despawned) — abort clean
            cs_gone = com.get("chain_state")
            if cs_gone is not None and not cs_gone.is_terminal():
                cs_gone.advance(ChainStage.FAILED, scene.tick_no, failure_reason="target_removed")
            _log_policy_reinvoke(aid, "target_removed", com)
            _chain_cleanup(scene, aid, com)
            _COMMIT.pop(aid, None)
            com = None
        # ANY commitment must abort mid-dwell on hazard or disable — not just macro-chains. This was scoped
        # to `"steps" in com` (chains), but the prebaked demo runs on plain single-step commitments, so an
        # agent kept eating at a counter showing "be back soon" for its whole dwell: the restock/cleanup
        # patch disabled the affordance, the OPTIONS honoured it (nobody new chose the counter), but the
        # already-committed diner was never re-checked. Widening it makes the closed sign actually clear
        # the counter, which is both the visible expectation and what the teacher simulator does.
        if com:
            cur_chk = _com_cur(com)
            chk_obj = scene.objects.get(cur_chk["target_id"]) if cur_chk["target_type"] == "object" else None
            disabled_now = chk_obj is not None and (
                world.overlays.is_hazard(chk_obj.zone_id) or
                world.overlays.is_disabled(chk_obj.id, cur_chk["action"]) or
                world.overlays.is_disabled(chk_obj.zone_id, cur_chk["action"]))
            if disabled_now:
                cs_int = com.get("chain_state")
                if cs_int is not None and not cs_int.is_terminal():
                    cs_int.advance(ChainStage.FAILED, scene.tick_no, failure_reason="hazard_or_disabled")
                _log_policy_reinvoke(aid, "hazard_or_disabled", com)
                _chain_cleanup(scene, aid, com)
                _COMMIT.pop(aid, None)
                actions.append({"agent_id": aid, "action": "idle", "zone_id": getattr(agent, "current_zone", ""),
                                "reason": "ecgp:idle", "semantic_origin": "deterministic_execution"})
                continue
        if com and _com_cur(com)["ttl"] > 0:
            cur = _com_cur(com)
            # ARRIVAL-GATED dwell (object targets): the use-timer only runs once the agent is physically AT
            # the object (live Unity position within _USE_RADIUS) — so 'eating' happens at the food, never
            # halfway across the room. While walking, the commitment holds without burning dwell; a target
            # never reached within _ARRIVE_TIMEOUT walking ticks is dropped (sealed-off/unreachable).
            at_target = True                              # no live position (headless/tests) -> legacy tick-dwell
            if cur["target_type"] == "object":
                pos = getattr(agent, "pos", None) if agent is not None else None
                obj = scene.objects.get(cur["target_id"])
                if pos is not None and obj is not None:
                    dx, dy = pos[0] - obj.pos[0], pos[1] - obj.pos[1]
                    at_target = dx * dx + dy * dy <= _USE_RADIUS * _USE_RADIUS
            if at_target:
                cur["ttl"] -= 1
            else:
                com["walk"] = com.get("walk", 0) + 1
                if com["walk"] > _ARRIVE_TIMEOUT:         # can't get there — give up and re-decide next tick
                    cs_stuck = com.get("chain_state")
                    if cs_stuck is not None and not cs_stuck.is_terminal():
                        cs_stuck.advance(ChainStage.FAILED, scene.tick_no, failure_reason="unreachable_target")
                    _log_policy_reinvoke(aid, "route_unreachable", com)
                    _chain_cleanup(scene, aid, com)
                    _COMMIT.pop(aid, None)
                    actions.append({"agent_id": aid, "action": "idle",
                                    "zone_id": getattr(agent, "current_zone", ""), "reason": "ecgp:idle",
                                    "semantic_origin": "deterministic_execution"})   # unreachable-target give-up, not a decision
                    continue
            cont_act = _option_to_action(world, scene, aid, cur)
            # Carry the commitment's origin forward onto every continuation tick. Tagging only the tick
            # that created the commitment leaves every dwell and arrival tick after it untagged
            # (source=None); provenance must cover the final serialized action of every tick.
            cont_act["source"] = com.get("source", "interaction"); cont_act["priority"] = com.get("priority", 1)
            cont_act["semantic_origin"] = com.get("semantic_origin", "learned_ecgp")
            cs = com.get("chain_state")
            if "steps" in com:                          # chain continuation — carry the full 7-field record
                cont_act["semantic_goal"] = com.get("semantic_goal"); cont_act["semantic_target"] = com.get("semantic_target")
                cont_act["chain_id"] = com.get("chain_id"); cont_act["chain_stage"] = com["i"]
                cont_act["execution_target"] = cur["target_id"]; cont_act["execution_origin"] = "deterministic_execution"
                if cs is not None:
                    # Derive the named stage from the same signals already driving execution (step index
                    # `i` plus arrival-gated `at_target`) rather than from reservations: at i==1, arriving
                    # this tick with ttl still at full dwell is SEATED, and a later tick mid-dwell is
                    # CONSUMING. This mirrors the teacher simulator's ACQUIRED/APPROACH_SEAT/SEATED/CONSUMING
                    # cascade but spread across real per-tick arrival gating instead of one collapsed tick.
                    if com["i"] == 0:
                        if cs.stage != ChainStage.APPROACH_PROVIDER:
                            cs.advance(ChainStage.APPROACH_PROVIDER, scene.tick_no)
                    else:
                        full = _dwell(cur["action"])
                        if not at_target and cs.stage not in (ChainStage.APPROACH_SEAT,):
                            cs.advance(ChainStage.APPROACH_SEAT, scene.tick_no)
                        elif at_target and cur["ttl"] == full and cs.stage != ChainStage.SEATED:
                            cs.advance(ChainStage.SEATED, scene.tick_no)
                        elif at_target and cur["ttl"] < full and cs.stage != ChainStage.CONSUMING:
                            cs.advance(ChainStage.CONSUMING, scene.tick_no)
                    cont_act["stage"] = cs.stage.value; cont_act["failure_reason"] = cs.failure_reason
                _log_chain_stage(aid, com.get("semantic_goal"), com.get("semantic_target"),
                                 com.get("semantic_origin"), com.get("chain_id"), com["i"], cur["target_id"], cs)
            actions.append(cont_act)
            if cur["ttl"] == 0:
                if "steps" in com and com["i"] < len(com["steps"]) - 1:
                    # CHAIN: acquired at the source — advance to the seat (claim the reserved slot), carry on.
                    com["i"] += 1
                    com["walk"] = 0
                    seat = scene.objects.get(com.get("seat", ""))
                    if seat is not None and com.get("token"):
                        if not seat.claim(aid, com.pop("token")):
                            # RESERVATION LOST (token expired, or the slot was taken) — a semantic
                            # decision point under the commitment-manager contract: release, mark FAILED,
                            # and hand the choice back to the graph policy next tick, rather than
                            # marching on into a seat the agent no longer holds.
                            if cs is not None and not cs.is_terminal():
                                cs.advance(ChainStage.FAILED, scene.tick_no,
                                           failure_reason="reservation_lost")
                            _log_policy_reinvoke(aid, "reservation_lost", com)
                            _chain_cleanup(scene, aid, com)
                            _COMMIT.pop(aid, None)
                            actions.append({"agent_id": aid, "action": "idle",
                                            "zone_id": getattr(agent, "current_zone", ""),
                                            "reason": "ecgp:idle",
                                            "semantic_origin": "deterministic_execution"})
                            continue
                    if cs is not None:
                        cs.acquired_item_id = com.get("src")
                        cs.advance(ChainStage.ACQUIRED, scene.tick_no)
                else:
                    # Complete: a chain applies the SOURCE's need effects (the food or drink carried to
                    # the seat) AND the seat's own effects (the energy/stress relief of sitting down).
                    # Applying only the source's effects drops the seat's benefit for every chained
                    # action, even though a standalone sit gets it via the `else` branch below — which is
                    # why a complete cluster (eat+sit) would never outscore a bare provider (eat-only).
                    # The agent then logically stays at the seat's zone; a single step applies itself.
                    if "steps" in com:
                        _apply_live_effect(scene, aid, {"target_type": "object",
                                                        "target_id": com.get("src", cur["target_id"]),
                                                        "action": cur["action"]})
                        seat = scene.objects.get(com.get("seat", ""))
                        if seat is not None:
                            seat_action = next((a.action for a in getattr(seat, "affordances", [])
                                               if a.action == "sit"), None)
                            if seat_action:
                                _apply_live_effect(scene, aid, {"target_type": "object", "target_id": seat.id,
                                                                "action": seat_action})
                            seat.release(aid)
                            if agent is not None:
                                agent.current_zone = seat.zone_id
                        if cs is not None:
                            cs.advance(ChainStage.COMPLETED, scene.tick_no)
                    else:
                        _apply_live_effect(scene, aid, cur)   # arrived + dwelled → satisfy the need now
                    _COMMIT.pop(aid, None)
                    _queue_follow_on(scene, aid, cur)         # e.g. wash your hands after the toilet
            continue
        # 2.5) HYBRID NEED-RELIEF SAFETY-NET (safe_demo_hybrid ONLY) — a LAST-RESORT deterministic rescue for a
        #      GENUINELY DESPERATE agent (need past the HIGH _RELIEF_BARS) that the weak V2 GNN failed to serve.
        #      Kept narrow so the LEARNED policy drives most of the crowd rather than being drowned out by
        #      hybrid_fallback: (1) high bars, (2) the model gets a
        #      GRACE window to self-correct first, (3) if the model ALREADY chose a relieving action this tick
        #      it is left alone and CREDITED to learned_ecgp. OFF in research_learned. Placed after commit-honor
        #      (an agent already walking to relief is left alone). Free object preferred, else queue at an
        #      occupied one, else fall through to the model (never force a bad target).
        if _HYBRID_NEED_RELIEF and agent is not None:
            _need = _relief_need(agent)
            if _need is None:
                _NEED_GRACE.pop(aid, None)
            elif (d.get("chosen") or {}).get("action") in {
                    _NEED_ACTION[n] for n in ("bladder", "thirst", "hunger", "energy")
                    if _need_past_relief_bar(agent, n)}:
                _NEED_GRACE.pop(aid, None)                       # model is already relieving it -> let the GNN drive
            else:
                _NEED_GRACE[aid] = _NEED_GRACE.get(aid, 0) + 1   # desperate + model not helping -> count down grace
                if _NEED_GRACE[aid] > _RELIEF_GRACE:
                    _ract = _NEED_ACTION[_need]
                    _robj = _find_relief_object(scene, _ract, getattr(agent, "current_zone", None))
                    if _robj is not None:
                        _NEED_GRACE.pop(aid, None)
                        _opt = {"target_type": "object", "target_id": _robj.id, "action": _ract, "p": 1.0,
                                "reason_override": "hybrid:need_relief"}
                        _act = _option_to_action(world, scene, aid, _opt)
                        _act["source"] = "event"; _act["priority"] = 2; _act["semantic_origin"] = "hybrid_fallback"
                        actions.append(_act)
                        _COMMIT[aid] = {"target_type": "object", "target_id": _robj.id, "action": _ract,
                                        "ttl": _dwell(_ract), "source": "event", "priority": 2,
                                        "semantic_origin": "hybrid_fallback"}
                        agent.current_zone = _robj.zone_id
                        continue
        # 2.55) PERFORMER SEAT — a musician takes the instrument's stool and STAYS there for the set.
        #     Deterministic for the same reason the post rule below is: re-emitting the identical
        #     (object, action) every tick yields the same intent signature -> the same command_id ->
        #     Unity's duplicate gate drops it, so the agent walks over once and then simply stays put.
        #     No _COMMIT on purpose — a dwell expiry would stand them up in the middle of the gig.
        if agent is not None and (getattr(agent, "role", "") or "").lower() in _PERFORMER_ROLES:
            _seat = _performer_seat(scene)
            if _seat is not None:
                _popt = {"target_type": "object", "target_id": _seat.id, "action": "sit", "p": 1.0,
                         # The reason string PICKS THE ANIMATION CLIP in Unity (ClipForReason does a
                         # substring match on it), so it must contain "sit". "performer_seat" did not —
                         # but "seat" contains "eat", so the musician played the standing EAT clip and
                         # appeared to stand at the piano instead of sitting down.
                         "reason_override": "hybrid:performer_sit"}
                _pact = _option_to_action(world, scene, aid, _popt)
                _pact["source"] = "event"; _pact["priority"] = 2
                _pact["semantic_origin"] = "deterministic_execution"
                actions.append(_pact)
                agent.current_zone = getattr(_seat, "zone_id", agent.current_zone)
                continue
        # 2.6) STAFF DUTY POST (safe_demo_hybrid only) — a staff member with NOTHING TO DO goes back to the
        #     front desk instead of drifting off into the gallery.
        #
        #     WHAT COUNTS AS "NOTHING TO DO" FOR STAFF. Two cases hand over to the post:
        #       (a) the model produced a do-nothing choice (idle / observe / continue), and
        #       (b) the model chose something OUTSIDE the post's own zone.
        #     (b) is the important one, and (a) alone was not enough. `so_pc`/`so_printer` were the venue's
        #     only `work` objects, so the staff option mask (O1 surfaces work objects) kept sending a barista
        #     to the printer room at x~22 — a real, non-idle choice, so the rule stepped aside and staff were
        #     never at the desk. Errands still win: spill dispatch, directed leave, role directives and any
        #     live commitment are all resolved ABOVE this point and `continue` before reaching it, so a staff
        #     member cleaning or restocking is never yanked back mid-task. Work AT the desk, and talking to
        #     customers there, are in-zone and pass through untouched — which is what keeps the model in
        #     charge of what staff actually do while on station.
        #     Set ECGP_STAFF_POST=0 to disable and observe the raw policy.
        #
        #     THIS MUST PRECEDE THE STAGGER GATE. The stagger emits `idle` and `continue`s on ~2 of every
        #     3 ticks, and Unity's DirectorIdle nulls the agent's zone and clears its target — so a post
        #     walk issued on a phase tick was being CANCELLED on the next non-phase tick, over and over.
        #     The visible result was staff never actually arriving at the desk even though the rule fired.
        #     Going to your post is not a fresh decision that needs rate-limiting; it is the absence of one.
        #
        #     No _COMMIT is written on purpose. `_option_to_action` is deterministic for a given
        #     (object, agent), so re-emitting the same post action every tick yields an identical intent
        #     signature -> the same command_id -> Unity's duplicate gate drops it and there is NO path reset.
        #     Writing a commitment instead would hand the agent to the commit-honor block, whose dwell
        #     expires and would ping-pong them between post and task.
        if (_HYBRID_POST and agent is not None and not _RESEARCH_MODE
                and _social_role(getattr(agent, "role", "")) == "staff"):
            post = _post_object(scene, agent)
            _ch = d["chosen"] or {}
            # in-zone == the model is keeping them on station; let it drive
            _tz = None
            if _ch.get("target_type") == "object":
                _o = scene.objects.get(_ch.get("target_id"))
                _tz = getattr(_o, "zone_id", None) if _o is not None else None
            elif _ch.get("target_type") == "zone":
                _tz = _ch.get("target_id")
            _on_station = post is not None and _tz == post.zone_id
            _doing_nothing = _ch.get("action") in _POST_IDLE_ACTIONS
            if post is not None and (_doing_nothing or not _on_station):
                # AUTHORED POST SLOTS: the scene can pin exactly where posted staff stand (demo.json
                # `post_slots` on the post object — slot 0/1 behind the counter, later ones flanking).
                # Assigned by STABLE RANK among the scene's staff so the same person keeps the same spot
                # every tick — a re-shuffle would change the target and cause a path reset. Without slots
                # (or more staff than slots) the generic ring targeting below still applies. This is what
                # keeps a whole 4-person crew from ringing one point computed for a single agent.
                _slots = getattr(post, "post_slots", None)
                _pact = None
                if _slots:
                    _staff_ids = sorted(a2 for a2, ag2 in scene.agents.items()
                                        if _social_role(getattr(ag2, "role", "")) == "staff")
                    _rank = _staff_ids.index(aid) if aid in _staff_ids else len(_slots)
                    if _rank < len(_slots):
                        _sx, _sy = _slots[_rank]
                        _pact = {"agent_id": aid, "action": "move_to_zone", "zone_id": post.zone_id,
                                 "smart_object_id": post.id, "target_x": _sx, "target_y": _sy,
                                 # face DOWN toward the entrance/customers, welcoming
                                 "face_x": _sx, "face_y": _sy - 1.0,
                                 "reason": "ecgp:idle at the front desk"}
                if _pact is None:
                    _pact = _option_to_action(world, scene, aid, {
                        "target_type": "object", "target_id": post.id, "action": "talk", "p": 1.0,
                        # 'idle' makes ClipForReason pick the standing Idle clip. Do NOT put sit/rest/eat/work
                        # in this string — that ladder is substring-matched and 'sit' is tested first.
                        "reason_override": "ecgp:idle at the front desk"})
                _pact["semantic_origin"] = "hybrid_fallback"
                actions.append(_pact)
                agent.current_zone = post.zone_id
                continue
        # 2.7) HONOUR AN EXISTING QUEUE SPOT. A queueing agent holds its EXACT place in line — same object,
        #      same point, same action every tick — until the object frees up (then its wait converts into a
        #      real commitment: first in line takes over) or the object dies/gets disabled (lock dropped, the
        #      agent re-decides normally). Without this the queue branch below re-ran from a FRESH sample
        #      every tick and the anchor object flip-flopped — see the lock-site comment.
        if agent is not None and aid in _QUEUE_LOCK:
            _ql = _QUEUE_LOCK[aid]
            _qobj = scene.objects.get(_ql["oid"])
            _qdead = (_qobj is None or getattr(_qobj, "removed", False)
                      or world.overlays.is_disabled(_ql["oid"], _ql["action"])
                      or (_qobj is not None and world.overlays.is_disabled(_qobj.zone_id, _ql["action"]))
                      or (_qobj is not None and world.overlays.is_hazard(_qobj.zone_id)))
            if _qdead:
                _QUEUE_LOCK.pop(aid, None)                # target gone/closed -> fall through, decide fresh
            elif len(_committed_object_users(_ql["oid"], exclude=aid)) >= _obj_capacity(_qobj):
                actions.append({"agent_id": aid, "action": "move_to_zone", "zone_id": _qobj.zone_id,
                                "target_x": _ql["pt"][0], "target_y": _ql["pt"][1],
                                "face_x": _qobj.pos[0], "face_y": _qobj.pos[1],
                                "reason": "ecgp:queue", "semantic_origin": "deterministic_execution"})
                agent.current_zone = _qobj.zone_id
                continue
            else:
                # a slot opened and this agent was already waiting: take it NOW, ahead of the sampler
                _QUEUE_LOCK.pop(aid, None)
                _served = {"target_type": "object", "target_id": _ql["oid"], "action": _ql["action"], "p": 1.0}
                _sact = _option_to_action(world, scene, aid, _served)
                _sact["source"] = "normal"; _sact["priority"] = 0; _sact["semantic_origin"] = "learned_ecgp"
                actions.append(_sact)
                _COMMIT[aid] = {"target_type": "object", "target_id": _ql["oid"], "action": _ql["action"],
                                "ttl": _dwell(_ql["action"]), "source": "normal", "priority": 0,
                                "semantic_origin": "learned_ecgp"}
                continue
        # 3) STAGGER: an uncommitted, non-urgent agent only re-decides on its own phase tick — otherwise it
        #    holds (idle in place), so the crowd re-plans in a rolling, desynchronised way instead of all at
        #    once. Urgent needs and events (handled above) already bypass this.
        if agent is not None and not _my_turn(aid, scene.tick_no) and not _urgent_tier1(agent):
            actions.append({"agent_id": aid, "action": "idle", "zone_id": agent.current_zone,
                            "reason": "ecgp:idle", "semantic_origin": "deterministic_execution"})   # scheduling, not a decision
            continue
        # 4) otherwise pick a fresh (sampled) action and COMMIT to it for a short dwell.
        chosen = d["chosen"]
        if chosen["action"] == "leave":                  # suppress spurious leaving in a normal tick
            alt = max((o for o in d["options"] if o["action"] != "leave"),
                      key=lambda o: o["p"], default=None)
            if alt:
                chosen = alt

        # ZONE-ACTION GATE: the chosen action must make sense in its target zone (no sit/rest at the toilet,
        # no eating in the restroom). Violations swap to the best ALLOWED option from the model's own
        # distribution; nothing allowed -> idle this tick.
        here = getattr(agent, "current_zone", None) if agent is not None else None
        def _tgt_zone(o):
            if o["target_type"] == "object":
                ob = scene.objects.get(o["target_id"])
                return getattr(ob, "zone_id", None) if ob is not None else None
            if o["target_type"] == "zone":
                return o["target_id"]
            return here                                   # self/agent target -> acts WHERE THE AGENT IS
        tz_g = _tgt_zone(chosen)
        if tz_g is not None and not _action_ok(scene, tz_g, chosen["action"]):
            alt = None; best_p = -1.0
            for o in d["options"]:
                if o["action"] == "leave" or o["p"] <= best_p:
                    continue
                oz = _tgt_zone(o)
                if oz is None or _action_ok(scene, oz, o["action"]):
                    alt = o; best_p = o["p"]
            if alt is None:
                actions.append({"agent_id": aid, "action": "idle", "zone_id": getattr(agent, "current_zone", ""),
                                "reason": "ecgp:idle", "semantic_origin": "deterministic_execution"})   # nothing legal here
                continue
            chosen = alt
        # SIT NEEDS A SEAT: sit/rest with a zone/self target would play the sit pose on the OPEN FLOOR.
        # Redirect to a free seat object (chair/sofa/bench); none free -> observe instead (stand, no sit pose).
        if chosen["action"] in ("sit", "rest") and chosen["target_type"] != "object":
            seat_o = _find_free_seat(scene, "sit")
            if seat_o is not None:
                chosen = {"target_type": "object", "target_id": seat_o.id,
                          "action": chosen["action"], "p": chosen.get("p", 1.0)}
            else:
                chosen = dict(chosen); chosen["action"] = "observe"
        # V2.1 CLUSTER CHAIN (section 5/6) — "eat@dining_cluster_root" is the model's SEMANTIC pick (one
        # macro-option, not three independent guesses at table+chair+counter); execution expands it using the
        # cluster's OWN authoritative membership (dsag_bridge.build_dining_clusters), not a zone-function
        # heuristic: acquire at the cluster's linked provider -> reserve one of ITS bound chairs -> consume
        # there. A cluster with no currently-free bound chair falls through to normal behaviour (infeasible
        # this tick, not a crash) rather than ever inventing an unbound seat.
        if (_CLUSTER_MASK_V2 and chosen["target_type"] == "object"
                and getattr(world.objects.get(chosen["target_id"]), "functional_role", None) == "cluster_root"):
            cluster = world.objects[chosen["target_id"]]     # graph-side synthesized object (not in scene.objects)
            members = cluster.members
            provider = scene.objects.get(members["provider"])
            free_chair = next((scene.objects[c] for c in members["chairs"]
                               if c in scene.objects and scene.objects[c].capacity_ok()), None)
            if provider is not None and free_chair is not None:
                tok = free_chair.reserve(aid)
                chain_id = f"{aid}_{scene.tick_no}_{cluster.id}"
                # Explicit ChainState: both the provider and the seat are already known and reserved here,
                # unlike the teacher's simplified same-tick model, so provider_id, seat_id and slot_id are
                # all set at creation rather than discovered later.
                cs = ChainState(chain_id=chain_id, semantic_goal="eat", semantic_target=cluster.id,
                                cluster_id=cluster.id, provider_id=provider.id, seat_id=free_chair.id,
                                slot_id=tok)
                cs.stage = ChainStage.APPROACH_PROVIDER; cs.stage_started_tick = scene.tick_no
                _COMMIT[aid] = {"steps": [
                    {"target_type": "object", "target_id": provider.id, "action": "eat",
                     "ttl": 1, "reason_override": "ecgp:order"},
                    {"target_type": "object", "target_id": free_chair.id, "action": "eat",
                     "ttl": _dwell("eat"), "face_override": tuple(getattr(provider, "prop_pos", provider.pos))}],
                    "i": 0, "src": provider.id, "seat": free_chair.id, "token": tok,
                    "semantic_origin": "learned_ecgp", "semantic_goal": "eat", "semantic_target": cluster.id,
                    "chain_id": chain_id, "chain_state": cs}
                step0 = _option_to_action(world, scene, aid, _COMMIT[aid]["steps"][0])
                # semantic_* describes the FROZEN goal the model chose (the cluster); execution_* describes
                # THIS tick's concrete sub-target (the provider) — the executor picked WHICH provider/chair
                # satisfies the goal, it did not invent "eat" or pick the cluster itself.
                step0["semantic_origin"] = "learned_ecgp"; step0["semantic_goal"] = "eat"
                step0["semantic_target"] = cluster.id
                step0["chain_id"] = chain_id; step0["chain_stage"] = 0
                step0["stage"] = cs.stage.value; step0["failure_reason"] = cs.failure_reason
                step0["execution_target"] = provider.id; step0["execution_origin"] = "deterministic_execution"
                _log_chain_stage(aid, "eat", cluster.id, "learned_ecgp", chain_id, 0, provider.id, cs)
                actions.append(step0)
                continue
        # PHASE B — ACTION CHAIN: eat/drink whose SOURCE sits in a service zone becomes a two-step behavior:
        # (1) acquire at the counter/source (brief, reason "ecgp:order"), (2) carry it to a reserved SEAT
        # (a chair bound to a table when the grammar has one) and consume THERE. Multi-object, one commitment;
        # every step arrival-gated. No free seat -> fall through to the normal eat-at-source behavior.
        if _CHAINS and chosen["target_type"] == "object" and chosen["action"] in ("eat", "drink"):
            srco = scene.objects.get(chosen["target_id"])
            zsrc = scene.zones.get(getattr(srco, "zone_id", "")) if srco is not None else None
            if (srco is not None and zsrc is not None
                    and "service" in (getattr(zsrc, "zone_function", None) or [])):
                seat = _find_free_seat(scene, chosen["action"])
                if seat is not None and seat.id != srco.id:
                    tok = seat.reserve(aid)
                    chain_id = f"{aid}_{scene.tick_no}_{srco.id}"
                    # Same ChainState as the CLUSTER CHAIN branch, with cluster_id=None: this chain has no
                    # authored dining-cluster grouping, since the source/seat pairing comes from a
                    # same-zone or service-zone heuristic rather than a grounded cluster.
                    cs = ChainState(chain_id=chain_id, semantic_goal=chosen["action"], semantic_target=srco.id,
                                    cluster_id=None, provider_id=srco.id, seat_id=seat.id, slot_id=tok)
                    cs.stage = ChainStage.APPROACH_PROVIDER; cs.stage_started_tick = scene.tick_no
                    _COMMIT[aid] = {"steps": [
                        {"target_type": "object", "target_id": srco.id, "action": chosen["action"],
                         "ttl": 1, "reason_override": "ecgp:order"},
                        {"target_type": "object", "target_id": seat.id, "action": chosen["action"],
                         "ttl": _dwell(chosen["action"]),
                         "face_override": tuple(getattr(srco, "prop_pos", srco.pos))}],   # face the food
                        "i": 0, "src": srco.id, "seat": seat.id, "token": tok,
                        "semantic_origin": "learned_ecgp", "semantic_goal": chosen["action"],
                        "semantic_target": srco.id, "chain_id": chain_id, "chain_state": cs}
                    step0 = _option_to_action(world, scene, aid, _COMMIT[aid]["steps"][0])
                    # the eat/drink DECISION (goal+source object) came from ECGP; the seat used to consume it
                    # is a sub-target the executor grounds, not a new semantic pick.
                    step0["semantic_origin"] = "learned_ecgp"; step0["semantic_goal"] = chosen["action"]
                    step0["semantic_target"] = srco.id
                    step0["chain_id"] = chain_id; step0["chain_stage"] = 0
                    step0["stage"] = cs.stage.value; step0["failure_reason"] = cs.failure_reason
                    step0["execution_target"] = srco.id; step0["execution_origin"] = "deterministic_execution"
                    _log_chain_stage(aid, chosen["action"], srco.id, "learned_ecgp", chain_id, 0, srco.id, cs)
                    actions.append(step0)              # acquire->seat sequencing is deterministic execution
                    continue
        # QUEUE (R4): if the object is already at capacity, WAIT in line behind it instead of piling on — the
        # ones already using finish first, then the next in line takes over (checked next decision tick).
        if _QUEUE_ENABLED and chosen["target_type"] == "object":
            oid = chosen["target_id"]; obj = scene.objects.get(oid)
            # `obj` CAN be None: options are encoded at tick start, but an object can be removed MID-tick
            # (a spill whose cleanup completes, a consumable cleared) — and _obj_capacity(None)==1 with any
            # OTHER agent still committed to the dead id sent this branch into `obj.zone_id` on None. That
            # single AttributeError aborted ecgp_tick -> rule-engine fallback for the whole crowd ("staff on
            # standby", no object targets) — a far worse outcome than one agent skipping its queue check.
            # With obj None we fall through to _option_to_action, whose object branch is already guarded.
            if obj is not None and len(_committed_object_users(oid, exclude=aid)) >= _obj_capacity(obj):
                qt = _queue_target(world, oid, aid)
                if qt is not None:
                    qt = _clamp_zone(scene, obj.zone_id, qt[0], qt[1])
                    # LOCK THE QUEUE. "No commitment" here used to mean the agent RE-SAMPLED its target
                    # every tick while waiting: tick A it queued behind the toilet, tick B the sampler
                    # picked the sink, tick C the toilet again — a different anchor and a different queue
                    # point each time, so Unity took a full path reset every ~5s and every capacity-blocked
                    # agent visibly twitched in line. With saturated needs that was HALF THE CROWD — the
                    # dominant remaining flicker. The lock pins (object, point) until the object frees up
                    # (then this agent converts its place in line into a real commitment, in _QUEUE_LOCK
                    # handling above) or the object dies/gets disabled.
                    _QUEUE_LOCK[aid] = {"oid": oid, "pt": qt, "action": chosen["action"]}
                    actions.append({"agent_id": aid, "action": "move_to_zone", "zone_id": obj.zone_id,
                                    "target_x": qt[0], "target_y": qt[1], "face_x": obj.pos[0], "face_y": obj.pos[1],
                                    "reason": "ecgp:queue", "semantic_origin": "deterministic_execution"})   # capacity wait
                    agent.current_zone = obj.zone_id
                    continue                             # hold the queue spot; no commitment
        fresh_act = _option_to_action(world, scene, aid, chosen)
        fresh_act["source"] = "normal"; fresh_act["priority"] = 0; fresh_act["semantic_origin"] = "learned_ecgp"
        actions.append(fresh_act)
        _COMMIT[aid] = {"target_type": chosen["target_type"], "target_id": chosen["target_id"],
                        "action": chosen["action"], "ttl": _dwell(chosen["action"]),
                        "source": "normal", "priority": 0, "semantic_origin": "learned_ecgp"}
    # a spill cleaner that had NO decision entry (e.g. a spawned staffer) still needs its clean action emitted
    for aid, ov in spill_overrides.items():
        if not any(a["agent_id"] == aid for a in actions):
            ov.setdefault("semantic_origin", "deterministic_execution")
            actions.append(ov)
    # R5: retire agents that have finished walking out (Unity-confirmed, or the safety-net timeout) — remove
    # them from the sim so the server stops deciding for them and they don't re-appear.
    for aid in _to_remove:
        # release any reservation the departing agent still held, or the object stays "occupied" by a
        # despawned agent forever (same leak class as the override branches — see _drop_commit)
        _drop_commit(scene, aid)
        scene.agents.pop(aid, None); _LEAVING.pop(aid, None); _FRUSTRATION.pop(aid, None)
        _QUEUE_LOCK.pop(aid, None)
    # LEAVE LIFECYCLE: a leave patch whose entire frozen target list has left is complete, so drop it
    # immediately rather than waiting for its ttl. It then stops being reported as active, and the
    # 'resolved to NOBODY' warning cannot fire for a patch whose job is already done.
    if _to_remove:
        done_patches = _dp_leave.leave_patches_complete(getattr(scene, "active_patches", []) or [], scene)
        if done_patches:
            scene.active_patches = [p for p in scene.active_patches if p not in done_patches]
            for p in done_patches:
                log.info(f"[ecgp] leave patch '{p.display_name}' complete — all "
                         f"{len(p.resolved_leavers)} member(s) exited")
    # R5 frustration counter = CONSECUTIVE ticks an agent spends stuck in a queue; any other action resets it.
    # An agent still queueing past _FRUST_LIMIT gives up next tick (decided in the loop above).
    for a in actions:
        aid2 = a["agent_id"]; r = a.get("reason", "")
        if r == "ecgp:queue":
            _FRUSTRATION[aid2] = _FRUSTRATION.get(aid2, 0) + 1
        elif r == "ecgp:idle":
            pass                                          # holding/waiting — neither progress nor a reset
        elif aid2 not in _LEAVING:
            _FRUSTRATION[aid2] = 0                        # got served / did something else -> patience refreshed
    # LIVE OCCUPANCY (R4): an object is 'in use' only when a committed agent is actually AT it (within
    # _USE_RADIUS of its position, using the live Unity positions mirrored in update_state) — a committed
    # agent still WALKING toward it does not light the object up. No position data yet -> not in use.
    if _QUEUE_ENABLED:
        counts = {}
        for a, c in _COMMIT.items():
            c = _com_cur(c)                               # a chain occupies its CURRENT step's object
            if c.get("target_type") != "object":
                continue
            ag = scene.agents.get(a)
            obj = scene.objects.get(c["target_id"])
            pos = getattr(ag, "pos", None) if ag is not None else None
            if obj is None or pos is None:
                continue
            dx, dy = pos[0] - obj.pos[0], pos[1] - obj.pos[1]
            if dx * dx + dy * dy <= _USE_RADIUS * _USE_RADIUS:
                counts[c["target_id"]] = counts.get(c["target_id"], 0) + 1
        for oid, o in scene.objects.items():
            if not getattr(o, "removed", False):
                o.occupancy = counts.get(oid, 0)
                o.occupied_by = "in_use" if counts.get(oid, 0) > 0 else None
    _social_tick(scene, actions)                     # live relationships: chats earn affinity
    object_states = [o.render_state() for o in scene.objects.values()]
    _log_origin_breakdown(scene, actions)
    return actions, object_states, _drain_obj_log(scene)


def _zone_dist(scene, za, zb):
    """Euclidean distance between two zone centers (for picking the nearest attracted zone)."""
    a = scene.zones.get(za); b = scene.zones.get(zb)
    if a is None or b is None:
        return 1e9
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
