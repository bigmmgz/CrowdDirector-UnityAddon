"""
Crowd Simulation Director — MCP Server
Maslow-hierarchy needs, social relationship system, free-text event interpretation.
"""

import asyncio
import base64
import json
import os
import re
import logging
import random as rnd
import shutil
import time
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import websockets
import anthropic

import sprite_gen
import dsag_bridge
import event_intent   # canonical event-intent resolver (LLM intent + MiniLM semantic fallback)
import grid_layout
import scene_spec
from dsag.patch import ScenePatch, PATCH_OP_KINDS, apply_structural_ops

# On-demand LPC character generation (pixel-art animated sheets). Guarded so the server still
# runs if the assetgen package or the cloned LPC repo is missing.
# DISABLED by default: Unity now renders agents with the multi-clip LimeZu characters (EnableLimeZuChars),
# so streaming LPC sheets is wasted work AND a late LPC swap would fight the LimeZu character on screen.
# Set STREAM_LPC=1 to re-enable the LPC path (e.g. if LimeZu is turned off in Unity).
try:
    from assetgen import lpc_runtime
    _LPC_OK = os.environ.get("STREAM_LPC", "0") == "1"
except Exception as _lpc_err:
    lpc_runtime = None
    _LPC_OK = False
    logging.getLogger("Director").warning(f"[lpc] runtime character generation disabled: {_lpc_err}")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("Director")

# ── INTENT COMMITMENT: command_id / intent_id tagging (the FINAL word on new-vs-resent) ────────────────────
# Every serialized action passes through `_finalize_intents` (the single choke point right before it is sent
# to Unity) exactly once. `intent_id` is a stable signature of WHAT the agent has been told to do; it stays
# IDENTICAL tick to tick while the decision hasn't changed, and `command_id` only advances when it does. Unity
# dedups by command_id — an EXPLICIT fact from Python, not a distance/time heuristic guessed on the Unity side.
# This is the mechanism for "separate new decision from same decision sent again on the next tick".
_LAST_INTENT_ID  = {}    # agent_id -> last intent_id string
_LAST_COMMAND_ID = {}    # agent_id -> last command_id int
_MEET_MET        = {}    # meet-pair key -> first tick the pair converged (dsag_patch.meet_patches_complete)
_CMD_SEQ = [0]           # monotonically increasing counter (mutable cell so the closure below can bump it)
_PRIORITY_BY_SOURCE = {"emergency": 4, "directive": 3, "event": 2, "interaction": 1, "normal": 0}
# SCENE EPOCH (item 4): bumped once per generate_scene (see OnSceneReady send). Unity rejects any action
# whose scene_epoch doesn't match its currently-loaded scene — agent ids are reused from zero each scene, so
# without this a stale in-flight command from the PREVIOUS scene could be misapplied to a same-numbered
# agent in the new one.
_SCENE_EPOCH = [0]
_AMBIENT_CADENCE = [None]        # per-scene ambient event cadence (see ecgp.runtime.ambient.configure)


def _reset_behavior_state():
    """Wipe the behaviour engines' per-agent bookkeeping on every new scene.

    `ecgp.runtime.live_bridge` keeps commitments, chain state, leave counters and party slots in module-level
    dicts keyed by AGENT ID — and agent ids restart at `agent_0` in every scene. Without this a freshly loaded
    scene inherits the previous one's commitments, so an agent is sent to a smart object that belonged to the
    OLD venue and no longer exists. Also clears the meet bookkeeping held here."""
    try:
        from ecgp.runtime import live_bridge as _lb
        _lb.reset_scene_state()
    except Exception as e:                       # never let a reset failure block a scene load
        log.warning(f"[scene] behaviour-state reset skipped: {e}")
    _MEET_MET.clear()
    try:
        from ecgp.runtime import ambient as _amb
        _AMBIENT_CADENCE[0] = _amb.configure(getattr(sim, "_scene_cfg", None))
    except Exception as e:
        log.warning(f"[ambient] configure skipped: {e}")


def _finalize_intents(actions: list) -> list:
    """Tag every FINAL action (after ALL overlays have run) with source/priority (defaulted if a code path
    didn't set them) and TWO separate identities (item I):
      movement_goal_id — action + zone/object/target-agent + rounded position ONLY. This is the PHYSICAL
        destination. Unity resets the path / calls DirectorMoveToPoint ONLY when this changes.
      intent_id (control identity) — movement_goal_id + source/priority. This is what command_id tracks for
        the plain duplicate-resend check. A normal decision that gets promoted to an event/directive with the
        SAME destination changes intent_id/command_id (so the control metadata — activity log, priority — is
        updated) but must NOT reset the path, because movement_goal_id is unchanged; Unity is responsible for
        making that distinction (see CrowdDirector.ApplyActions).
    This is the ONLY place that computes them — so 'the final serialized action for every agent' (not an
    intermediate/object-targeted subset) is what gets traced.

    ONE ACTION PER AGENT is enforced here, before any id is minted. ecgp_tick can legitimately append twice
    for one agent (a chain continuation, then an `idle` when the seat reservation turned out to be lost), and
    the whole intent architecture silently breaks on that: the second entry gets a FRESH command_id, because
    the first iteration already overwrote _LAST_INTENT_ID[aid] — so Unity's duplicate gate lets both through,
    applies them in order, and the `idle` wins. DirectorIdle nulls _currentZone, so the agent ends up frozen
    (not merely re-pathed) until the next order. Collapsing to the LAST entry matches what the meet/leave
    overlays already assume via their `by_id` view, and keeps the wire consistent with it.
    NOTE: the list is mutated IN PLACE. The caller serialises its own `actions` object and ignores the return
    value, so rebinding a filtered list here would compile, log correctly and change nothing on the wire."""
    if len(actions) > 1:
        seen = {}
        for a in actions:
            seen[a.get("agent_id")] = a          # keep last
        if len(seen) != len(actions):
            dupes = len(actions) - len(seen)
            log.warning(f"[intent] {dupes} duplicate action(s) collapsed — an agent got more than one "
                        f"decision this tick (chain continuation + reservation-lost idle is the known case)")
            actions[:] = list(seen.values())
    for a in actions:
        aid = a.get("agent_id")
        src = a.get("source") or "normal"
        prio = a.get("priority")
        if prio is None:
            prio = _PRIORITY_BY_SOURCE.get(src, 0)
        a["source"], a["priority"] = src, prio
        # round coordinates: two structurally-identical decisions must compare equal even if a downstream
        # zone-clamp/party-slot recompute introduces float noise in the last couple of decimal places.
        tx = a.get("target_x"); ty = a.get("target_y")
        rx = round(tx, 2) if isinstance(tx, (int, float)) else ""
        ry = round(ty, 2) if isinstance(ty, (int, float)) else ""
        goal_sig = "|".join(str(x) for x in (
            a.get("action"), a.get("zone_id"), a.get("smart_object_id"), a.get("target_agent_id"), rx, ry))
        sig = f"{goal_sig}|{src}"
        if _LAST_INTENT_ID.get(aid) == sig:
            cmd = _LAST_COMMAND_ID.get(aid, 0) or 1
        else:
            _CMD_SEQ[0] += 1
            cmd = _CMD_SEQ[0]
            _LAST_INTENT_ID[aid] = sig
            _LAST_COMMAND_ID[aid] = cmd
        a["intent_id"], a["movement_goal_id"], a["command_id"] = sig, goal_sig, cmd
        a["scene_epoch"] = _SCENE_EPOCH[0]
    return actions

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HOST    = "localhost"
PORT    = 8765
MODEL   = "claude-sonnet-4-6"

if not API_KEY:
    log.error("ANTHROPIC_API_KEY not set! Set the environment variable before starting.")

# ── EventLog folder ─────────────────────────────────────────────────────────────
# Defaults to server/EventLog; override with CROWDDIRECTOR_EVENTLOG to point it at a
# Unity project instead.
#   scenes/    — one JSON per generated scene (zones + character types)
#   inbox/     — DROP a .txt/.json file here to trigger an event (works like the
#                hardcoded buttons; picked up on the next director tick)
#   processed/ — inbox files are moved here after they fire
#   events/    — JSON record of every interpreted event (from UI or inbox)
EVENTLOG = Path(os.environ.get("CROWDDIRECTOR_EVENTLOG",
                               Path(__file__).resolve().parent / "EventLog"))
for _sub in ("scenes", "inbox", "processed", "events"):
    (EVENTLOG / _sub).mkdir(parents=True, exist_ok=True)
try:
    (EVENTLOG / "README.txt").write_text(
        "Drop a .txt or .json file into 'inbox/' with a sentence describing an event,\n"
        "e.g. 'Closing time' or 'Alex wants to go to the Exit'.\n"
        "The server picks it up on the next tick, interprets it, and directs the agents\n"
        "(emergencies evacuate everyone, just like the Fire Alarm button).\n"
        "Processed files move to 'processed/'. Scene dumps are in 'scenes/'.\n",
        encoding="utf-8",
    )
except Exception:
    pass
log.info(f"EventLog folder: {EVENTLOG}")

# ── State ──────────────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    id:           str
    name:         str
    agent_type:   str
    personality:  str = "extrovert"
    x: float = 0.0
    y: float = 0.0
    needs: dict = field(default_factory=lambda: {
        "hunger": 20.0, "thirst": 20.0, "bladder": 5.0, "energy": 85.0,
        "stress": 10.0, "loneliness": 15.0, "groupAffinity": 30.0,
        "status": 20.0, "curiosity": 50.0, "urgentTier": 1
    })
    current_zone:     str  = ""
    group_id:         Optional[str] = None                # social unit (family/friend party) — item 6
    relationships:    list = field(default_factory=list)  # [{other_id, familiarity, trust, tension, status}]
    friends:          list = field(default_factory=list)
    encounter_counts: dict = field(default_factory=dict)

class SimState:
    def __init__(self):
        self.agents:          dict[str, AgentState] = {}
        self.social_groups:   dict                  = {}   # group_id -> {"type","members"} (item 6)
        self.zones:           list[dict]             = []
        self.scene_name:      str                    = ""
        self.theme:           str                    = "cafe"   # controlled venue theme -> Unity floor plan/props/clothing
        self.scene_spec:      Optional[dict]         = None     # open-vocab SceneSpec (kits/smart_objects/affordances)
        self.scene_graph:     Optional[dict]         = None     # placed graph exported back from Unity (behavior input)
        self.description:     str                    = ""
        self.behaviour_notes: str                    = ""
        self.events:          list[dict]             = []
        self.pending_actions: list[dict]             = []
        self.active_event:    str                    = ""
        self.event_message:   str                    = ""
        self.event_hint:      str                    = ""  # one-shot hint, injected into the very next tick only
        self.standing_orders: list[str]              = []  # persistent per-agent commands, enforced every tick until cleared
        # ── dsag path (isolated from the LLM director; "Generate with DSAG" button) ──
        self.director_mode:   str                    = "llm"   # "llm" | "dsag"
        self.dsag_scene                              = None    # dsag.SceneModel when in dsag mode
        # per-agent behavior back-end for the graph path: "ecgp" (the trained Event-Conditioned Graph
        # Policy, 64x3 → outputs/final_64x3) or "rule" (deterministic symbolic engine). LLM stays the
        # director either way; this only picks who turns the patched Scene-Affordance Graph into each
        # agent's move. Defaults to ECGP; falls back to "rule" if the checkpoint/torch is unavailable.
        self.behavior_engine: str                    = os.environ.get("DSAG_BEHAVIOR", "ecgp").lower()

sim = SimState()

# ── Director system prompt ─────────────────────────────────────────────────────
DIRECTOR_SYSTEM = """You are an AI crowd behavioral director for a 2D simulation.
Your role: assign each agent an action each tick that makes the crowd feel alive and realistic.

MASLOW HIERARCHY RULES (strictly enforce this):
- Tier 1 (Physiological): hunger, thirst, bladder, energy. If ANY of these is urgent, the agent MUST address it before anything else.
- Tier 2 (Safety): stress. If stress > 60, agent should seek open/calm spaces or exit.
- Tier 3 (Belonging): loneliness, groupAffinity. Agent seeks social contact / familiar peers.
- Tier 4 (Esteem): status. VIPs seek prominent zones; shy/loner types avoid crowds.
- Tier 5 (Engagement): curiosity. Agent explores when all other needs satisfied.

The field "urgentTier" tells you the lowest tier that needs attention. NEVER send a Tier 3 action to an agent with urgentTier=1.

RELATIONSHIP RULES:
- "friend" pairs: consider sending them to the same zone (group_move) or starting a conversation.
- "avoiding" pairs: NEVER send them to the same zone or assign start_conversation between them.
- "acquaintance" pairs: light conversations are good; they can become friends over time.

ACTIONS available:
- move_to_zone: {"agent_id","action":"move_to_zone","zone_id","reason"}
- start_conversation: {"agent_id","action":"start_conversation","target_agent_id","reason"}
- group_move: {"agent_id","action":"group_move","zone_id","target_agent_id","reason"} — friends move together
- rest: {"agent_id","action":"rest","reason"}
- idle: {"agent_id","action":"idle","reason"}

Return ONLY a JSON array covering ALL agents. No markdown, no extra text."""

# ── Theme classification ───────────────────────────────────────────────────────
THEME_VOCAB = ("cafe", "restaurant", "office", "hospital", "school", "shop",
               "home", "nightclub", "airport", "park", "gym", "museum")
# keyword -> canonical theme (checked against scene_name + description + zone words)
_THEME_KEYWORDS = [
    ("gym", "gym"), ("fitness", "gym"), ("workout", "gym"), ("weight", "gym"),
    ("hospital", "hospital"), ("clinic", "hospital"), ("ward", "hospital"), ("patient", "hospital"), ("emergency room", "hospital"),
    ("museum", "museum"), ("gallery", "museum"), ("exhibit", "museum"),
    ("office", "office"), ("workplace", "office"), ("meeting", "office"), ("cubicle", "office"),
    ("nightclub", "nightclub"), ("club", "nightclub"), ("disco", "nightclub"), ("dance floor", "nightclub"),
    ("airport", "airport"), ("terminal", "airport"), ("departure", "airport"),
    ("school", "school"), ("classroom", "school"), ("university", "school"), ("lecture", "school"),
    ("park", "park"), ("garden", "park"), ("playground", "park"),
    ("restaurant", "restaurant"), ("diner", "restaurant"), ("dining", "restaurant"),
    # café before generic shop/store so "coffee shop" resolves to cafe, not shop
    ("cafe", "cafe"), ("café", "cafe"), ("coffee", "cafe"), ("pub", "cafe"), ("bar", "cafe"), ("bakery", "cafe"),
    ("store", "shop"), ("market", "shop"), ("mall", "shop"), ("grocery", "shop"), ("retail", "shop"), ("shop", "shop"),
    ("home", "home"), ("house", "home"), ("apartment", "home"), ("living room", "home"), ("bedroom", "home"),
]

def classify_theme(claude_theme: str | None, description: str, config: dict) -> str:
    """Trust Claude's theme if it is in-vocabulary; otherwise infer from keywords; default cafe."""
    if claude_theme:
        t = claude_theme.strip().lower()
        if t in THEME_VOCAB:
            return t
    hay = (description + " " + config.get("scene_name", "") + " " +
           " ".join(z.get("label", "") + " " + z.get("zone_type", "") for z in config.get("zones", []))).lower()
    for kw, theme in _THEME_KEYWORDS:
        if kw in hay:
            return theme
    return "cafe"

# ── Crowd-size control ─────────────────────────────────────────────────────────
# Hard ceiling so a runaway prompt ("a billion people") can't OOM the process. Generous by default and
# env-overridable — the real limit is Unity rendering + GNN memory, not the server. The LLM is called ONCE
# per scene regardless of crowd size; the server instantiates every individual from the agent_types.
MAX_AGENTS = int(os.environ.get("MAX_AGENTS", "5000"))

# a number followed by a people-word (generic crowd terms + common activity roles), or a leading
# "crowd/group of N" — captures the user's crowd size. Object words (lanes/tables/stars) are NOT here,
# so "a 4-lane pool with 40 swimmers" reads 40, not 4. Any role not listed still works via "N people".
_COUNT_WORD = (r"(\d[\d,]*)[-\s]*(?:people|persons?|ppl|agents?|humans?|characters?|npcs?|crowd|guests?|"
               r"visitors?|attendees?|shoppers?|passengers?|fans?|students?|pupils?|patrons?|customers?|"
               r"clients?|members?|tourists?|travell?ers?|commuters?|spectators?|protesters?|worshipp?ers?|"
               r"swimmers?|bathers?|sunbathers?|runners?|joggers?|dancers?|clubbers?|partygoers?|gamers?|"
               r"players?|athletes?|diners?|drinkers?|workers?|employees?|staff|nurses?|doctors?|patients?|"
               r"teachers?|kids?|children|adults?|men|women)")
_COUNT_RE = re.compile(_COUNT_WORD, re.I)
_LEAD_RE  = re.compile(r"^\s*(?:a\s+)?(?:crowd|group|party|team|class)\s+of\s+(\d[\d,]*)", re.I)


def parse_requested_count(description: str):
    """Pull an explicit crowd size from the prompt: '40 people', 'a crowd of 500', '2000 fans'. None if
    the user gave no number (then the LLM picks 15-30 as before)."""
    m = _LEAD_RE.search(description) or _COUNT_RE.search(description)
    if not m:
        return None
    try:
        n = int(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return n if n > 0 else None


def enforce_agent_count(config: dict, requested: int):
    """Scale the per-type `count`s so the TOTAL equals `requested` exactly (largest-remainder rounding,
    >=1 per kept type). The LLM defines the TYPES + rough proportions; the server decides the exact head
    count. If fewer people than types are asked for, keep only that many types."""
    types = config.get("agent_types") or []
    if not types or not requested:
        return
    requested = max(1, min(requested, MAX_AGENTS))
    if requested <= len(types):
        config["agent_types"] = types[:requested]
        for t in config["agent_types"]:
            t["count"] = 1
        return
    weights = [max(1, int(t.get("count", 1) or 1)) for t in types]
    tot = sum(weights)
    raw  = [requested * w / tot for w in weights]
    base = [max(1, int(x)) for x in raw]
    diff = requested - sum(base)
    frac_order = sorted(range(len(types)), key=lambda i: -(raw[i] - int(raw[i])))
    i = 0
    while diff > 0:                                   # hand extras to the largest fractional remainders
        base[frac_order[i % len(frac_order)]] += 1; diff -= 1; i += 1
    big_order = sorted(range(len(types)), key=lambda i: -base[i])
    i = 0
    while diff < 0:                                   # min-1 bumping overshot -> trim from the biggest types
        j = big_order[i % len(big_order)]
        if base[j] > 1:
            base[j] -= 1; diff += 1
        i += 1
    for t, c in zip(types, base):
        t["count"] = c


# ── Social units (item 6): people arrive in parties. The LLM may set `social_unit` per agent_type; else it
# is inferred from the type/personality. Individuals of a grouped type are chunked into small groups that
# share a group_id + an authored relationship (family/friend), which is what surfaces talk/help/group options
# in the ECGP graph and lets the behavior prior express family cohesion. ────────────────────────────────────
_FAMILY_KW = ("family", "parent", "child", "kid", "couple", "mother", "father", "grandparent", "toddler")
_SOLO_KW = ("worker", "staff", "barista", "waiter", "guard", "security", "clerk", "nurse", "cashier",
            "chef", "cook", "attendant", "solo", "loner", "business", "employee", "vendor", "guide")
_FRIEND_KW = ("friend", "tourist", "student", "buddy", "mate", "party", "tour", "group", "fan", "visitor")
_UNIT_SIZE = {"solo": 1, "pair": 2, "friends": 3, "family": 4, "group": 4}
_UNIT_RTYPE = {"pair": "friend", "friends": "friend", "family": "family", "group": "acquaintance"}


def _social_unit_for(atype: dict) -> str:
    u = (atype.get("social_unit") or "").strip().lower()
    if u in _UNIT_SIZE:
        return u
    blob = f"{atype.get('type', '')} {atype.get('personality_type', '')}".lower()
    if any(k in blob for k in _FAMILY_KW):
        return "family"
    if any(k in blob for k in _SOLO_KW):
        return "solo"
    if any(k in blob for k in _FRIEND_KW):
        return "friends"
    return "solo"


def assign_social_groups(agent_dicts: list, rnd) -> dict:
    """Chunk each agent_type's individuals into small social groups (family/friends), tagging every member
    dict with group_id + group_type and returning {group_id: {"type", "members"}}. Solo units get nothing."""
    groups: dict = {}
    by_type: dict = {}
    for a in agent_dicts:
        by_type.setdefault(a["agent_type"], []).append(a)
    gi = 0
    for atype, members in by_type.items():
        unit = a_unit = _social_unit_meta.get(atype, "solo")
        size = _UNIT_SIZE.get(unit, 1)
        if size <= 1:
            continue
        rtype = _UNIT_RTYPE.get(unit, "friend")
        for s in range(0, len(members), size):
            chunk = members[s:s + size]
            if len(chunk) < 2:                          # a lone remainder isn't a group
                break
            gid = f"grp_{gi}"; gi += 1
            groups[gid] = {"type": rtype, "members": [m["id"] for m in chunk]}
            for m in chunk:
                m["group_id"] = gid
                m["group_type"] = rtype
    return groups


_social_unit_meta: dict = {}   # agent_type -> social_unit (populated per scene before assign_social_groups)


def _match_zone_id(scene, zkey):
    """Resolve a zone keyword/id to a real zone id in the dsag scene (reuses the patch zone matcher)."""
    if not zkey:
        return None
    if zkey in scene.zones:
        return zkey
    from dsag.patch import PatchOp, zone_matches
    op = PatchOp(op="_", zone=zkey)
    return next((z.id for z in scene.zones.values() if zone_matches(op, z)), None)


# ── Scene generation ───────────────────────────────────────────────────────────
async def generate_scene(description: str) -> dict:
    client = anthropic.Anthropic(api_key=API_KEY)

    system = """You are a crowd simulation scene designer. Output ONLY valid JSON, no markdown, no backticks.

{
  "scene_name": "string",
  "scene_theme_text": "the venue in the user's own words, e.g. 'swimming pool', 'hotel lobby', 'library'",
  "scene_family": "indoor | outdoor | hybrid",
  "style_tags": ["1-3 style words, e.g. 'modern interior', 'leisure', 'institutional'"],
  "theme": "closest of: cafe, restaurant, office, hospital, school, shop, home, nightclub, airport, park, gym, museum (a coarse hint only; scene_theme_text is authoritative)",
  "zones": [
    {"id":"string","label":"string","zone_type":"string","x":number,"y":number,"w":number,"h":number,"color":"#hexcolor","sprite_prompt":"string",
     "allowed_actions":["which of: sit rest eat drink relieve wash work observe talk dance — the actions/animations that make SENSE in this zone. A toilet allows ONLY relieve+wash (nobody sits down to rest or eats there); a lounge/seating area allows sit+rest+eat+drink+talk; a counter/bar allows eat+drink+observe+talk+work; a work area allows work+sit+talk; a dance floor allows dance+drink+observe. Be strict — this gates where each animation can play."]}
  ],
  "agent_types": [
    {
      "type":"string",
      "label":"string",
      "count":number,
      "color":"#hexcolor",
      "social_unit":"solo|pair|family|friends|group  (how this type ARRIVES together — a family of 4, friends in pairs, or solo. Optional; inferred from the type if omitted)",
      "personality_type":"introvert|extrovert|socializer|loner|vip",
      "initial_needs":{
        "hunger":number,"thirst":number,"bladder":number,"energy":number,
        "stress":number,"loneliness":number,"groupAffinity":number,"status":number,"curiosity":number
      },
      "personality":"one sentence describing behavioral tendencies",
      "role":"short role word for this scene, e.g. 'swimmer', 'librarian', 'student', 'nurse', 'shopper'",
      "outfit_category":"clothing style, e.g. 'swimwear', 'business', 'scrubs', 'casual', 'sportswear'",
      "sprite_prompt":"string"
    }
  ],
  "behaviour_notes": "2-3 sentences about key crowd dynamics",
  "narrative": "a 2-3 sentence STORY opening that sets the scene like the first lines of a short story"
}

RULES:
- OPEN-VOCABULARY: the scene can be ANYTHING the user types (a swimming pool, library, hotel lobby, beach bar, airport...). Design zones + agent types that FIT that specific venue; do NOT force it into a generic template. scene_theme_text captures the venue in plain words and drives asset composition; theme is only a coarse fallback hint.
- Give each zone a clear function via its label/zone_type (e.g. 'changing room', 'reading room', 'reception', 'snack bar') so the renderer can equip it with the right smart objects (sinks/showers, desks/bookshelves, counters, food/drink) by affordance.
- theme: pick the CLOSEST of the listed keywords as a coarse hint. A pub/bar/coffee shop = cafe; a gym/fitness studio = gym; a clinic/ward = hospital; a store/market/mall = shop; a house/apartment = home; a club = nightclub; a gallery/exhibit/library = museum. If none fit well, pick the nearest.
- Coordinate system: X from -8 to 8, Y from -5 to 5 (centre is 0,0)
- List 5-8 zones. Their x/y/w/h are only rough placeholders — a layout engine repositions every zone to tile the map on a grid with NO gaps, so don't worry about exact packing, just give plausible numbers.
- ALWAYS include: a toilet/restroom zone AND at least one entrance/exit zone
- Zone types: toilet, counter, table, lounge, entrance, water, activity, shelter, vip, stage, balcony
- Total agents: if the user's description states a number of people (e.g. "40 people", "a crowd of 500", "2000 fans"), the counts across all agent_types MUST sum to that number; otherwise use 15-30. Define a SMALL set of agent_types (3-8) — the server instantiates every individual from these types, so "count" is just how many of that type and may be scaled server-side to hit the exact requested total. Large crowds (hundreds or thousands) ARE supported: keep the number of TYPES small and put big counts on them (do NOT list individuals).
- Needs 0-100. Bladder starts low (5-15). Stress starts low (5-20).
- personality_type determines behavior: introvert=avoids crowds, socializer=seeks conversation, vip=seeks status zones, loner=independent, extrovert=balanced
- social_unit is the SOCIAL STRUCTURE of the crowd: families arrive as a family (they stay near + talk to each other), friends/tourists in small groups, staff/workers solo. Set it so the scene has believable parties, e.g. a café: a "family" type (social_unit:family), a "couple"/"friends" type (friends), and solo "barista"/"remote worker" (solo).
- Set initial needs to match personality: tourists have high curiosity (60-80), VIPs have high status (50-70), loners have low loneliness (10-20)
- Colors: vivid hex colors matching the theme
- sprite_prompt (agent_types): a vivid visual description of ONE character of that type for an image generator — appearance, clothing, an item they carry, and a mood word. Do NOT mention background, style, or camera; those are added automatically. e.g. "an elderly man in a flat cap and cardigan holding a newspaper, relaxed"
- sprite_prompt (zones): a short visual description of that area's floor/surface as a top-down tile — e.g. "polished wooden cafe floor with a small round table". Do NOT mention people, background, style, or camera.
- narrative: write it like the OPENING OF A STORY, not a data summary. Set the time/weather/mood, name who is here (use the real agent types + counts) and what they're doing, in flowing prose. e.g. "On a bright Monday morning the corner cafe is already buzzing - a family of four lingers over pastries by the window while two baristas keep the espresso machine hissing and a trio of tourists drifts in from the street, cameras swinging." This is shown to the user as the scene's story, so make it vivid and human, never a bullet list."""

    # 8192 (not 2500): a rich scene — many zones + agent_types each with a verbose sprite_prompt + per-type
    # needs — overruns 2500 output tokens and the JSON gets TRUNCATED mid-structure, which surfaces as a
    # "missing ',' delimiter" parse error right at the cap (~char 7000). Extra headroom prevents that.
    response = await asyncio.to_thread(
        client.messages.create,
        model=MODEL, max_tokens=8192, system=system,
        messages=[{"role": "user", "content": f"Design a crowd simulation scene for: {description}"}]
    )
    raw     = response.content[0].text
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    start   = cleaned.find("{")
    end     = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        cleaned = cleaned[start:end]

    try:
        config = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e} — asking Claude to fix it")
        fix_response = await asyncio.to_thread(
            client.messages.create,
            model=MODEL, max_tokens=8192,   # match the generate cap so the REPAIR isn't truncated too
            messages=[{"role": "user", "content": f"Fix this invalid JSON, return ONLY valid JSON. "
                                                  f"Complete any truncated field:\n\n{cleaned}"}]
        )
        fixed = fix_response.content[0].text.strip()
        fixed = fixed.replace("```json", "").replace("```", "").strip()
        s = fixed.find("{"); e2 = fixed.rfind("}") + 1
        if s != -1 and e2 > s: fixed = fixed[s:e2]
        config = json.loads(fixed)

    # Honour an explicit crowd size the user typed ("40 people", "a crowd of 500"): scale the per-type
    # counts so the total matches exactly, instead of letting the LLM approximate (it hovered near 30).
    requested = parse_requested_count(description)
    if requested:
        enforce_agent_count(config, requested)
        got = sum(t.get("count", 0) for t in config.get("agent_types", []))
        log.info(f"Crowd size: requested {requested} -> {got} agents "
                 f"(capped at {MAX_AGENTS})" if requested > MAX_AGENTS else
                 f"Crowd size: honouring requested {requested} agents")

    # Normalise the venue theme to the controlled vocabulary (drives Unity's floor-plan template,
    # prop set, and character clothing). Fall back to keyword classification if Claude omitted it.
    config["theme"] = classify_theme(config.get("theme"), description, config)

    # Stardew-style layout: ignore whatever coordinates Claude produced and repartition the
    # world into a gapless grid of rooms, so Unity renders one continuous top-down map with
    # zones that align perfectly instead of scattered floating boxes.
    if config.get("zones"):
        grid_layout.assign_grid_layout(config["zones"])

    # OPEN-VOCABULARY SceneSpec: compose reusable ThemeKits + resolve smart objects by AFFORDANCE from the
    # catalog (works for any prompt, no fixed prefab). Behavior-agnostic — the renderer places these and the
    # DSAG/GNN behavior pipeline later targets them. Templates are only an optional per-kit shortcut.
    try:
        stt = config.get("scene_theme_text") or description
        spec = scene_spec.build_scenespec(
            stt, config.get("zones", []),
            style_tags=config.get("style_tags"),
            scene_family=config.get("scene_family", "indoor"),
            agent_roles=[t.get("role") or t.get("type") for t in config.get("agent_types", [])],
            event_context="")
        scene_spec.attach_layout(spec, config.get("zones", []))
        config["scene_spec"] = spec
        log.info(f"SceneSpec: kits={spec['selected_kits']} template={spec['template'] or 'procedural'} "
                 f"smart_objects={len(spec['smart_objects'])}")
    except Exception as e:
        log.warning(f"SceneSpec build failed (renderer will fall back to theme): {e}")
        config["scene_spec"] = None

    total_agents = sum(t.get("count", 0) for t in config.get("agent_types", []))
    log.info(f"Scene generated: '{config['scene_name']}' with {len(config['zones'])} zones, {total_agents} agents")
    return config

# ── Prebuilt scenes ──────────────────────────────────────────────────────────────
# Hand-authored demo scenes (fixed floor-plan template + fixed-coord smart objects) that skip the LLM
# designer entirely, so they render instantly and identically every time. Lives next to this script.
# Isolated from generate_scene above: the "Tea House" button in the UI loads one of these; the free-text
# Generate button and its LLM path are untouched. To add a scene, drop a <key>.json here (same shape a
# generate_scene config has) and register the key in Unity's prebuilt list.
PREBUILT_DIR = Path(__file__).resolve().parent / "prebuilt_scenes"

def load_prebuilt_config(key: str) -> dict:
    safe = "".join(ch for ch in (key or "") if ch.isalnum() or ch in ("_", "-"))
    path = PREBUILT_DIR / f"{safe}.json"
    if not path.exists():
        raise FileNotFoundError(f"prebuilt scene '{key}' not found at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    config.setdefault("theme", "home")
    config.setdefault("behaviour_notes", "")
    config.setdefault("scene_spec", None)
    return config

# ── Helpers ────────────────────────────────────────────────────────────────────
def find_zone_by_keywords(zones, keywords):
    for z in zones:
        for k in keywords:
            if (k in z.get("id", "").lower()
                    or k in z.get("zone_type", "").lower()
                    or k in z.get("label", "").lower()):
                return z
    return None

def get_relationship(agent, other_id):
    for r in agent.relationships:
        if r.get("other_id") == other_id:
            return r
    return None

def are_friends(agent, other_id):
    r = get_relationship(agent, other_id)
    return r is not None and r.get("status") == "friend"

def are_avoiding(agent, other_id):
    r = get_relationship(agent, other_id)
    return r is not None and r.get("status") == "avoiding"

# ── Director tick ──────────────────────────────────────────────────────────────
# need field -> the affordance that satisfies it (for smart-object targeting from the exported scene graph)
_NEED_AFFORDANCE = {"bladder": "relieve", "thirst": "drink", "hunger": "eat",
                    "energy": "rest", "stress": "sit", "loneliness": "sit"}


def pick_smart_object(agent, affordance):
    """Nearest AVAILABLE, nav-reachable smart object offering `affordance` from the exported scene graph.
    This is what lets the behavior pipeline target a concrete smart_object_id (not just a zone)."""
    g = sim.scene_graph
    if not g:
        return None
    best, best_d = None, 1e18
    ax, ay = getattr(agent, "x", 0.0), getattr(agent, "y", 0.0)
    for o in g.get("smart_objects", []):
        if affordance not in (o.get("affordances") or []):
            continue
        if not o.get("nav_reachable", True):
            continue
        if (o.get("availability") or "available") != "available":
            continue
        ix, iy = o.get("ix", o.get("x", 0.0)), o.get("iy", o.get("y", 0.0))
        d = (ix - ax) ** 2 + (iy - ay) ** 2
        if d < best_d:
            best_d, best = d, o
    return best


def object_action(aid, agent, affordance, fallback_zone, reason):
    """Target the nearest matching smart object (with its interaction point) when the scene graph is
    available; otherwise fall back to the zone (keeps working without the export)."""
    o = pick_smart_object(agent, affordance)
    if o:
        return {"agent_id": aid, "action": "move_to_zone", "zone_id": o.get("zone_id"),
                "smart_object_id": o.get("smart_object_id"),
                "target_x": o.get("ix"), "target_y": o.get("iy"),
                "reason": f"{reason} -> {o.get('prop_id', 'object')}"}
    if fallback_zone:
        return {"agent_id": aid, "action": "move_to_zone", "zone_id": fallback_zone["id"], "reason": reason}
    return None


async def run_director_tick():
    client = anthropic.Anthropic(api_key=API_KEY)
    sim.pending_actions.clear()

    zones    = sim.zones
    agents   = list(sim.agents.values())
    zone_ids = [z["id"] for z in zones]

    if not agents:
        return [], list(sim.events)

    is_emergency = bool(sim.active_event)

    toilet_zone   = find_zone_by_keywords(zones, ["toilet","bathroom","restroom","wc"])
    drink_zone    = find_zone_by_keywords(zones, ["counter","bar","drink","water","cafe","fountain"])
    food_zone     = find_zone_by_keywords(zones, ["counter","food","cafe","kitchen","concession"])
    rest_zone     = find_zone_by_keywords(zones, ["lounge","rest","bench","sofa","shelter","seating"])
    activity_zone = find_zone_by_keywords(zones, ["activity","field","pitch","game","dance","stage"])
    exit_zone     = find_zone_by_keywords(zones, ["entrance","exit","gate","lobby"])
    vip_zone      = find_zone_by_keywords(zones, ["vip","premium","vip_lounge"])

    other_zones = [z for z in zones if z not in [toilet_zone, drink_zone, food_zone, rest_zone, activity_zone, exit_zone, vip_zone] if z]
    if not other_zones:
        other_zones = zones

    # ── Maslow-ordered rule-based fallback ────────────────────────────────────
    fallback_actions = []
    for i, agent in enumerate(agents):
        needs       = agent.needs
        aid         = agent.id
        name        = agent.name
        urgent_tier = int(needs.get("urgentTier", 1))
        action      = None

        if is_emergency and exit_zone:
            action = {"agent_id": aid, "action": "move_to_zone", "zone_id": exit_zone["id"],
                      "reason": f"{name} evacuating ({sim.active_event})"}

        # Tier 1 — Physiological (must be satisfied first). Target the concrete smart object serving the
        # need (thirst->drink machine, bladder->toilet, ...) from the exported scene graph; fall back to zone.
        elif urgent_tier == 1:
            if needs.get("bladder", 0) > 70:
                action = object_action(aid, agent, "relieve", toilet_zone, f"{name}: urgent bladder")
            elif needs.get("thirst", 0) > 70:
                action = object_action(aid, agent, "drink", drink_zone, f"{name}: thirsty")
            elif needs.get("hunger", 0) > 70:
                action = object_action(aid, agent, "eat", food_zone, f"{name}: hungry")
            elif needs.get("energy", 100) < 30:
                action = object_action(aid, agent, "rest", rest_zone, f"{name}: tired")

        # Tier 2 — Safety (stressed → open space or exit)
        elif urgent_tier == 2:
            if exit_zone:
                action = {"agent_id": aid, "action": "move_to_zone", "zone_id": exit_zone["id"],
                          "reason": f"{name}: stressed, needs space"}
            elif rest_zone:
                action = {"agent_id": aid, "action": "move_to_zone", "zone_id": rest_zone["id"],
                          "reason": f"{name}: stressed, seeking calm"}

        # Tier 3 — Belonging (seek friends or social zone)
        elif urgent_tier == 3:
            friend_id = None
            for r in agent.relationships:
                if r.get("status") == "friend":
                    friend_id = r.get("other_id")
                    break
            if friend_id and friend_id in sim.agents and activity_zone:
                action = {"agent_id": aid, "action": "group_move", "zone_id": activity_zone["id"],
                          "target_agent_id": friend_id,
                          "reason": f"{name}: wants to be with {sim.agents[friend_id].name}"}
            elif activity_zone:
                action = {"agent_id": aid, "action": "move_to_zone", "zone_id": activity_zone["id"],
                          "reason": f"{name}: feeling social"}

        # Tier 4 — Esteem (VIP/status zones)
        elif urgent_tier == 4:
            target_zone = vip_zone or activity_zone
            if target_zone:
                action = {"agent_id": aid, "action": "move_to_zone", "zone_id": target_zone["id"],
                          "reason": f"{name}: seeking recognition"}

        # Tier 5 — Curiosity / Engagement (explore)
        if action is None:
            zone = other_zones[i % len(other_zones)]
            action = {"agent_id": aid, "action": "move_to_zone", "zone_id": zone["id"],
                      "reason": f"{name}: exploring {zone.get('label', '')}"}

        fallback_actions.append(action)

    # During an active emergency the fallback already evacuates everyone to the exit.
    # Skip the (slow) Claude call entirely — evacuation is instant and deterministic.
    if is_emergency:
        sim.pending_actions = fallback_actions
        log.info(f"Emergency active ({sim.active_event}) — evacuating {len(fallback_actions)} agents (no Claude call)")
        events = list(sim.events)
        sim.events.clear()
        return sim.pending_actions, events

    # ── Build Claude prompt ───────────────────────────────────────────────────
    event_text = ""
    if is_emergency:
        event_text = f"ACTIVE EMERGENCY: {sim.active_event}. All agents must move to entrance/exit zone immediately."
    elif any(e["type"] == "all_clear" for e in sim.events):
        event_text = "ALL CLEAR: emergency is over. Reassign agents based on their current Maslow tier."

    if sim.standing_orders:
        order_lines = "\n".join(f"  {i}. {o}" for i, o in enumerate(sim.standing_orders, 1))
        event_text += (
            f"\n\n*** STANDING ORDERS (enforce EVERY tick until cancelled) ***\n{order_lines}\n"
            f"Keep the named agents at their ordered zones every tick, overriding their normal Maslow "
            f"needs. These stay in force until the user gives an all-clear. If two orders name the same "
            f"agent, the LATER one wins. Agents NOT named by any order behave normally.\n"
            f"EXCEPTION — a truly urgent physiological need may BRIEFLY break a standing order: if an "
            f"ordered agent has bladder>75, thirst>80, hunger>80, or energy<20, send them THIS tick to the "
            f"matching zone (toilet / water or counter / food or counter / rest area) to relieve it, then "
            f"return them to their ordered zone on the following ticks once the need drops. Only the single "
            f"most urgent need earns this exception; if no need is that severe, the agent STAYS put."
        )

    if sim.event_hint:
        event_text += (
            f"\n\n*** USER DIRECTIVE (HIGHEST PRIORITY, this tick) ***\n{sim.event_hint}\n"
            f"You MUST honor this directive this tick, even if it overrides normal Maslow ordering "
            f"for the agents it names. If it names specific agents, assign those agents the requested "
            f"action/zone explicitly."
        )
        sim.event_hint = ""

    zone_summary = "\n".join(
        f"- {z['id']} ({z.get('zone_type','?')}): {z.get('label','')}"
        for z in zones
    )

    agent_lines = []
    for a in agents:
        n           = a.needs
        urgent_tier = int(n.get("urgentTier", 1))
        tier_label  = {1: "PHYSIOLOGICAL", 2: "SAFETY", 3: "BELONGING", 4: "ESTEEM", 5: "CURIOSITY"}.get(urgent_tier, "?")

        # Relationship summary for this agent
        rel_parts = []
        for r in a.relationships:
            if r.get("familiarity", 0) >= 10:
                other_name = sim.agents[r["other_id"]].name if r["other_id"] in sim.agents else r["other_id"]
                rel_parts.append(f"{other_name}({r.get('status','?')},fam={r.get('familiarity',0):.0f})")
        rel_str = ", ".join(rel_parts) if rel_parts else "no relationships yet"

        line = (
            f"- {a.name} ({a.agent_type}, {a.personality}) urgentTier={urgent_tier}[{tier_label}] | "
            f"hunger={n.get('hunger',0):.0f} thirst={n.get('thirst',0):.0f} "
            f"bladder={n.get('bladder',0):.0f} energy={n.get('energy',100):.0f} | "
            f"stress={n.get('stress',0):.0f} loneliness={n.get('loneliness',0):.0f} "
            f"groupAffinity={n.get('groupAffinity',0):.0f} status={n.get('status',0):.0f} "
            f"curiosity={n.get('curiosity',0):.0f} | "
            f"zone={a.current_zone} | relationships: {rel_str}"
        )
        agent_lines.append(line)

    agent_summary = "\n".join(agent_lines)

    prompt = (
        f"Scene: {sim.scene_name}. {sim.description}\n"
        f"{event_text}\n\n"
        f"Zones:\n{zone_summary}\n\n"
        f"Agents (urgentTier tells you the HIGHEST Maslow priority to address):\n{agent_summary}\n\n"
        f"Default actions (rule-based, Maslow-ordered). Improve them: add group moves for friends, "
        f"better zone matches, start conversations for acquaintances/friends with high loneliness. "
        f"MUST return JSON array covering ALL {len(agents)} agents.\n\n"
        f"Default actions:\n{json.dumps(fallback_actions, indent=2)}\n\n"
        f'Output: [{{"agent_id":"agent_0","action":"move_to_zone","zone_id":"counter","reason":"thirsty"}}]'
    )

    try:
        response = await asyncio.to_thread(
            client.messages.create,
            model=MODEL, max_tokens=8192,   # a per-agent action list for an 80-90 agent crowd overruns 4000
            system=DIRECTOR_SYSTEM,          # tokens and truncates every tick -> silently always uses fallback
            messages=[{"role": "user", "content": prompt}]
        )
        raw   = response.content[0].text.strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            actions = json.loads(raw[start:end])
            valid   = []
            for a in actions:
                act_type = a.get("action")
                if act_type == "start_conversation":
                    if a.get("target_agent_id") in sim.agents:
                        # Don't pair avoiding agents
                        requester = sim.agents.get(a["agent_id"])
                        if requester and not are_avoiding(requester, a["target_agent_id"]):
                            valid.append(a)
                elif act_type in ("move_to_zone", "group_move"):
                    if a.get("zone_id") in zone_ids:
                        valid.append(a)
                    else:
                        fb = next((f for f in fallback_actions if f["agent_id"] == a.get("agent_id")), None)
                        if fb: valid.append(fb)
                elif act_type in ("idle", "rest"):
                    valid.append(a)

            sim.pending_actions = valid if valid else fallback_actions
            log.info(f"Claude assigned {len(sim.pending_actions)} actions")
        else:
            sim.pending_actions = fallback_actions
            log.info(f"Using {len(fallback_actions)} fallback actions")
    except Exception as e:
        log.error(f"Claude error: {e} — using fallback")
        sim.pending_actions = fallback_actions

    for a in sim.pending_actions:
        aname = sim.agents[a["agent_id"]].name if a["agent_id"] in sim.agents else "?"
        log.info(f"  → {aname}: {a.get('reason','')}")

    events = list(sim.events)
    sim.events.clear()
    return sim.pending_actions, events

# ── Free-text event interpretation ─────────────────────────────────────────────
async def interpret_event(description: str, scene_name: str) -> dict:
    client = anthropic.Anthropic(api_key=API_KEY)

    system = """You interpret crowd simulation events from natural language typed by a human director.
Return ONLY valid JSON (no markdown) with this structure:
{
  "event_type": "short_snake_case_name",
  "display_name": "Human-readable name",
  "need_deltas": {
    "stress": number,        // positive = increases stress
    "curiosity": number,     // negative = satisfies curiosity
    "loneliness": number,
    "groupAffinity": number,
    "energy": number,
    "hunger": number,
    "thirst": number,
    "status": number
  },
  "is_emergency": boolean,
  "clears_emergency": boolean,
  "is_standing_order": boolean,
  "directive": "An imperative instruction for the behavioral director, naming specific agents/zones when the user did. e.g. 'Move Robin to the exit zone now.' or 'Everyone evacuate to the entrance.'",
  "affected_agents": "all",
  "behavior_hint": "One sentence describing what agents should do next tick"
}

Rules:
- need_deltas values are CHANGES (positive = increase, negative = decrease), range -50 to +50
- For exciting events (celebrity, performance): curiosity and social deltas are negative (satisfying)
- For alarming events (fight, accident): stress rises, energy may drop
- For pleasant events (music, food arrival): hunger/thirst/loneliness deltas are negative (satisfying)
- is_emergency = true ONLY for events that force everyone to leave/evacuate (fire, closing time, gas leak, lockdown). Then the director sends ALL agents to the exit.
- is_emergency = false for everything else, including commands aimed at specific agents.
- clears_emergency = true ONLY when the user signals the danger is OVER and normal life resumes (e.g. "all clear", "it's safe now", "emergency over", "everyone can relax", "false alarm", "go back inside"). Otherwise false.
- A message can set clears_emergency=true AND still give a directive (e.g. "all clear, everyone go to the center" → clears_emergency=true, directive="Everyone gather at the center seating.").
- is_standing_order = true for ANY command telling one or more agents to GO TO or STAY AT a place — a movement order that should persist until cancelled. This includes:
    * individuals by name: "Sam go to the exit", "keep Logan at the counter"
    * groups: "all VIPs go to the booth", "the tourists wait by the entrance"
    * everyone: "there's a party in the living room, everybody go there", "everyone gather at the stage"
  Write the directive so it names the destination zone clearly. The targeted agents stay there every tick until an all-clear.
- is_standing_order = false ONLY for ambiance/mood events that do NOT relocate anyone — they just change needs (e.g. "music starts", "free cake at the counter", "a celebrity walks in") — and for emergencies/all-clears.
- "directive" is the most important field: rewrite the user's request as a clear command the director MUST obey, preserving any agent names and destinations the user mentioned.
- behavior_hint and directive will both be shown to the director in the next tick to guide actions."""

    prompt = (
        f'Scene: "{scene_name}". A human director typed this event/command: "{description}".\n'
        f'Interpret it and return the JSON. Honor any specific agent names or destinations they mentioned.'
    )

    try:
        response = await asyncio.to_thread(
            client.messages.create,
            model=MODEL, max_tokens=500, system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        raw     = response.content[0].text.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        s = cleaned.find("{"); e = cleaned.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(cleaned[s:e])
    except Exception as ex:
        log.error(f"Event interpretation error: {ex}")

    # Fallback: treat the raw text as a directive
    return {
        "event_type": "announcement",
        "display_name": description[:40],
        "need_deltas": {"curiosity": -15, "loneliness": -5},
        "is_emergency": False,
        "clears_emergency": False,
        "is_standing_order": True,
        "directive": description,
        "affected_agents": "all",
        "behavior_hint": description
    }

# ── Free-text event -> STRUCTURED GRAPH PATCH (the user-interruption contribution) ──
# The DSAG director does NOT instruct agents one by one. It turns the director's sentence into a
# small set of typed ops that reshape the world (which zones pull, which affordances exist, how
# needs drift); the whole crowd then redirects by reading the patched graph — one LLM call, N agents.
async def interpret_event_patch(description: str, scene) -> dict:
    client = anthropic.Anthropic(api_key=API_KEY)
    zone_list = ", ".join(f'"{z.id}"({z.zone_type})' for z in scene.zones.values()) or "(none)"
    agent_list = ", ".join(f'"{a.name}"({a.role})' for a in scene.agents.values()) or "(none)"

    # Real placed smart objects — id, type, zone, and the actions each affords — so remove_object /
    # affordance ops reference an object that ACTUALLY EXISTS (a prop_id like `cafe__drink`, not an invented
    # 'tray of glasses'). Only lists objects currently in service (a removed one isn't offered).
    def _obj_desc(o):
        affs = ",".join(sorted({a.action for a in getattr(o, "affordances", [])})) or "-"
        return f'"{o.id}"({getattr(o, "object_type", "prop")} @{getattr(o, "zone_id", "?")}: {affs})'
    object_list = "; ".join(_obj_desc(o) for o in scene.objects.values()
                            if not getattr(o, "removed", False)) or "(none)"

    system = (
        "You convert a human director's free-text interruption of a live crowd simulation into a "
        "STRUCTURED GRAPH PATCH: a few typed ops that reshape the WORLD so the crowd redirects itself. "
        "For CROWD-WIDE events reshape the world; for a SPECIFIC NAMED person (or two people interacting) "
        "use `agent_directive` with their exact name. Return ONLY JSON.\n\n"
        "Op kinds (use 1-6 relevant ones):\n"
        '  {"op":"zone_attraction","zone":"<zone keyword or id>","delta":0..90}   pull agents toward a zone\n'
        '  {"op":"enable_affordance","zone":"<zone>","action":"<affordance>"}      the zone now offers this action\n'
        '  {"op":"disable_affordance","action":"<affordance>","zone":"<zone?>"}    stop offering it (omit zone = everywhere)\n'
        '  {"op":"need_rate","need":"<need>","delta":-8..8,"role":"<role?>"}       change per-tick need DRIFT\n'
        '  {"op":"need_shift","need":"<need>","delta":-40..40,"role":"<role?>"}    one-shot need nudge right now\n'
        '  {"op":"role_priority","role":"<role>","zone":"<zone>","weight":0..60}   bias one role toward a zone\n'
        '  {"op":"agent_directive","agent":"<name>","zone":"<zone>"}               send ONE named person to a zone\n'
        '  {"op":"agent_directive","agent":"<name>","target_agent":"<name>"}       send ONE named person to ANOTHER (to meet/talk)\n'
        '  {"op":"agent_directive","agent":"<name>","action":"<affordance>"}       make ONE named person DO a specific thing NOW (resolved to a real reachable object) — "Alex wants to sit down" -> {agent:"Alex",action:"sit"}\n'
        '  {"op":"agent_leave","agent":"<name>"}                                  send ONE named person OUT of the building (they walk to the exit and leave/disappear)\n'
        '  {"op":"agent_leave","role":"<group/type>"}                             send a SUBSET out by group/type — role="family" (the family group), "tourist", "student"… ONLY those people leave, NOT everyone\n'
        "  -- GRAPH-EDIT ops (change the scene STRUCTURE — use these for 'X appears/arrives' or 'Y is removed/broken'):\n"
        '  {"op":"remove_object","object":"<id or keyword>","zone":"<zone?>"}      a smart object goes OUT OF SERVICE (a glass broke, a cup cleared) — prefer a real id/type from the object list below; add `zone` to pin WHERE\n'
        '  {"op":"spawn_agent","role":"<role>","count":1..8,"zone":"<zone>"}      NEW people ENTER (firefighters/medics/police arrive, more customers show up) at that zone\n'
        '  {"op":"role_directive","role":"<role>","zone":"<zone>"}                send EVERY agent of a role to a zone (responders → the hazard); overrides evacuation for them\n\n'
        "GRAPH-EDIT guidance: 'a glass broke / spill cleared' -> remove_object. 'firefighters/paramedics/police "
        "arrive' -> spawn_agent (role=firefighter|medic|security, count 2-4, zone=the entrance/access) AND a "
        "role_directive sending that role to the emergency zone. A fire is typically: intent=evacuation (crowd "
        "leaves) + spawn_agent firefighters + role_directive firefighter->hazard zone (they go IN while the "
        "crowd goes OUT). Only spawn/remove when the text actually says something appears or is removed.\n"
        "needs: hunger thirst bladder energy stress loneliness groupAffinity taskProgress curiosity status\n"
        "affordances: drink eat sit dance observe relieve talk_to_staff order leave\n"
        "roles (optional scope): staff, casual_young, tourist, student, elderly, business_person, vip; omit = everyone\n"
        f"zones in THIS scene (use these ids/keywords in `zone`): {zone_list}\n"
        f"people in THIS scene (use these exact names in `agent`/`target_agent`): {agent_list}\n"
        f"smart objects in THIS scene (use a real id/type in `remove_object.object`): {object_list}\n\n"
        "Return exactly:\n"
        '{"intent":"<canonical>","display_name":"Human Name","ttl":25,'
        '"behavior_hint":"one sentence","ops":[ ... ]}\n\n'
        "`intent` MUST be exactly one of: end_of_day, evacuation, hazard, all_clear, object_disable, "
        "object_enable, block_edge, agent_directive, group_directive, announcement.\n"
        "Guidance:\n"
        "- 'X goes home' / 'X wants to go home' / 'X leaves' / 'kick X out' / 'send X home' -> ONE "
        "{op:agent_leave, agent:X} PER named person (they walk to the exit and leave the building). Use "
        "ttl >= 20 so they have time to reach the exit.\n"
        "- A GROUP/TYPE going home ('the family goes home', 'the family is tired, they leave', 'the tourists "
        "head out') -> ONE {op:agent_leave, role:<group/type>} (role='family'|'tourist'|…). intent MUST be "
        "group_directive, NOT end_of_day — ONLY that subset leaves, everyone else stays. Reserve end_of_day / "
        "evacuation for when EVERYONE must leave (closing time, fire). If in doubt, prefer the scoped agent_leave.\n"
        "- a SPECIFIC named person ('Drew go to the bar', 'the waiter clean table 2') -> ONE agent_directive "
        "with that person's exact name and the zone. 'X talk to Y' / 'X wants to talk to Y' / 'X and Y "
        "should meet' -> a SINGLE {agent:X,target_agent:Y}. The FIRST person named is `agent` (the one who "
        "WALKS over); the second is `target_agent` (stays put). Do NOT also emit {agent:Y,target_agent:X} — "
        "sending both makes them swap places and never meet. Match each typed name to the closest person.\n"
        "- a named person told to DO A SPECIFIC THING right now, with no zone/other-person named ('Alex wants "
        "to sit down', 'Sam is hungry, get them something to eat', 'Morgan should use the restroom') -> ONE "
        "agent_directive {agent:X, action:<affordance>} — pick the affordance from the list below (sit, eat, "
        "drink, relieve, rest, work, observe, talk_to_staff). The engine finds and reserves a real reachable "
        "object for them; you never invent zone/object ids for this form.\n"
        "- `agent_directive` ALWAYS requires `agent` (the exact name) — never emit one with only a zone/action "
        "and no name; if you cannot identify WHO from the text, do not emit an agent_directive at all.\n"
        "- party/gathering at X -> zone_attraction X (60-85), enable dance/drink at X, need_rate stress -4, "
        "need_shift loneliness +25 (so they WANT to gather).\n"
        "- ANY 'go to X' / 'come to X' / 'gather/move/head to X' / 'X is now open, come over' / an OFFERING "
        "at X ('food offering in the dining area', 'free samples at the counter', 'show starting on stage') "
        "is a POSITIVE relocation directive: you MUST emit zone_attraction X with a STRONG delta (70-90) so "
        "the crowd actually walks there. This is the main way the user redirects the crowd — never leave it "
        "as a mere announcement with no attraction.\n"
        "- Choose `intent` by MEANING (the server derives priority + is_emergency from it): "
        "end_of_day = venue closing / everyone head home (CALM, not an emergency); "
        "evacuation = danger, everyone out NOW (fire/gas/alarm/lockdown — an EMERGENCY); "
        "hazard = a localized danger to AVOID at one spot (spill/fight/glass); "
        "all_clear = danger over, resume normal; "
        "object_disable = a facility is unavailable ('counter closed', 'restroom out of order'); "
        "object_enable = available again; block_edge = a passage/door/exit is physically sealed "
        "('close the emergency exit'); agent_directive = one or two NAMED people; group_directive = a "
        "named group; announcement = ambient/mood. 'everyone go home' is end_of_day (calm), NOT "
        "evacuation. A global leave (end_of_day OR evacuation) makes the engine route EVERY agent out.\n"
        "- CLOSED / HAZARD / AVOID at X (restroom closed, spill, fight, rain outside): you MUST push the "
        "crowd AWAY from X — set zone_attraction X to a strong NEGATIVE delta (-40..-70, repulsion) AND "
        "disable the affordance at X, and usually attract a safe alternative zone. Do NOT raise the need "
        "the closed facility served (a closed restroom should not increase bladder).\n"
        "- calm/ambient mood (music, free cake) -> mostly need_rate/need_shift, small or no attraction.\n"
        "- pick the `zone` whose id/type best matches what the user named; ttl ~ how long it should last."
    )
    prompt = f'The director typed: "{description}". Return the graph patch JSON.'
    try:
        resp = await asyncio.to_thread(
            client.messages.create, model=MODEL, max_tokens=600, system=system,
            messages=[{"role": "user", "content": prompt}])
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s != -1 and e > s:
            return _ground_intent(json.loads(raw[s:e]), description)
    except Exception as ex:
        log.error(f"Patch interpretation error: {ex}")
    # Fallback: still resolve the intent from the raw text (MiniLM / safety net), gentle attention pull
    return _ground_intent({"display_name": description[:40], "ttl": 20, "behavior_hint": description,
                           "ops": [{"op": "need_shift", "need": "curiosity", "delta": -15}]}, description)


import re as _re

# Subjects that mean the WHOLE crowd -> a global end_of_day, NOT a scoped agent_leave (let intent handle it).
_LEAVE_GLOBAL = {"everyone", "everybody", "all", "the crowd", "people", "all of them", "everyone here"}
# Group/type words -> emit as role= (agents_leaving resolves to the social group / agent_type). Else a NAME.
_LEAVE_GROUPS = {"family", "families", "friend", "friends", "group", "groups", "couple", "couples", "party",
                 "tourist", "tourists", "student", "students", "staff", "kid", "kids", "child", "children",
                 "guest", "guests", "team", "regulars", "regular", "loners", "loner"}
# Leave-command patterns: capture WHO leaves. Ordered specific -> general.
_LEAVE_PATS = [
    _re.compile(r"\b(?:send|kick|get|make|show)\s+(?P<who>.+?)\s+(?:home|out|outside|to leave|away)\b"),
    _re.compile(r"\b(?P<who>.+?)\s+(?:wants?|need(?:s)?|would like|decide[sd]?)\s+to\s+(?:go\s+home|leave|head\s+out|go)\b"),
    _re.compile(r"\b(?P<who>.+?)\s+(?:go(?:es|ing)?|head(?:s|ing)?)\s+(?:home|out|outside)\b"),
    _re.compile(r"\b(?P<who>.+?)\s+(?:is|are|'re|has|have|'ve)\s+(?:leaving|going home|heading (?:home|out))\b"),
    _re.compile(r"\b(?P<who>.+?)\s+leaves?\b"),
]

def _leave_fallback_ops(description: str) -> list:
    """Deterministic safety net: turn an obvious 'go home / leave' command into agent_leave op(s) when the LLM
    misclassifies it (the reported bug: 'morgan wants to go home' / 'friends leave' did nothing). The SUBJECT
    is grounded downstream by dsag.patch.agents_leaving (name vs social group / type). Returns [] for a global
    'everyone leaves' (that stays end_of_day) or when no leave command is detected."""
    t = " " + (description or "").lower().strip().strip(".!") + " "
    who = None
    for pat in _LEAVE_PATS:
        m = pat.search(t)
        if m:
            who = m.group("who").strip()
            break
    if not who:
        return []
    who = _re.sub(r"^(the|a|an|my|our|those|these|some|that|this)\s+", "", who).strip(" ,.'\"")
    if not who or len(who.split()) > 4 or who in _LEAVE_GLOBAL:
        return []
    head = who.split()[0]
    if who in _LEAVE_GROUPS or head in _LEAVE_GROUPS or who.rstrip("s") in _LEAVE_GROUPS:
        return [{"op": "agent_leave", "role": who}]
    return [{"op": "agent_leave", "agent": who}]


def _ground_intent(result: dict, description: str) -> dict:
    """Ground a director event to a canonical intent (LLM `intent` first, else MiniLM semantic, else a
    tiny safety net) and set the INDEPENDENT fields event_type / global_directive / is_emergency /
    priority / clears_emergency. No large keyword lists — the semantic stage does the work."""
    canon = event_intent.resolve(description, llm_intent=result.get("intent"))
    # DETERMINISTIC LEAVE SAFETY NET: if the text is clearly a scoped 'go home' but the LLM produced no
    # agent_leave op, inject one so the person/group reliably walks out (independent of LLM classification).
    if not canon["is_emergency"] and not any(o.get("op") == "agent_leave" for o in result.get("ops", [])):
        fb = _leave_fallback_ops(description)
        if fb:
            result.setdefault("ops", []).extend(fb)
            log.info(f"[intent]   leave-fallback injected {fb} (LLM produced no agent_leave)")
    result["intent"] = canon["intent"]
    result["event_type"] = canon["event_type"]
    result["global_directive"] = canon["global_directive"]
    result["is_emergency"] = canon["is_emergency"]
    result["priority"] = canon["priority"]
    result["clears_emergency"] = canon["clears_emergency"]
    # SCOPED LEAVE (issue 2): if the patch enumerates WHO leaves (agent_leave ops, by name or group/role),
    # it is a SUBSET — never a global end_of_day. Otherwise a 'the family goes home' gets misread as
    # 'everyone go home' and empties the whole venue. Only a leave with NO agent_leave ops stays global.
    if not canon["is_emergency"] and any(o.get("op") == "agent_leave" for o in result.get("ops", [])):
        result["global_directive"] = None
        if result["intent"] == "end_of_day":
            result["intent"] = "group_directive"
        log.info("[intent]   scoped agent_leave present -> subset leave (NOT global end_of_day)")
    result.setdefault("display_name", description[:40])
    log.info(f"[intent] '{description[:50]}' -> {canon['intent']} "
             f"(emergency={canon['is_emergency']}, global={canon['global_directive']}, src={canon['source']})")
    return result


# Keyword gates for the typed-event hooks. Plain `k in text` substring matching silently
# misfires: "travel" contains "rave" (party), "message" contains "mess" (spill), "presentation"
# contains "present" (gift), "abandon" contains "band" (gig). Anchor every term on a word
# boundary. A few terms are deliberate PREFIXES ("celebrat" -> celebrating/celebration), so they
# alone may carry a suffix.
_KW_PREFIX = {"celebrat", "danc", "fest", "spillag"}
_kw_rx: dict = {}


def _kw_hit(blob: str, words) -> bool:
    rx = _kw_rx.get(words)
    if rx is None:
        rx = _kw_rx[words] = re.compile(
            r"\b(?:" + "|".join(re.escape(w) + (r"\w*" if w in _KW_PREFIX else r"\b")
                                for w in words) + ")")
    return rx.search(blob) is not None


async def process_inbox_dsag(scene):
    """DSAG path: drain EventLog/inbox, turn each dropped event into a ScenePatch, and apply it to
    the live scene so the crowd redirects this tick. The all-clear sentinel drops all patches."""
    inbox, processed = EVENTLOG / "inbox", EVENTLOG / "processed"
    for f in sorted(p for p in inbox.glob("*") if p.suffix.lower() in (".txt", ".json", ".md")):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore").strip().lstrip("﻿").strip()
        except Exception:
            text = ""
        if text == ALL_CLEAR_SENTINEL:
            scene.active_patches = []
            scene.zone_decor = {}                          # party over -> take the balloons down
            log.info(f"[inbox] all-clear -> cleared active patches ('{f.name}')")
        elif text:
            pd = await interpret_event_patch(text, scene)
            if pd.get("clears_emergency") or pd.get("intent") == "all_clear":
                scene.active_patches = []                 # free-text "all clear" resolves here too
                scene.zone_decor = {}                     # and take the party balloons down
                log.info(f"[inbox] '{text[:50]}' -> all_clear: cleared active patches")
            else:
                patch = ScenePatch.from_dict(pd)
                # MALFORMED OPS: reject and LOG loudly instead of silently doing nothing (the reported bug —
                # an agent_directive with no `agent` looked like a normal patch in the log but matched nobody).
                for raw, reason in patch.rejected_ops:
                    log.warning(f"[inbox]    REJECTED malformed op {raw} — {reason}")
                if patch.is_valid() and patch.ops:
                    scene.apply_patch(patch)
                    log.info(f"[inbox] '{text[:50]}' -> {pd.get('intent','?')} patch "
                             f"'{patch.display_name}' ({len(patch.ops)} ops, ttl={patch.ttl}, "
                             f"emergency={patch.is_emergency}, global={patch.global_directive})")
                    # full ops (so redirection can be verified: a 'go to X'/'party at X' MUST include a
                    # positive zone_attraction, else there is nothing to redirect toward). agent/target_agent
                    # included — their absence used to be invisible in this log, hiding a malformed directive.
                    for o in patch.ops:
                        log.info(f"[inbox]    op: {o.op} agent={o.agent!r} target_agent={o.target_agent!r} "
                                 f"zone={o.zone!r} zone_function={o.zone_function!r} "
                                 f"action={o.action!r} object={o.object!r} role={o.role!r} count={o.count} delta={o.delta}")
                    # MATCH SURFACE for the event hooks below. The raw typed text alone is too thin: a live test
                    # typed "coffee machine outage", which never says "power", so it missed the outage gate and
                    # the event silently did nothing. Match the INTERPRETED event too — display_name and
                    # behavior_hint restate it in fuller language — and keep the intent as a separate signal,
                    # which beats any keyword (that outage came back as intent=object_disable).
                    _ev = " ".join(str(x) for x in (text, pd.get("display_name") or "",
                                                    pd.get("behavior_hint") or "")).lower()
                    _intent = str(pd.get("intent") or pd.get("event_type") or "").lower()
                    # GRAPH-EDIT ops: remove_object marks objects UNAVAILABLE (reversible; the GNN's next-tick
                    # graph excludes them) and spawn_agent adds agents NOW. SPAWNS are mirrored to Unity via the
                    # scene_mutation despawn/spawn channel; REMOVALS are conveyed reversibly via object_states
                    # (`available:false` -> Unity hides, `true` on restore -> shows), so they are NOT queued into
                    # the destructive despawn path.
                    mut = apply_structural_ops(scene, patch)
                    if mut["spawned_agents"]:
                        q = getattr(scene, "pending_mutations", None) or {"removed_objects": [], "spawned_agents": []}
                        q["spawned_agents"].extend(mut["spawned_agents"])
                        scene.pending_mutations = q
                    if mut["removed_objects"] or mut["spawned_agents"]:
                        log.info(f"[inbox]    graph-edit: unavailable {mut['removed_objects']} "
                                 f"spawned {[a['id']+':'+a['role'] for a in mut['spawned_agents']]}")
                    from ecgp.runtime.live_bridge import _olog, _oname
                    for oid in mut["removed_objects"]:                 # broken/dropped -> amber object log
                        ob = scene.objects.get(oid)
                        _olog(scene, f"the {_oname(ob, oid)} was broken — out of service")
                    for a in mut["spawned_agents"]:
                        _olog(scene, f"{a['role']} arrived on the scene")
                    # Visibility: a remove_object that matched NOTHING is a silent no-op otherwise — surface it
                    # so a demo miss (the LLM named a prop the scene doesn't contain, e.g. 'tray of glasses' in
                    # a fixture-only cafe) is obvious, and show what WAS available to remove.
                    wanted_removes = [o.object for o in patch.ops if o.op == "remove_object" and o.object]
                    if wanted_removes and not mut["removed_objects"]:
                        avail = [f"{getattr(ob,'id','?')}:{getattr(ob,'object_type','?')}"
                                 for ob in list(scene.objects.values())[:12]]
                        log.warning(f"[inbox]    remove_object matched NOTHING for {wanted_removes} — "
                                    f"scene has no such prop; available objects: {avail}")
                    # STAGE 2 trigger: 'someone spilled/dropped a drink, there's a mess' -> spawn a spill in the
                    # named zone; the tick loop then dispatches staff to clean it (restriction until done).
                    if _kw_hit(_ev, ("spill", "spilled", "spillage", "drop", "dropped",
                                     "dirty", "mess", "messy", "clean up", "knocked",
                                     "puddle", "stain", "broke a", "made a mess")):
                        zid = next((o.zone for o in patch.ops if o.zone and o.op in
                                    ("zone_attraction", "disable_affordance")), None)
                        zid = _match_zone_id(scene, zid) or (next(iter(scene.zones)) if scene.zones else None)
                        if zid:
                            from ecgp.runtime.spill import spawn_spill
                            sid = spawn_spill(scene, zid)
                            log.info(f"[inbox]    SPILL spawned {sid} in zone {zid} -> staff will clean it")
                    # PARTY / CELEBRATION -> hang balloons over the zone until the patch's ttl runs out (or an
                    # all-clear). Same zone-from-ops extraction as the spill trigger (the party zone is the one
                    # the patch pulls the crowd TOWARD — a positive zone_attraction).
                    if _kw_hit(_ev, ("party", "celebrat", "birthday", "festival", "fiesta",
                                     "disco", "rave", "danc", "wedding", "anniversary",
                                     "gathering", "gala", "carnival", "fest")):
                        zid = next((o.zone for o in patch.ops
                                    if o.zone and o.op == "zone_attraction" and (o.delta or 0) > 0), None)
                        zid = _match_zone_id(scene, zid) or (next(iter(scene.zones)) if scene.zones else None)
                        if zid:
                            decor = getattr(scene, "zone_decor", None)
                            if decor is None:
                                decor = {}; scene.zone_decor = decor
                            _party_s = max(20.0, float(patch.ttl or 60))
                            decor[zid] = time.monotonic() + _party_s
                            # SOMEONE TAKES THE MIC. If the party zone owns a performance fixture (the
                            # event hall's `so_mic` — a `talk` object), pull attention to it so a visitor
                            # actually goes and uses it, and float music notes above it. Both die with the
                            # party. Purely additive: a zone with no such fixture just gets the balloons.
                            _mic = next((o for o in scene.objects.values()
                                         if getattr(o, "zone_id", None) == zid
                                         and not getattr(o, "removed", False)
                                         and any(getattr(a, "action", None) == "talk"
                                                 for a in getattr(o, "affordances", []))), None)
                            if _mic is not None:
                                from ecgp.runtime.spill import _queue as _mutq
                                from dsag.patch import PatchOp as _POp
                                patch.ops.append(_POp(op="object_attraction", object=_mic.id, delta=70.0))
                                _dec = getattr(scene, "_decor_props", None)
                                if _dec is None:
                                    _dec = scene._decor_props = {}
                                _nid = f"party_notes_{len(_dec)}"
                                _mx, _my = getattr(_mic, "pos", (0.0, 0.0))
                                _mutq(scene, "spawned_objects",
                                      {"id": _nid, "object_type": "music_notes", "zone_id": zid,
                                       "x": float(_mx) + 0.9, "y": float(_my) + 1.6})
                                _dec[_nid] = time.monotonic() + _party_s
                                log.info(f"[inbox]    PARTY -> {_mic.id} on the mic + notes")
                            log.info(f"[inbox]    PARTY -> balloons on zone {zid} for ~{int(patch.ttl or 60)}s")
                    # FIRE -> a visible blaze in the zone while the emergency runs, auto-resolved after
                    # ~12s: the fire prop despawns, the emergency patch is dropped (all-clear), responders
                    # walk out, and an arrival burst repopulates the venue. Staff HOLD at reception via
                    # role_directive instead of evacuating — emergency evac despawns agents that cross the
                    # exit portal, and a venue that permanently loses its staff to every fire is broken.
                    # Gated on is_emergency so "firefighters arrive" alone never spawns a blaze.
                    if patch.is_emergency and _kw_hit(_ev,
                                ("fire", "burning", "blaze", "flames", "smoke")):
                        # The blaze belongs in the HAZARD zone, which the patch marks with a NEGATIVE
                        # attraction (the positive one is the evacuation target — placing the fire there
                        # would be exactly inverted). Fall back to a disabled zone, then the responders'
                        # destination, and only then to op order.
                        _neg = sorted((o for o in patch.ops
                                       if o.op == "zone_attraction" and o.zone and (o.delta or 0) < 0),
                                      key=lambda o: (o.delta or 0))
                        zid = (next((o.zone for o in _neg), None)
                               or next((o.zone for o in patch.ops
                                        if o.zone and o.op == "disable_affordance"), None)
                               or next((o.zone for o in patch.ops
                                        if o.zone and o.op == "role_directive"), None)
                               or next((o.zone for o in patch.ops if o.zone), None))
                        zid = _match_zone_id(scene, zid) if zid else None
                        if zid is None:                       # fall back to naming the zone in the text
                            low = _ev
                            zid = next((z for z in scene.zones
                                        if z.lower() in low or z.replace("_", " ").lower() in low), None)
                        zid = zid or (next(iter(scene.zones)) if scene.zones else None)
                        if zid:
                            from ecgp.runtime.spill import _queue as _mutq
                            fires = getattr(scene, "_fire_props", None)
                            if fires is None:
                                fires = scene._fire_props = {}
                            fid = f"fire_{len(fires)}"
                            fcx, fcy = scene.zones[zid].center
                            _mutq(scene, "spawned_objects", {"id": fid, "object_type": "fire",
                                                             "zone_id": zid, "x": float(fcx), "y": float(fcy)})
                            fires[fid] = {"zone": zid, "until": time.monotonic() + 12.0}
                            # keep the staff on station while the visitors evacuate
                            from dsag.patch import PatchOp as _POp
                            recep = next((z for z in scene.zones if "recep" in z.lower()), None)
                            if recep:
                                staff_roles = sorted({(getattr(a, "role", "") or "").lower()
                                                      for a in scene.agents.values()
                                                      if (getattr(a, "role", "") or "").lower()
                                                      in ("barista", "waiter", "bartender", "cashier", "staff")})
                                for r in staff_roles:
                                    patch.ops.append(_POp(op="role_directive", role=r, zone=recep))
                            log.info(f"[inbox]    FIRE -> blaze {fid} in {zid}, auto-clear in ~12s "
                                     f"(staff hold at {recep})")
                    # FOOD EVENT -> pop-up food carts customers can actually EAT at, for the patch's ttl.
                    # Real SmartObjects (provider role, eat affordance) so the policy targets them like any
                    # counter, plus an object_attraction so the event pulls attention. NO stock attr on
                    # purpose: stock 0 would trigger the restock machinery and close the zone mid-event.
                    if _kw_hit(_ev, ("free food", "food party", "tasting", "taste event",
                                     "buffet", "food festival", "food stall", "food truck",
                                     "food giveaway", "free snack")):
                        zid = next((o.zone for o in patch.ops
                                    if o.zone and o.op == "zone_attraction" and (o.delta or 0) > 0), None)
                        zid = _match_zone_id(scene, zid) if zid else None
                        if zid is None:                       # default to wherever the food already is
                            zid = next((o.zone_id for o in scene.objects.values()
                                        if any(getattr(a, "action", None) == "eat"
                                               for a in getattr(o, "affordances", []))), None)
                        if zid and zid in scene.zones:
                            from ecgp.runtime.spill import _queue as _mutq
                            from dsag.smart_object import SmartObject, Affordance
                            from dsag.patch import PatchOp as _POp
                            pops = getattr(scene, "_popups", None)
                            if pops is None:
                                pops = scene._popups = {}
                            pcx, pcy = scene.zones[zid].center
                            until = time.monotonic() + min(120.0, max(40.0, float(patch.ttl or 25) * 5.0))
                            for i, otype in enumerate(("food_cart_a", "food_cart_b")):
                                oid = f"popup_cart_{len(pops)}"
                                obj = SmartObject(id=oid, object_type=otype, zone_id=zid,
                                                  states=["available"], state="available",
                                                  affordances=[Affordance("eat",
                                                      need_effects={"hunger": -45, "status": -5})],
                                                  pos=(float(pcx) + (i * 2 - 1) * 1.0, float(pcy) - 0.6))
                                obj.capacity = 3
                                obj.functional_role = "provider"
                                obj.display_name = "Food Cart"
                                scene.objects[oid] = obj
                                pops[oid] = until
                                _mutq(scene, "spawned_objects", {"id": oid, "object_type": otype,
                                                                 "zone_id": zid,
                                                                 "x": obj.pos[0], "y": obj.pos[1]})
                                patch.ops.append(_POp(op="object_attraction", object=oid, delta=60.0))
                            log.info(f"[inbox]    FOOD EVENT -> 2 pop-up carts in {zid} for "
                                     f"~{int(until - time.monotonic())}s")
                    # Shared helper for the people-spawning events below. spawn_agent OPS cannot be used
                    # here — graph-edit ops run ONCE at patch creation, which has already happened — so
                    # these spawn directly, following ambient._spawn_visitor's contract exactly: register a
                    # real AgentInstance AND queue a COMPLETE spawn record (a record without an id crashes
                    # Unity's relationship dicts; an instance without a record is an undriven body).
                    def _spawn_event_agent(role, name, zid_home):
                        from dsag.patch import _next_agent_id
                        from dsag.world import AgentInstance
                        from dsag.needs import Needs
                        from ecgp.runtime.spill import _queue as _mq
                        said = _next_agent_id(scene)
                        scene.agents[said] = AgentInstance(id=said, name=name, role=role,
                                                           needs=Needs(), current_zone=zid_home)
                        # ALWAYS spawn at the DOOR (Unity-exported egress gate) and let the role's
                        # directive walk them to zid_home — a VIP or musician materialising mid-room
                        # was the visible bug; the walk in through reception IS the entrance moment.
                        zc = dsag_bridge.entry_point(scene, near_zone=zid_home)
                        _mq(scene, "spawned_agents", {"id": said, "name": name, "role": role,
                                                      "agent_type": role, "zone_id": zid_home,
                                                      "x": float(zc[0]), "y": float(zc[1]),
                                                      "needs": Needs().as_dict()})
                        return said
                    # MUSICIAN / LIVE BAND -> a performer walks IN through the door, sits at the
                    # instrument, and only THEN do the amps + floating notes appear; they vanish the
                    # moment the set ends (or the musician stands up early). The hook itself spawns
                    # nothing but the performer — the seated-state machine in _check_event_props owns
                    # the decor lifecycle, because "props before the music starts" was the reported bug.
                    if _kw_hit(_ev, ("musician", "band", "concert", "performer", "singer",
                                     "guitarist", "live music", "performance", "recital", "gig",
                                     "orchestra", "jazz", "piano", "pianist", "a show", "the show",
                                     "show starts", "live show", "plays music", "playing music")):
                        zid = next((o.zone for o in patch.ops
                                    if o.zone and o.op == "zone_attraction" and (o.delta or 0) > 0), None)
                        zid = _match_zone_id(scene, zid) if zid else None
                        zid = zid or next((z for z in scene.zones if "music" in z.lower()
                                           or "event" in z.lower() or "stage" in z.lower()), None)
                        if zid and zid in scene.zones:
                            from dsag.patch import PatchOp as _POp
                            gigs = getattr(scene, "_gigs", None)
                            if gigs is None:
                                gigs = scene._gigs = {}
                            gid = f"gig_{len(gigs)}"
                            mid = next((a.id for a in scene.agents.values()
                                        if (getattr(a, "role", "") or "").lower() == "musician"), None)
                            if mid is None:
                                mid = _spawn_event_agent("musician", "The Musician", zid)
                            patch.ops.append(_POp(op="role_directive", role="musician", zone=zid))
                            if not any(o.op == "zone_attraction" and (o.delta or 0) > 0 for o in patch.ops):
                                patch.ops.append(_POp(op="zone_attraction", zone=zid, delta=65.0))
                            # state machine: arriving -> (seated) playing 10s -> torn down.
                            # `deadline` bounds the walk-in so a musician who can never reach the seat
                            # doesn't leave a gig record pending forever.
                            gigs[gid] = {"zone": zid, "musician": mid, "state": "arriving",
                                         "props": [], "play_s": 10.0,
                                         "deadline": time.monotonic() + 90.0}
                            log.info(f"[inbox]    GIG -> musician {mid} walking in; decor appears when seated in {zid}")
                    # VIP / CELEBRITY -> a distinguished guest arrives, the crowd gathers where they hold
                    # court, and they slip out when the visit is over. No prop — the crowd IS the visual.
                    if _kw_hit(_ev, ("vip", "celebrity", "famous", "influencer", "star guest")):
                        zid = next((o.zone for o in patch.ops
                                    if o.zone and o.op == "zone_attraction" and (o.delta or 0) > 0), None)
                        zid = _match_zone_id(scene, zid) if zid else None
                        zid = (zid or next((z for z in scene.zones if "event" in z.lower()
                                            or "lounge" in z.lower()), None)
                                   or (next(iter(scene.zones)) if scene.zones else None))
                        if zid and zid in scene.zones:
                            from dsag.patch import PatchOp as _POp
                            vips = getattr(scene, "_vips", None)
                            if vips is None:
                                vips = scene._vips = {}
                            if not any((getattr(a, "role", "") or "").lower() == "vip"
                                       for a in scene.agents.values()):
                                _spawn_event_agent("vip", "The Celebrity", zid)
                            patch.ops.append(_POp(op="role_directive", role="vip", zone=zid))
                            if not any(o.op == "zone_attraction" and (o.delta or 0) > 0 for o in patch.ops):
                                patch.ops.append(_POp(op="zone_attraction", zone=zid, delta=70.0))
                            vips[f"vip_{len(vips)}"] = {"until": time.monotonic()
                                                        + min(180.0, max(60.0, float(patch.ttl or 25) * 5.0))}
                            log.info(f"[inbox]    VIP -> celebrity holds court in {zid}")
                    # POWER OUTAGE / SHORT CIRCUIT -> one machine goes dark (object-scoped disable, so the
                    # rest of the venue keeps serving) with a spark prop on it until the patch expires.
                    if (_intent in ("object_disable", "equipment_failure")
                            or _kw_hit(_ev, ("outage", "power cut", "short circuit", "blackout",
                                             "sparking", "power failure", "out of order", "broken",
                                             "not working", "malfunction", "breakdown", "no power"))):
                        mach = (next((o for o in scene.objects.values()
                                      if "machine" in o.id.lower() or "coffee" in o.id.lower()), None)
                                or next((o for o in scene.objects.values()
                                         if any(getattr(a, "action", None) == "drink"
                                                for a in getattr(o, "affordances", []))), None))
                        if mach is not None:
                            from ecgp.runtime.spill import _queue as _mutq
                            from dsag.patch import PatchOp as _POp
                            outs = getattr(scene, "_outages", None)
                            if outs is None:
                                outs = scene._outages = {}
                            spid = f"spark_{len(outs)}"
                            mpx, mpy = getattr(mach, "prop_pos", mach.pos)
                            _mutq(scene, "spawned_objects", {"id": spid, "object_type": "spark",
                                                             "zone_id": mach.zone_id,
                                                             "x": float(mpx), "y": float(mpy)})
                            # OUT OF SERVICE for real. An object-scoped `disable_affordance` op does
                            # NOT work — dsag.patch.action_disabled matches on action+zone and never
                            # reads op.object, so agents kept using the "broken" machine and only the
                            # spark appeared. break_object() marks it removed (every option loop skips
                            # a removed object, and Unity hides it), dispatches a staff member, and
                            # restores it ~8s later, taking the spark down with it.
                            from ecgp.runtime.live_bridge import break_object as _break
                            _break(scene, mach, props=[spid])
                            outs.pop(spid, None)          # the repair owns this prop's lifetime now
                            log.info(f"[inbox]    OUTAGE -> {mach.id} OUT OF SERVICE, spark {spid}, "
                                     f"staff repair in ~8s")
                    # GIFT GIVEAWAY -> a gift crate customers walk up to and receive from — the first event
                    # to exercise the `help` affordance (Unity plays the Help/spellcast clip at it).
                    if _kw_hit(_ev, ("gift", "giveaway", "present", "souvenir", "free merch")):
                        zid = next((o.zone for o in patch.ops
                                    if o.zone and o.op == "zone_attraction" and (o.delta or 0) > 0), None)
                        zid = _match_zone_id(scene, zid) if zid else None
                        zid = (zid or next((z for z in scene.zones if "recep" in z.lower()
                                            or "event" in z.lower()), None)
                                   or (next(iter(scene.zones)) if scene.zones else None))
                        if zid and zid in scene.zones:
                            from ecgp.runtime.spill import _queue as _mutq
                            from dsag.smart_object import SmartObject, Affordance
                            from dsag.patch import PatchOp as _POp
                            pops = getattr(scene, "_popups", None)
                            if pops is None:
                                pops = scene._popups = {}
                            gcx, gcy = scene.zones[zid].center
                            oid = f"popup_gift_{len(pops)}"
                            obj = SmartObject(id=oid, object_type="gift_box", zone_id=zid,
                                              states=["available"], state="available",
                                              affordances=[Affordance("help",
                                                  need_effects={"status": -15, "loneliness": -10})],
                                              pos=(float(gcx) + 1.2, float(gcy) - 0.4))
                            obj.capacity = 3
                            obj.display_name = "Gift Table"
                            scene.objects[oid] = obj
                            pops[oid] = time.monotonic() + min(120.0, max(40.0, float(patch.ttl or 25) * 5.0))
                            _mutq(scene, "spawned_objects", {"id": oid, "object_type": "gift_box",
                                                             "zone_id": zid, "x": obj.pos[0], "y": obj.pos[1]})
                            patch.ops.append(_POp(op="object_attraction", object=oid, delta=60.0))
                            log.info(f"[inbox]    GIVEAWAY -> gift table in {zid}")
                elif patch.rejected_ops and not patch.ops:
                    log.warning(f"[inbox] '{text[:50]}' -> ALL ops rejected as malformed — no patch applied")
            write_event_log(pd, source=f"inbox:{f.name}")
        try:
            shutil.move(str(f), str(processed / f"{_timestamp()}_{f.name}"))
        except Exception:
            try: f.unlink()
            except Exception: pass


# ── EventLog file helpers ───────────────────────────────────────────────────────
def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def write_scene_log(payload: dict):
    """Write one JSON file per generated scene (zones + character types + agents)."""
    try:
        name = payload.get("scene_name", "scene")
        safe = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_") or "scene"
        path = EVENTLOG / "scenes" / f"{_timestamp()}_{safe}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"Scene log written: {path}")
    except Exception as e:
        log.warning(f"scene log write failed: {e}")

def write_event_log(result: dict, source: str):
    """Record every interpreted event (UI field or inbox file)."""
    try:
        path = EVENTLOG / "events" / f"{_timestamp()}_{datetime.datetime.now().microsecond}_event.json"
        path.write_text(json.dumps({"source": source, "result": result}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    except Exception as e:
        log.warning(f"event log write failed: {e}")

def apply_event_result(description: str, result: dict, source: str):
    """Turn an interpreted event into a directive/emergency the next tick will honor."""
    directive = result.get("directive") or result.get("behavior_hint") or description
    # One-shot hint guarantees the agents react on the very next tick.
    sim.event_hint = f'USER SAID: "{description}". {directive}'

    if result.get("is_emergency"):
        sim.active_event  = result.get("event_type", "emergency")
        sim.event_message = result.get("display_name", description)
        log.info(f"Emergency triggered ({source}): {sim.active_event}")
    elif result.get("clears_emergency"):
        if sim.active_event:
            log.info(f"Emergency cleared ({source}): was '{sim.active_event}'")
        if sim.standing_orders:
            log.info(f"Standing orders cleared ({source}): {len(sim.standing_orders)} dropped")
        sim.active_event     = ""
        sim.event_message    = ""
        sim.standing_orders  = []
        # Wipe stale current_zone so the director re-decides everyone from scratch
        for agent in sim.agents.values():
            agent.current_zone = ""
    elif result.get("is_standing_order"):
        # Persist this command — Claude will keep enforcing it every tick until an all-clear.
        sim.standing_orders.append(directive)
        # Keep the list bounded; newest orders win if they conflict with older ones.
        if len(sim.standing_orders) > 8:
            sim.standing_orders = sim.standing_orders[-8:]
        log.info(f"Standing order added ({source}): {directive!r} (total {len(sim.standing_orders)})")

    write_event_log(result, source)

# Sentinel the All-Clear button drops into the inbox — cleared directly, no Claude call.
ALL_CLEAR_SENTINEL = "__ALL_CLEAR__"

def clear_all_directives(source: str):
    """Drop every active emergency AND standing order; let agents resume their needs."""
    if sim.active_event:
        log.info(f"Emergency cleared ({source}): was '{sim.active_event}'")
    if sim.standing_orders:
        log.info(f"Standing orders cleared ({source}): {len(sim.standing_orders)} dropped")
    sim.active_event    = ""
    sim.event_message   = ""
    sim.event_hint      = ""
    sim.standing_orders = []
    for agent in sim.agents.values():
        agent.current_zone = ""

async def process_inbox():
    """Scan EventLog/inbox for dropped event files; interpret and queue each one.
    Called every director tick, so a dropped file reacts within one tick — like
    pressing a button, but from a file."""
    inbox     = EVENTLOG / "inbox"
    processed = EVENTLOG / "processed"
    files = sorted(p for p in inbox.glob("*") if p.suffix.lower() in (".txt", ".json", ".md"))
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore").strip().lstrip("﻿").strip()
        except Exception:
            text = ""
        if text == ALL_CLEAR_SENTINEL:
            log.info(f"[inbox] all-clear sentinel '{f.name}'")
            clear_all_directives(source=f"inbox:{f.name}")
        elif text:
            log.info(f"[inbox] event file '{f.name}': {text}")
            result = await interpret_event(text, sim.scene_name)
            apply_event_result(text, result, source=f"inbox:{f.name}")
        # Move the file out of the inbox so it only fires once
        try:
            dest = processed / f"{_timestamp()}_{f.name}"
            shutil.move(str(f), str(dest))
        except Exception:
            try: f.unlink()
            except Exception: pass

# ── WebSocket handler ──────────────────────────────────────────────────────────
NAMES = [
    "Alex","Jamie","Sam","Casey","Jordan","Morgan","Taylor","Riley",
    "Avery","Quinn","Sage","River","Blake","Drew","Reese","Skylar",
    "Harley","Finley","Rowan","Emery","Parker","Logan","Charlie","Frankie",
    "Robin","Jessie","Cameron","Dakota","Kendall","Peyton","Hayden","Spencer",
    "Addison","Bailey","Corey","Devon","Elliot","Flynn","Glenn","Harper"
]

async def stream_scene_sprites(send, config):
    """Generate a sprite per character type + per zone via Gemini and stream each PNG to
    Unity as it completes. Keyed by agent_type (characters) and zone id (zones) so Unity
    can hot-swap the matching agents/zones. Runs as a background task; never blocks a tick."""
    chars = [(t.get("type"), t.get("sprite_prompt", "")) for t in config.get("agent_types", []) if t.get("sprite_prompt")]
    zones = [(z.get("id"),   z.get("sprite_prompt", "")) for z in config.get("zones", [])       if z.get("sprite_prompt")]
    total = len(chars) + len(zones)
    if total == 0:
        return

    await send(json.dumps({"type": "sprite_status", "state": "start", "total": total}))
    sem  = asyncio.Semaphore(4)          # bound concurrency so we don't hammer the API
    prog = {"done": 0}
    prog_lock = asyncio.Lock()

    async def one(kind, key, prompt):
        async with sem:
            png = await sprite_gen.generate_sprite(prompt, kind)
        if png:
            b64 = base64.b64encode(png).decode("ascii")
            await send(json.dumps({"type": "sprite_asset", "kind": kind, "key": key, "png_base64": b64}))
        async with prog_lock:
            prog["done"] += 1
            log.info(f"[sprite] {kind} '{key}' {'ok' if png else 'FAILED'} ({prog['done']}/{total})")

    tasks  = [asyncio.create_task(one("character", k, p)) for k, p in chars]
    tasks += [asyncio.create_task(one("zone",      k, p)) for k, p in zones]
    await asyncio.gather(*tasks, return_exceptions=True)
    await send(json.dumps({"type": "sprite_status", "state": "done", "total": total}))
    log.info(f"[sprite] scene sprite streaming complete ({total} sprites)")


async def stream_lpc_characters(send, config):
    """Generate a bespoke animated LPC walk sheet per agent_type — Claude picks pixel layers from
    each type's description, the compositor stacks + recolours them — and stream each to Unity,
    which slices and hot-swaps it onto the matching agents. Cached on disk (each character made
    once). Runs in the background so the scene is playable immediately: agents animate with a
    pre-baked role sheet until their bespoke one arrives. Not limited to the pre-made roles."""
    if not _LPC_OK:
        return
    types = [(t.get("type"), t.get("sprite_prompt") or t.get("label") or t.get("type"),
              int(t.get("count", 1) or 1))
             for t in config.get("agent_types", []) if t.get("type")]
    if not types:
        return
    # Pass the venue theme alongside the name so the recipe biases clothing to the setting (gym-goers in
    # athletic wear, nurses in scrubs, office workers in business attire) even when the per-type prompt is thin.
    scene = f"{config.get('scene_name', '')} ({config.get('theme', '')} setting)".strip()
    # Generation runs BEFORE the scene is shown, so parallelise across types to keep the wait short
    # (each type is one Claude recipe call + fast recolours); capped to stay within API rate limits.
    sem = asyncio.Semaphore(min(max(len(types), 1), 8))
    # bounded pool of distinct looks per type — memory is independent of agent count (all agents of
    # a type share these K textures), so this scales to 80-90+ agents. Only the first look costs a
    # Claude call; the rest are cheap skin/hair/clothes recolours of it.
    VARIANT_CAP = 12

    async def one(atype, desc, count):
        k = max(1, min(count, VARIANT_CAP))
        async with sem:
            try:
                pngs = await asyncio.to_thread(
                    lpc_runtime.sheet_variants_png, atype, desc, scene, k)
            except Exception as e:
                log.warning(f"[lpc] '{atype}' generation failed: {e}")
                return
        n = len(pngs)
        for i, png in enumerate(pngs):
            if not png:
                continue
            b64 = base64.b64encode(png).decode("ascii")
            await send(json.dumps({"type": "lpc_sheet", "key": atype,
                                   "variant": i, "variants": n, "png_base64": b64}))
        log.info(f"[lpc] streamed {n} variant look(s) for '{atype}' ({count} agents)")

    await asyncio.gather(*[one(a, d, c) for a, d, c in types], return_exceptions=True)
    log.info(f"[lpc] character streaming complete ({len(types)} types)")


async def handle_client(ws):
    log.info(f"Unity connected from {ws.remote_address}")

    # All sends funnel through this lock: the background sprite-streaming task sends on the
    # same connection as the main request loop, and websockets forbids concurrent sends.
    send_lock = asyncio.Lock()
    async def send(payload):
        async with send_lock:
            await ws.send(payload)

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            mtype = msg.get("type", "")

            if mtype == "generate_scene":
                description    = msg.get("description", "a busy public place")
                render_mode    = msg.get("render_mode", "library")  # "library" | "generate"
                director_mode  = msg.get("director_mode", "llm")    # "llm" | "dsag"
                sim.director_mode = director_mode
                # behavior back-end for the graph path. Older Unity builds send "gnn"/"gnn_hybrid" (the
                # removed learned model) — normalise ANY learned-engine request to "ecgp"; only an explicit
                # "rule" uses the symbolic engine. So the demo runs the trained ECGP policy by default.
                _be = msg.get("behavior_engine", sim.behavior_engine).lower()
                sim.behavior_engine = "rule" if _be == "rule" else "ecgp"
                log.info(f"[ecgp] behavior engine = {sim.behavior_engine}")
                sim.dsag_scene    = None
                sim.description = description
                try:
                    await send(json.dumps({"type": "generating", "message": "Claude is designing your scene..."}))
                    config = await generate_scene(description)

                    sim.scene_name      = config["scene_name"]
                    sim.theme           = config.get("theme", "cafe")
                    sim.scene_spec      = config.get("scene_spec")
                    sim._scene_cfg      = config          # ambient-event cadence reads this
                    sim.scene_graph     = None
                    sim.zones           = config["zones"]
                    sim.behaviour_notes = config.get("behaviour_notes", "")
                    sim.agents.clear()
                    sim.events.clear()
                    sim.pending_actions.clear()
                    sim.active_event    = ""
                    sim.event_message   = ""
                    sim.event_hint      = ""
                    sim.standing_orders = []

                    agent_list = []
                    agent_id   = 0
                    _social_unit_meta.clear()
                    for atype in config["agent_types"]:
                        _social_unit_meta[atype["type"]] = _social_unit_for(atype)
                        personality_type = atype.get("personality_type", "extrovert")
                        for i in range(atype.get("count", 3)):
                            aid   = f"agent_{agent_id}"
                            name  = NAMES[agent_id % len(NAMES)]
                            needs = dict(atype.get("initial_needs", {}))
                            sim.agents[aid] = AgentState(
                                id=aid, name=name,
                                agent_type=atype["type"],
                                personality=personality_type,
                                x=rnd.uniform(-5, 5),
                                y=rnd.uniform(-3, 3),
                                needs=needs
                            )
                            agent_list.append({
                                "id":               aid,
                                "name":             name,
                                "agent_type":       atype["type"],
                                "personality_type": personality_type,
                                "color":            atype.get("color", "#6688cc"),
                                "x":                sim.agents[aid].x,
                                "y":                sim.agents[aid].y,
                                "needs":            needs
                            })
                            agent_id += 1
                    # SOCIAL UNITS (item 6): cluster individuals into families/friend groups + authored ties,
                    # so co-grouped agents surface talk/help/group options and the behavior prior can express
                    # family cohesion. group_id rides on each agent dict -> Unity + the scene model.
                    sim.social_groups = assign_social_groups(agent_list, rnd)
                    for a in agent_list:
                        sim.agents[a["id"]].group_id = a.get("group_id")
                    log.info(f"[social] {len(sim.social_groups)} groups: "
                             + ", ".join(f"{g}({d['type']}x{len(d['members'])})"
                                         for g, d in list(sim.social_groups.items())[:8]))

                    # Generate the bespoke animated LPC characters BEFORE the scene is shown, so
                    # every agent appears already-correct and NEVER changes appearance mid-scene
                    # (no placeholder swap — the previous behaviour the user rejected). The sheets
                    # stream first; Unity buffers them (bracketed by lpc_status) and applies the
                    # right per-agent variant at spawn. Timeout-guarded so a slow/failed generation
                    # can't stall the scene — agents then fall back to the pre-baked role sheets.
                    if _LPC_OK:
                        await send(json.dumps({"type": "lpc_status", "state": "generating"}))
                        try:
                            await asyncio.wait_for(stream_lpc_characters(send, config), timeout=60)
                        except asyncio.TimeoutError:
                            log.warning("[lpc] character generation timed out — using fallback sheets")
                        except Exception as e:
                            log.warning(f"[lpc] character generation error: {e}")
                        await send(json.dumps({"type": "lpc_status", "state": "ready"}))

                    _SCENE_EPOCH[0] += 1     # item 4: every scene_ready invalidates in-flight commands from before
                    _LAST_INTENT_ID.clear(); _LAST_COMMAND_ID.clear()   # agent ids restart from 0 in a new scene
                    _reset_behavior_state()
                    await send(json.dumps({
                        "type":            "scene_ready",
                        "scene_epoch":     _SCENE_EPOCH[0],
                        "scene_name":      sim.scene_name,
                        "theme":           sim.theme,
                        "scene_spec":      sim.scene_spec,   # open-vocab: kits + smart_objects (placed by affordance)
                        "description":     sim.description,
                        "zones":           sim.zones,
                        "agents":          agent_list,
                        "behaviour_notes": sim.behaviour_notes
                    }))
                    log.info(f"Scene sent to Unity: {sim.scene_name}")

                    # ISSUE 4: surface the generated STORY + agent roster in the Unity log so the user can read
                    # what Claude designed (the cast, their personalities, and the social groups in the scene).
                    roster = []
                    for atype in config.get("agent_types", []):
                        line = f"{atype.get('count', 0)}x {atype.get('type', '?')}"
                        if atype.get("personality_type"):
                            line += f" ({atype['personality_type']})"
                        su = _social_unit_meta.get(atype.get("type", ""), "solo")
                        if su != "solo":
                            line += f" [{su}]"
                        roster.append(line)
                    grp_summary = {}
                    for _g, _d in sim.social_groups.items():
                        grp_summary[_d["type"]] = grp_summary.get(_d["type"], 0) + 1
                    # Lead with Claude's story narrative (prose), then a compact cast line for the agent
                    # roster — story first, information second (the user wants it to read like a story).
                    narrative = (config.get("narrative") or "").strip()
                    story_lines = []
                    if narrative:
                        story_lines.append(narrative)
                    else:
                        story_lines.append(f"{sim.scene_name} ({sim.theme})")
                        if sim.behaviour_notes:
                            story_lines.append(sim.behaviour_notes)
                    story_lines.append("CAST: " + ", ".join(roster))
                    if grp_summary:
                        story_lines.append("GROUPS: " + ", ".join(f"{n} {t}" for t, n in grp_summary.items()))
                    await send(json.dumps({"type": "director_log", "lines": story_lines}))

                    # Dump the full scene (zones + character types + agents) to EventLog/scenes
                    write_scene_log({
                        "scene_name":      sim.scene_name,
                        "description":     sim.description,
                        "behaviour_notes": sim.behaviour_notes,
                        "zones":           sim.zones,
                        "agent_types":     config.get("agent_types", []),
                        "agents":          agent_list,
                    })

                    # DSAG path: build the affordance-graph scene model and push the initial
                    # smart-object states to Unity. The existing LLM director is bypassed for
                    # this session (see request_tick); nothing on the "llm" path changes.
                    if director_mode == "dsag":
                        sim.dsag_scene = dsag_bridge.build_scene_model(sim.zones, agent_list)
                        objs = [o.render_state() for o in sim.dsag_scene.objects.values()]
                        await send(json.dumps({"type": "object_states", "objects": objs}))
                        log.info(f"[ecgp] scene built: {len(sim.dsag_scene.agents)} agents, "
                                 f"{len(sim.dsag_scene.objects)} smart objects")

                    # Runtime sprite generation (Option B): if the user chose "generate"
                    # mode, stream Nano-Banana sprites in the background as each finishes,
                    # so the scene is playable immediately and characters/zones fill in.
                    if render_mode == "generate":
                        if sprite_gen.is_available():
                            asyncio.create_task(stream_scene_sprites(send, config))
                            log.info("[sprite] generate mode — streaming custom sprites in background")
                        else:
                            await send(json.dumps({"type": "sprite_status", "state": "unavailable",
                                                   "message": "Server has no GEMINI_API_KEY; using library sprites."}))
                    # (Bespoke LPC characters were already generated + streamed above, before
                    # scene_ready, so agents render correct from the first frame.)
                except Exception as e:
                    log.error(f"Scene error: {e}")
                    await send(json.dumps({"type": "error", "message": str(e)}))

            elif mtype == "load_prebuilt":
                # PREBUILT SCENE (isolated from generate_scene): load a hand-authored config with a fixed
                # floor-plan template + fixed-coord smart objects, then run the SAME downstream as a generated
                # scene (agents, social groups, scene_ready, DSAG). No LLM designer and no bespoke LPC
                # generation, so it appears instantly and identically. The generate_scene branch is untouched.
                key           = msg.get("key", "")
                director_mode = msg.get("director_mode", "dsag")
                sim.director_mode = director_mode
                _be = msg.get("behavior_engine", sim.behavior_engine).lower()
                sim.behavior_engine = "rule" if _be == "rule" else "ecgp"
                sim.dsag_scene = None
                try:
                    config = load_prebuilt_config(key)
                    sim.scene_name      = config["scene_name"]
                    sim.description     = config.get("scene_theme_text", sim.scene_name)
                    sim.theme           = config.get("theme", "home")
                    sim.scene_spec      = config.get("scene_spec")
                    sim._scene_cfg      = config          # ambient-event cadence reads this
                    sim.scene_graph     = None
                    sim.zones           = config["zones"]
                    sim.behaviour_notes = config.get("behaviour_notes", "")
                    sim.agents.clear(); sim.events.clear(); sim.pending_actions.clear()
                    sim.active_event = ""; sim.event_message = ""; sim.event_hint = ""
                    sim.standing_orders = []

                    agent_list = []; agent_id = 0
                    _social_unit_meta.clear()
                    for atype in config["agent_types"]:
                        _social_unit_meta[atype["type"]] = _social_unit_for(atype)
                        personality_type = atype.get("personality_type", "extrovert")
                        for _i in range(atype.get("count", 1)):
                            aid  = f"agent_{agent_id}"
                            name = NAMES[agent_id % len(NAMES)]
                            needs = dict(atype.get("initial_needs", {}))
                            sim.agents[aid] = AgentState(
                                id=aid, name=name, agent_type=atype["type"],
                                personality=personality_type,
                                x=rnd.uniform(-5, 5), y=rnd.uniform(-3, 3), needs=needs)
                            agent_list.append({
                                "id": aid, "name": name, "agent_type": atype["type"],
                                "personality_type": personality_type,
                                # AUTHORED ROLE, carried end-to-end. The scene already declares `role` on every
                                # agent_type and this record used to DROP it, so both the behaviour engine and
                                # Unity re-derived a role from `agent_type` keywords instead. That guess
                                # collapses the cast: `regular` and `remote_worker` both land on one role, as do
                                # `friends_group` and `visitor_family` — which is why so many agents ended up
                                # sharing a character sheet, and why the server could not tell a barista from a
                                # waiter for post assignment.
                                "role": atype.get("role"),
                                "outfit_category": atype.get("outfit_category"),
                                "color": atype.get("color", "#6688cc"),
                                "x": sim.agents[aid].x, "y": sim.agents[aid].y, "needs": needs})
                            agent_id += 1
                    sim.social_groups = assign_social_groups(agent_list, rnd)
                    for a in agent_list:
                        sim.agents[a["id"]].group_id = a.get("group_id")

                    _SCENE_EPOCH[0] += 1
                    _LAST_INTENT_ID.clear(); _LAST_COMMAND_ID.clear()
                    _reset_behavior_state()
                    await send(json.dumps({
                        "type":            "scene_ready",
                        "scene_epoch":     _SCENE_EPOCH[0],
                        "scene_name":      sim.scene_name,
                        "theme":           sim.theme,
                        "scene_spec":      sim.scene_spec,
                        "description":     sim.description,
                        "zones":           sim.zones,
                        "agents":          agent_list,
                        "behaviour_notes": sim.behaviour_notes,
                        "prebuilt":        config.get("prebuilt_key", key),   # -> Unity enables template mode
                        # PREBAKED level: the Unity scene already contains the hand-built floors/walls/props
                        # (e.g. CafeDemo.unity's CafeLevel), so the renderer must NOT draw or clear geometry —
                        # it only registers zones, makes smart-object anchors, and rebuilds nav from the
                        # existing wall colliders. Template-backdrop scenes (tea house) leave this false.
                        "prebaked_level":  bool(config.get("prebaked_level", False)),
                    }))
                    log.info(f"Prebuilt scene sent to Unity: {sim.scene_name} ({len(agent_list)} agents)")

                    narrative = (config.get("narrative") or "").strip()
                    story = [narrative] if narrative else [f"{sim.scene_name} ({sim.theme})"]
                    story.append("CAST: " + ", ".join(
                        f"{t.get('count', 0)}x {t.get('type', '?')}" for t in config.get("agent_types", [])))
                    await send(json.dumps({"type": "director_log", "lines": story}))

                    write_scene_log({
                        "scene_name": sim.scene_name, "description": sim.description,
                        "behaviour_notes": sim.behaviour_notes, "zones": sim.zones,
                        "agent_types": config.get("agent_types", []), "agents": agent_list})

                    if director_mode == "dsag":
                        sim.dsag_scene = dsag_bridge.build_scene_model(sim.zones, agent_list)
                        # A PREBAKED level authors its own smart objects at fixed positions. Replace the
                        # idealized per-zone template objects build_scene_model() seeds, or their made-up ids
                        # go out in this first object_states and Unity scatters a placeholder sprite for each
                        # across the zone — the stray smart-object dots on empty floor.
                        spec_objs = (sim.scene_spec or {}).get("smart_objects") or []
                        n_seed = dsag_bridge.seed_objects_from_spec(sim.dsag_scene, spec_objs)
                        if n_seed:
                            log.info(f"[ecgp] seeded {n_seed} authored smart objects (templates discarded)")
                        objs = [o.render_state() for o in sim.dsag_scene.objects.values()]
                        await send(json.dumps({"type": "object_states", "objects": objs}))
                        log.info(f"[ecgp] prebuilt scene built: {len(sim.dsag_scene.agents)} agents, "
                                 f"{len(sim.dsag_scene.objects)} smart objects")
                except Exception as e:
                    log.error(f"Prebuilt scene error: {e}")
                    await send(json.dumps({"type": "error", "message": f"Prebuilt '{key}': {e}"}))

            elif mtype == "scene_graph":
                # Unity exports the PLACED scene graph (smart objects + interaction points + nav-reachability
                # + spawn points). This grounds the DSAG/EventPatch behavior pipeline so it can target concrete
                # smart objects, not just zones. Behavior stays Python-side; Unity only executes movement.
                sim.scene_graph = msg.get("graph")
                if sim.scene_graph:
                    n_obj = len(sim.scene_graph.get("smart_objects", []))
                    reach = sum(1 for o in sim.scene_graph.get("smart_objects", []) if o.get("nav_reachable"))
                    log.info(f"[scene_graph] {n_obj} smart objects placed ({reach} nav-reachable), "
                             f"{len(sim.scene_graph.get('zones', []))} zones")
                    # GAP 1 — ground the DSAG behavior graph in the REAL placed objects: the GNN/rule now
                    # target concrete, nav-reachable, placed smart objects (prop:<id>) instead of the
                    # idealized template objects, and actions carry each object's real interaction point.
                    if sim.dsag_scene is not None:
                        n = dsag_bridge.ground_scene_in_graph(sim.dsag_scene, sim.scene_graph)
                        if n:
                            objs = [o.render_state() for o in sim.dsag_scene.objects.values()]
                            await send(json.dumps({"type": "object_states", "objects": objs}))
                            log.info(f"[ecgp] grounded {n} placed smart objects from scene_graph "
                                     f"(behavior now targets real objects)")

            elif mtype == "update_state":
                for a in msg.get("agents", []):
                    # mirror the LIVE Unity position onto the behavior-scene agent so proximity checks
                    # (e.g. "is anyone actually AT this object" for the in-use status) use real positions
                    if sim.dsag_scene is not None and a["id"] in sim.dsag_scene.agents:
                        sim.dsag_scene.agents[a["id"]].pos = (float(a.get("x", 0.0)), float(a.get("y", 0.0)))
                    if a["id"] in sim.agents:
                        ag = sim.agents[a["id"]]
                        ag.x            = a.get("x", ag.x)
                        ag.y            = a.get("y", ag.y)
                        ag.current_zone = a.get("current_zone", ag.current_zone)
                        ag.personality  = a.get("personality", ag.personality)
                        if "needs"             in a: ag.needs.update(a["needs"])
                        if "relationships"     in a: ag.relationships = a["relationships"]
                        if "encounter_counts"  in a: ag.encounter_counts = a["encounter_counts"]
                        if "friends"           in a: ag.friends = a["friends"]
                # Store event hint for next tick
                if "event_hint" in msg and msg["event_hint"]:
                    sim.event_hint = msg["event_hint"]

            elif mtype == "agent_despawned":
                # Unity CONFIRMS this agent actually crossed the exit portal (CrowdDirector.SendAgentDespawned).
                # Retires it from the server's bookkeeping on FACT, not a blind tick timer (see
                # ecgp.runtime.live_bridge._maybe_retire) — the fix for a jammed-door "ghost" agent that the
                # server used to forget about while Unity still showed it stuck and undriven. Imported directly
                # here (not the `request_tick`-local `ecgp_live`, which may not exist yet on first connect).
                aid = msg.get("agent_id")
                if aid:
                    from ecgp.runtime import live_bridge as _confirm_mod
                    _confirm_mod.confirm_despawned(aid)
                    log.info(f"[ecgp] confirmed despawn: {aid} (crossed exit portal)")

            elif mtype == "request_tick":
                if not sim.agents:
                    await send(json.dumps({"type": "actions", "actions": [], "events": []}))
                    continue

                # DSAG path: no per-agent LLM in the loop. A user interruption becomes ONE structured
                # graph patch (one LLM call) that reshapes the world; the crowd redirects by reading
                # the patched graph through the affordance engine. Object states pushed too.
                if sim.director_mode == "dsag" and sim.dsag_scene is not None:
                    await process_inbox_dsag(sim.dsag_scene)      # free-text event -> live ScenePatch
                    # Behavior back-end: ECGP (the trained Event-Conditioned Graph Policy, 64x3) if
                    # requested + available, else the deterministic symbolic rule engine. Both read the
                    # SAME grounded, patched Scene-Affordance Graph; the LLM stays the director. ECGP runs
                    # one forward pass in a worker thread so the event loop / keepalive never blocks.
                    engine = "rule"
                    ecgp_live = None
                    if sim.behavior_engine == "ecgp":
                        try:
                            from ecgp.runtime import live_bridge as ecgp_live
                            ecgp_live.get_policy()                 # load the 64x3 checkpoint once (fail fast)
                            engine = "ecgp"
                        except Exception as e:
                            log.warning(f"[ecgp] ecgp behavior unavailable ({e}); using rule engine")
                            ecgp_live = None
                    # AMBIENT WORLD EVENTS — the hybrid half. v3 still decides every agent action; this only
                    # changes the world around them (a spill, a delivery, arrivals, someone leaving) so the
                    # scene has life of its own between director commands. Runs BEFORE the tick so the
                    # policy sees the changed world immediately.
                    try:
                        from ecgp.runtime import ambient as _amb
                        _story = []
                        if _amb.tick(sim.dsag_scene, _AMBIENT_CADENCE[0], _story) and _story:
                            await send(json.dumps({"type": "director_log", "lines": _story}))
                    except Exception as e:
                        log.warning(f"[ambient] skipped: {e}")
                    if engine == "ecgp":
                        try:
                            actions, object_states, dsag_events = await asyncio.to_thread(
                                ecgp_live.ecgp_tick, sim.dsag_scene, sim.zones)
                        except Exception as e:                     # never break the loop on a model error
                            log.warning(f"[ecgp] ecgp tick failed ({e}); falling back to rule engine")
                            actions, object_states, dsag_events = dsag_bridge.tick_to_unity(sim.dsag_scene)
                    else:
                        actions, object_states, dsag_events = dsag_bridge.tick_to_unity(sim.dsag_scene)
                    # Per-agent directive overlays on top of the tick's actions:
                    from dsag import patch as dsag_patch
                    by_id = {a["agent_id"]: a for a in actions}
                    # (1) "X talk to Y": turn the subject's action into a `meet` so Unity walks X to Y's live
                    #     position and they stop + face + converse (a plain zone move only shares a zone).
                    # A meet directive COMPLETES once the pair has converged + had a few ticks to talk — drop it
                    # then (don't wait out the full ttl), so 'X talk to Y' is one interaction, not 25 ticks locked
                    # together. The ttl remains the backstop if the two can never physically reach each other.
                    _done_meets = dsag_patch.meet_patches_complete(
                        sim.dsag_scene.active_patches, sim.dsag_scene, _MEET_MET)
                    if _done_meets:
                        sim.dsag_scene.active_patches = [p for p in sim.dsag_scene.active_patches
                                                         if p not in _done_meets]
                        log.info(f"[ecgp] meet complete (converged) -> released {len(_done_meets)} directive(s)")

                    meets = dsag_patch.meet_pairs(sim.dsag_scene.active_patches, sim.dsag_scene)
                    if meets:
                        applied = []
                        subj_ids = {s for s, _ in meets}
                        tgt_subj = {}
                        for subj, tgt in meets:
                            a = by_id.get(subj)
                            if a is not None:
                                a["action"] = "meet"
                                a["target_agent_id"] = tgt
                                a.pop("smart_object_id", None); a.pop("target_x", None); a.pop("target_y", None)
                                a["reason"] = f"{subj} goes to meet {tgt}"
                                a["source"] = "directive"; a["priority"] = 3
                                applied.append(f"{subj}->{tgt}")
                                tgt_subj.setdefault(tgt, subj)
                        # MUTUAL CONVERGENCE: the TARGET also walks toward the SUBJECT (unless it is itself the
                        # subject of another meet), so the pair meets in the MIDDLE. Without this the target's own
                        # ECGP action walks it away every tick and the subject chases it forever — the endless
                        # 'meet' bug (agent_19 chased a wandering agent_1 for 10+ ticks). Unity's DirectorMeet
                        # still stops+faces+talks BOTH the instant they are within range (whoever arrives first
                        # triggers the conversation); the directive's ttl then releases them.
                        for tgt, subj in tgt_subj.items():
                            ta = by_id.get(tgt)
                            if ta is not None and tgt not in subj_ids:
                                ta["action"] = "meet"
                                ta["target_agent_id"] = subj
                                ta.pop("smart_object_id", None); ta.pop("target_x", None); ta.pop("target_y", None)
                                ta["reason"] = f"{tgt} comes to meet {subj}"
                                ta["source"] = "directive"; ta["priority"] = 3
                        log.info(f"[ecgp] meet overlay: {applied}")   # 'X talk to Y' -> Unity DirectorMeet (mutual)
                    # (2) "X goes home": route the named agent(s) OUT of the building (exit -> off-map -> hide),
                    #     exactly like an evacuation but only for that person — a personal, on-request leave.
                    leavers = dsag_patch.agents_leaving(sim.dsag_scene.active_patches, sim.dsag_scene)
                    _leave_ops = [o for p in sim.dsag_scene.active_patches for o in p.ops if o.op == "agent_leave"]
                    if _leave_ops and not leavers:
                        log.warning(f"[ecgp] agent_leave op(s) present but resolved to NOBODY — check name/group: "
                                    f"{[(o.agent or o.role) for o in _leave_ops]} vs agents "
                                    f"{[getattr(a,'name','?') for a in sim.dsag_scene.agents.values()][:12]}")
                    if leavers:
                        # SAFETY LAYER (item 4): the ECGP precedence (step 0.0) should already have emitted the
                        # leave action via dsag_bridge.build_leave_action. Here we only VERIFY that, and reapply the
                        # SAME shared builder if the engine (e.g. the rule fallback) didn't — logging any mismatch,
                        # rather than constructing a second, potentially-divergent leave action.
                        added, fixed = [], []
                        for aid in leavers:
                            want = dsag_bridge.build_leave_action(sim.dsag_scene, aid)
                            a = by_id.get(aid)
                            if a is None:                                    # engine emitted nothing for this agent
                                actions.append(want); by_id[aid] = want; added.append(aid)
                            elif not (a.get("evacuate") is True and a.get("action") == "move_to_zone"
                                      and a.get("zone_id") == want["zone_id"]):
                                a.update(want); a.pop("smart_object_id", None); a.pop("target_agent_id", None)
                                fixed.append(aid)
                        if added:
                            log.warning(f"[ecgp] leave safety-layer ADDED missing action(s) for {added} (engine={engine})")
                        if fixed:
                            log.info(f"[ecgp] leave safety-layer normalized {len(fixed)} action(s) (engine={engine})")
                        if not added and not fixed:
                            log.info(f"[ecgp] leave verified — ECGP precedence handled all {len(leavers)}: {leavers}")
                    _finalize_intents(actions)
                    # AUTHORITATIVE NEEDS SNAPSHOT. The server and Unity each ran their OWN needs simulation
                    # and nothing ever reconciled them (update_state mirrors positions into the dsag scene
                    # but not needs, and the server never sent needs back). They diverge within a minute or
                    # two — Unity's inspect panel showed agents starving/"Thirsty" with a red urgency dot
                    # while the server, which actually decides behaviour AND applies relief on eat/drink,
                    # considered them fine. Sending the server's values every tick makes Unity's bars an
                    # honest display of the sim that is actually in charge; Unity still drifts locally
                    # between ticks so the bars move smoothly, then re-anchors here.
                    agent_needs = {}
                    for _aid, _ag in sim.dsag_scene.agents.items():
                        try:
                            agent_needs[_aid] = _ag.needs.as_dict()
                        except Exception:
                            pass
                    await send(json.dumps({"type": "actions", "actions": actions, "events": [],
                                           "agent_needs": agent_needs}))
                    await send(json.dumps({"type": "object_states", "objects": object_states}))
                    # ZONE STATUS: a zone is CLOSED when an active patch disables its affordances (a spill
                    # restriction, 'toilet is dirty', 'counter closed'). Unity shows the closed sign there
                    # until the patch clears. Only zone-scoped disables count (not a global disable).
                    restock_zones = list(getattr(sim.dsag_scene, "_restocks", {}) or {})
                    closed_zones = [z.id for z in sim.dsag_scene.zones.values()
                                    if z.id not in restock_zones
                                    and any(o.op == "disable_affordance" and o.zone
                                            and dsag_patch.zone_matches(o, z)
                                            for p in sim.dsag_scene.active_patches for o in p.ops)]
                    await send(json.dumps({"type": "zone_status", "closed_zones": closed_zones,
                                           "restock_zones": restock_zones}))   # restock -> 'be back soon' sign
                    # ZONE DECOR: balloons over any zone with an active party (until its ttl runs out). Stateless
                    # sync — Unity adds a balloon for each listed zone and removes it from any zone not listed.
                    _decor = getattr(sim.dsag_scene, "zone_decor", None) or {}
                    _now = time.monotonic()
                    for _z in [z for z, exp in _decor.items() if exp <= _now]:
                        _decor.pop(_z, None)               # prune expired parties
                    await send(json.dumps({"type": "zone_decor",
                                           "zones": [{"zone_id": z, "sprite": "balloon"} for z in _decor]}))
                    # OBJECT LOG: smart-object lifecycle messages ('X finished the cup — empty', 'staff cleaned
                    # the toilet', 'staff restocked the food') shown in the Unity log in amber.
                    # NORMALIZE TO STRINGS. The two engines return different event shapes — ecgp_tick emits
                    # plain strings, but the rule fallback's tick_to_unity emits dicts (SceneEvent.as_dict).
                    # Unity deserializes `messages` as List<string>, so one fallback tick's dict blew up the
                    # whole HandleMessage with a JsonReaderException. Coerce every entry here, at the single
                    # send site, so the wire contract holds no matter which engine produced the events.
                    if dsag_events:
                        _msgs = [m if isinstance(m, str)
                                 else (m.get("message") or m.get("type") or json.dumps(m)) if isinstance(m, dict)
                                 else str(m)
                                 for m in dsag_events]
                        await send(json.dumps({"type": "object_log", "messages": _msgs}))
                    # GRAPH-EDIT: forward any structural scene mutations (despawn props / spawn agents) so
                    # Unity mirrors the graph the policy is already acting on.
                    mut = getattr(sim.dsag_scene, "pending_mutations", None)
                    if mut and (mut.get("removed_objects") or mut.get("spawned_agents")
                                or mut.get("spawned_objects")):
                        # spawned_objects/removed_objects carry the STAGE-2 spill lifecycle (transient props
                        # Unity creates then destroys); spawned_agents = graph-edit responders.
                        await send(json.dumps({"type": "scene_mutation",
                                               "removed_objects": mut.get("removed_objects", []),
                                               "spawned_agents": mut.get("spawned_agents", []),
                                               "spawned_objects": mut.get("spawned_objects", [])}))
                        sim.dsag_scene.pending_mutations = {"removed_objects": [], "spawned_agents": []}
                    # Gap 1 integration guard + decision trace: every object the policy picked MUST be a
                    # real, nav-reachable, currently-exported object. Warn on any violation and log a sample
                    # of concrete (engine, agent, object, interaction-point) decisions for the paper trace.
                    obj_actions = [a for a in actions if a.get("smart_object_id")]
                    if sim.scene_graph:
                        reachable = {o.get("smart_object_id")
                                     for o in sim.scene_graph.get("smart_objects", [])
                                     if o.get("nav_reachable", True)}
                        # a spill (or any object spawned AFTER the graph export) is real + reachable server-side
                        # even though it isn't in the exported graph — don't flag those as violations.
                        live_objs = set(getattr(sim.dsag_scene, "objects", {}))
                        bad = [a["smart_object_id"] for a in obj_actions
                               if a["smart_object_id"] not in reachable and a["smart_object_id"] not in live_objs]
                        if bad:
                            log.warning(f"[{engine}] GUARD: {len(bad)} action(s) target "
                                        f"non-exported/unreachable objects: {bad[:5]}")
                    for a in obj_actions[:5]:
                        log.info(f"[{engine}] {a['agent_id']} -> {a['action']} {a['smart_object_id']} "
                                 f"@{a['zone_id']} pt=({a.get('target_x')},{a.get('target_y')}) :: {a.get('reason','')}")
                    # NON-object-targeted actions (zone-only moves — this is EVERY hybrid:party/hybrid:gather
                    # move, since a party slot has no smart_object_id) were previously INVISIBLE in this log:
                    # only obj_actions got a per-agent sample, so an active party's agents never appeared here
                    # even when the mechanism was working correctly — reading this log alone made a working
                    # party look identical to a broken one. Show a few, and a source/priority breakdown for
                    # EVERY action, every tick, so 'is an event actually reaching agents' is never ambiguous.
                    zone_actions = [a for a in actions if not a.get("smart_object_id")]
                    for a in zone_actions[:5]:
                        log.info(f"[{engine}] {a['agent_id']} -> {a['action']} @{a.get('zone_id')} "
                                 f"pt=({a.get('target_x')},{a.get('target_y')}) :: {a.get('reason','')} "
                                 f"[source={a.get('source')}]")
                    src_counts = {}
                    for a in actions:
                        s = a.get("source", "?")
                        src_counts[s] = src_counts.get(s, 0) + 1
                    log.info(f"[{engine}] tick {sim.dsag_scene.tick_no}: {len(actions)} actions "
                             f"({len(obj_actions)} object-targeted), {len(dsag_events)} object events, "
                             f"by source: {src_counts}")
                    continue

                # Pick up any event files dropped into EventLog/inbox before directing
                await process_inbox()
                log.info("Director tick...")
                try:
                    actions, events = await run_director_tick()
                    await send(json.dumps({
                        "type":    "actions",
                        "actions": actions,
                        "events":  events
                    }))
                    log.info(f"Sent {len(actions)} actions, {len(events)} events")
                except Exception as e:
                    log.error(f"Tick error: {e}")
                    await send(json.dumps({"type": "actions", "actions": [], "events": []}))

            elif mtype == "trigger_event":
                evt_type = msg.get("event_type", "announcement")
                evt_msg  = msg.get("message", "")
                evt = {"type": evt_type, "message": evt_msg}
                sim.active_event  = evt_type
                sim.event_message = evt_msg
                log.info(f"Active event set: {evt_type}")
                await send(json.dumps({
                    "type":    "actions",
                    "actions": [],
                    "events":  [evt]
                }))

            elif mtype == "describe_event":
                description = msg.get("description", "")
                scene_name  = msg.get("scene_name", sim.scene_name)
                # DSAG mode: a WebSocket event becomes ONE ScenePatch on the live graph (network path,
                # equivalent to the EventLog/inbox file path) — lets a distributable client run without
                # sharing the server's filesystem.
                if sim.director_mode == "dsag" and sim.dsag_scene is not None:
                    try:
                        if description.strip().lower() in ("all clear", "all-clear",
                                                           ALL_CLEAR_SENTINEL.strip().lower()):
                            sim.dsag_scene.active_patches = []
                            await send(json.dumps({"type": "ok", "message": "patches cleared"}))
                        else:
                            # capability-gate PRE-PASS (evaluation harness): support-status + grounded
                            # ops from the live scene; UNSUPPORTED/AMBIGUOUS/INVALID_REFERENCE are
                            # rejected with the reason instead of guessing a patch. ECGP_COMPILER_GATE=0
                            # disables rejection (verdict still logged for the episode trace).
                            from ecgp.runtime import compiler_gate
                            verdict = compiler_gate.precheck(description, sim.dsag_scene, sim.zones)
                            try:
                                from evaluation.unity_harness import episode_logger
                                episode_logger.log_instruction(description, verdict)
                            except Exception:
                                pass                        # logger is optional; never break the path
                            if verdict["gate"] == "reject":
                                await send(json.dumps({"type": "event_rejected",
                                                       "support_status": verdict["support_status"],
                                                       "reason": verdict.get("reason", "")}))
                                log.info(f"[ws] '{description[:50]}' REJECTED by capability gate: "
                                         f"{verdict['support_status']}")
                                continue
                            pd = await interpret_event_patch(description, sim.dsag_scene)
                            pd["support_status"] = verdict["support_status"]
                            pd["compiler_ops"] = verdict.get("operations", [])
                            if verdict["support_status"] == "PARTIALLY_EXECUTABLE":
                                pd["missing_capabilities"] = verdict.get("unresolved", [])
                            patch = ScenePatch.from_dict(pd)
                            if patch.is_valid() and patch.ops:
                                sim.dsag_scene.apply_patch(patch)
                            write_event_log(pd, source="ws/describe_event")
                            await send(json.dumps({"type": "event_interpreted", "dsag": True,
                                                   "display_name": pd.get("display_name", description)}))
                            log.info(f"[ws] '{description[:50]}' -> patch '{pd.get('display_name','?')}'")
                    except Exception as e:
                        log.error(f"describe_event(dsag) error: {e}")
                        await send(json.dumps({"type": "error", "message": str(e)}))
                    continue
                log.info(f"Interpreting event: {description}")
                try:
                    result = await interpret_event(description, scene_name)
                    result["type"] = "event_interpreted"

                    # Store the directive on the SERVER right now so the next director
                    # tick honors it (don't rely on Unity's update_state round-trip,
                    # which can arrive after the tick has already fired).
                    apply_event_result(description, result, source="ui_field")

                    await send(json.dumps(result))
                    log.info(f"Event interpreted: {result.get('display_name','?')} | "
                             f"directive: {result.get('directive', description)}")
                except Exception as e:
                    log.error(f"describe_event error: {e}")
                    await send(json.dumps({"type": "error", "message": str(e)}))

            elif mtype == "clear_event":
                clear_all_directives(source="clear_button")
                sim.events.append({
                    "type":    "all_clear",
                    "message": "Emergency over — reassign agents based on their Maslow tier."
                })
                await send(json.dumps({"type": "ok", "message": "event cleared"}))

            elif mtype == "ping":
                await send(json.dumps({"type": "pong"}))

    except websockets.exceptions.ConnectionClosed:
        log.info(f"Unity disconnected ({ws.remote_address})")

async def main():
    log.info(f"Crowd Director Server — ws://{HOST}:{PORT}")
    log.info(f"API key: {'set OK' if API_KEY else 'NOT SET — set ANTHROPIC_API_KEY env var'}")
    # Relaxed keepalive: Unity's render thread (25 agents via OnGUI) can be too busy to
    # answer pings within the 20s default, which was killing the connection. On localhost
    # we don't need aggressive liveness checks — give Unity 2 minutes before timing out.
    async with websockets.serve(
        handle_client, HOST, PORT,
        ping_interval=30,
        ping_timeout=120,
        max_size=64 * 1024 * 1024,  # large update_state frames for big crowds (thousands of agents)
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
