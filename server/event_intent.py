"""
event_intent.py — canonical event-intent resolver for the director's free-text interruptions.

Grounds a phrase to ONE canonical intent, then derives four INDEPENDENT fields (event_type,
global_directive, priority, is_emergency) from the intent — so e.g. "everyone go home" resolves to
end_of_day + global_directive=leave + is_emergency=FALSE, while a fire is evacuation + leave + emergency.

Resolution order (per design):
  1. Trust the LLM's structured `intent` if it is a valid canonical value.
  2. Otherwise embed the phrase with MiniLM and take the nearest curated intent prototype,
  3. gated by an absolute confidence threshold AND a nearest-vs-second-nearest margin.
  4. Only if that is inconclusive, a VERY small deterministic net for unmistakable danger words.
No large hard-coded synonym lists; the semantic stage does the work.
"""
import logging

log = logging.getLogger("event_intent")

# canonical intents → the four independent fields (is_emergency is NOT "any global leave")
CANONICAL = {
    "end_of_day":      dict(event_type="end_of_day",      global_directive="leave",  is_emergency=False, priority="high",     clears_emergency=False),
    "evacuation":      dict(event_type="evacuate",        global_directive="leave",  is_emergency=True,  priority="critical", clears_emergency=False),
    "hazard":          dict(event_type="hazard",          global_directive="avoid",  is_emergency=False, priority="high",     clears_emergency=False),
    "all_clear":       dict(event_type="all_clear",       global_directive="resume", is_emergency=False, priority="high",     clears_emergency=True),
    "object_disable":  dict(event_type="object_disable",  global_directive=None,     is_emergency=False, priority="normal",   clears_emergency=False),
    "object_enable":   dict(event_type="object_enable",   global_directive=None,     is_emergency=False, priority="normal",   clears_emergency=False),
    "block_edge":      dict(event_type="block_edge",      global_directive=None,     is_emergency=False, priority="normal",   clears_emergency=False),
    "agent_directive": dict(event_type="agent_directive", global_directive=None,     is_emergency=False, priority="normal",   clears_emergency=False),
    "group_directive": dict(event_type="group_directive", global_directive=None,     is_emergency=False, priority="normal",   clears_emergency=False),
    "announcement":    dict(event_type="announcement",    global_directive=None,     is_emergency=False, priority="low",      clears_emergency=False),
}

# curated intent PROTOTYPES for the semantic fallback (paraphrases, not a match list the code scans)
PROTOTYPES = {
    "end_of_day": ["closing time", "the building is closing for the day", "we're closing up for today",
                   "end of day, everyone head home", "the venue is closing now", "everybody go home",
                   "the shop is closed for today please leave", "we are done for the day"],
    "evacuation": ["evacuate immediately", "fire everyone out now", "emergency evacuation",
                   "gas leak get out of the building", "clear the building now", "danger leave right now",
                   "there is a fire, run"],
    "hazard": ["there is a spill on the floor", "wet floor be careful", "a fight broke out near the bar",
               "avoid the east wing", "broken glass over there", "something dangerous over there"],
    "all_clear": ["all clear", "it is safe now", "the emergency is over", "false alarm",
                  "you can go back inside", "everyone can relax now", "danger has passed"],
    "object_disable": ["the counter is closed", "the restroom is out of order", "that machine is broken",
                       "the bar is not serving right now", "the coffee machine is down", "this station is unavailable"],
    "object_enable": ["the counter is open again", "the restroom is back in service", "the bar is serving now",
                      "the machine is fixed and working"],
    "block_edge": ["close the emergency exit", "block the north corridor", "seal the door between the rooms",
                   "the passage is blocked", "shut the door to the hallway", "that doorway is closed off"],
    "agent_directive": ["Drew go to the bar", "send Robin to the exit", "Alex talk to Sam",
                        "the waiter should clean table two", "tell Jamie to sit down"],
    "group_directive": ["the family should regroup", "staff gather at the entrance",
                        "the tour group move together", "the team should split up"],
    "announcement": ["free cake at the counter", "live music is starting", "happy hour begins now",
                     "the show is about to start"],
}

# TINY deterministic net — only unmistakable safety-critical danger phrases (last resort if the LLM
# output is absent AND the semantic stage is unavailable/inconclusive).
_SAFETY = ("fire", "gas leak", "smoke", "bomb", "explosion", "evacuate", "active shooter")

_THRESHOLD = 0.40      # absolute cosine floor
_MARGIN = 0.04         # nearest must beat second-nearest by this

_model = None          # lazy MiniLM
_proto_emb = None       # {intent: np.ndarray[n, d]} normalized


def _get_model():
    global _model
    if _model is False:
        return None
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("[intent] MiniLM loaded for semantic event-intent grounding")
        except Exception as e:
            log.warning(f"[intent] MiniLM unavailable ({e}); using LLM + safety net only")
            _model = False
            return None
    return _model


def _proto_embeddings(model):
    global _proto_emb
    if _proto_emb is None:
        import numpy as np
        _proto_emb = {}
        for intent, phrases in PROTOTYPES.items():
            emb = model.encode(phrases, normalize_embeddings=True)
            _proto_emb[intent] = np.asarray(emb, dtype=np.float32)
    return _proto_emb


def _semantic(text):
    """Nearest curated intent by max cosine, gated by absolute threshold + nearest/second margin."""
    model = _get_model()
    if model is None:
        return None
    import numpy as np
    q = np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
    scores = []
    for intent, emb in _proto_embeddings(model).items():
        scores.append((float((emb @ q).max()), intent))     # best-matching prototype for this intent
    scores.sort(reverse=True)
    (best, intent), (second, _2) = scores[0], scores[1]
    if best < _THRESHOLD or (best - second) < _MARGIN:
        log.info(f"[intent] semantic inconclusive for {text!r}: {scores[:3]}")
        return None
    return {**CANONICAL[intent], "intent": intent, "confidence": round(best, 3),
            "margin": round(best - second, 3), "source": "minilm"}


def resolve(text: str, llm_intent: str = None) -> dict:
    """Return the canonical intent record for a director phrase (+ optional LLM-provided intent)."""
    if llm_intent and llm_intent in CANONICAL:                # 1-2. trust a valid structured LLM intent
        return {**CANONICAL[llm_intent], "intent": llm_intent, "confidence": 1.0, "source": "llm"}
    res = _semantic(text or "")                               # 3-4. MiniLM semantic fallback
    if res is not None:
        return res
    low = (text or "").lower()                                # 5. tiny deterministic safety net
    if any(k in low for k in _SAFETY):
        return {**CANONICAL["evacuation"], "intent": "evacuation", "confidence": 1.0, "source": "safety"}
    return {**CANONICAL["announcement"], "intent": "announcement", "confidence": 0.0, "source": "default"}
