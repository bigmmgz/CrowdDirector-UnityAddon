"""
dsag — Dynamic Scene-Affordance Graph core for LLM-directed crowd simulation.

A dependency-pure package (stdlib only): no websockets, no LLM client, no engine
bindings. The WebSocket server and any future ML code import from here; the core
stays importable, testable, and open-sourceable on its own.

Layers (built incrementally):
  vocabulary   — controlled tag/enum dictionaries (the ontology)
  needs        — SDT/Maslow-grounded agent need model
  smart_object — objects as need/affordance entities with LLM-generated policies
  graph        — the Scene-Affordance Graph (nodes/edges)            [next]
  behavior     — affordance-constrained per-tick action scorer       [next]
  patch        — event patches from "what's happening"               [next]
  trace        — decision provenance logging                         [next]
"""

__version__ = "0.1.0"

from . import vocabulary
from .needs import Needs, NEED_GROUPS, urgent_tier, top_need
from .smart_object import Affordance, PolicyRule, SmartObject, Event
from .world import AgentInstance, ZoneInstance
from .behavior import Decision, choose_action
from .scene import SceneModel
from .trace import TraceLog, TraceRecord
from .graph import SceneAffordanceGraph, Node, Edge

__all__ = [
    "vocabulary",
    "Needs", "NEED_GROUPS", "urgent_tier", "top_need",
    "Affordance", "PolicyRule", "SmartObject", "Event",
    "AgentInstance", "ZoneInstance",
    "Decision", "choose_action",
    "SceneModel", "TraceLog", "TraceRecord",
    "SceneAffordanceGraph", "Node", "Edge",
]
