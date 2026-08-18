"""
graph.py — the explicit Scene-Affordance Graph.

Until now the graph was implicit in behavior.py's reasoning. This makes it a first-class
object: typed nodes (agent/zone/prop/action/need/event) and typed edges (affords / satisfies
/ supports / contains / in / needs / affects). It is a *projection* of the SceneModel — the
SceneModel stays authoritative for state; the graph is the structured view rebuilt from it.

Enables the next steps:
  - subgraph()  -> local retrieval for event patches (SayPlan-style)
  - explain()   -> the decision provenance as a graph structure (traceability)
  - to_dict()   -> serialisation for visualisation and (Tier-1) GNN input
"""

from collections import deque
from dataclasses import dataclass, field
from .needs import top_need

# which afforded actions a zone offers, by its function (zone-level affordances)
ACTIONS_BY_ZONEFUNC = {
    "service": ["order", "drink", "eat"], "seating": ["sit"], "hygiene": ["relieve"],
    "activity": ["observe", "dance"], "access": ["leave"], "work": ["work"],
}


@dataclass
class Node:
    id: str
    ntype: str                       # agent | zone | prop | action | need | event
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    src: str
    rel: str
    dst: str


class SceneAffordanceGraph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._edge_set: set = set()          # dedupe (src, rel, dst)
        self._adj: dict[str, set] = {}       # undirected adjacency for traversal

    # ── construction ─────────────────────────────────────────────────────────
    def add_node(self, nid: str, ntype: str, **attrs) -> str:
        if nid in self.nodes:
            self.nodes[nid].attrs.update(attrs)
        else:
            self.nodes[nid] = Node(nid, ntype, dict(attrs))
            self._adj.setdefault(nid, set())
        return nid

    def add_edge(self, src: str, rel: str, dst: str):
        key = (src, rel, dst)
        if key in self._edge_set:
            return
        self._edge_set.add(key)
        self.edges.append(Edge(src, rel, dst))
        self._adj.setdefault(src, set()).add(dst)
        self._adj.setdefault(dst, set()).add(src)

    # ── queries ──────────────────────────────────────────────────────────────
    def neighbors(self, nid: str) -> set:
        return set(self._adj.get(nid, ()))

    def in_edges(self, nid: str, rel: str = None) -> list:
        return [e for e in self.edges if e.dst == nid and (rel is None or e.rel == rel)]

    def out_edges(self, nid: str, rel: str = None) -> list:
        return [e for e in self.edges if e.src == nid and (rel is None or e.rel == rel)]

    def subgraph(self, seeds, radius: int = 1) -> "SceneAffordanceGraph":
        """The r-hop neighbourhood around `seeds` — the local view an event patch touches."""
        seen = set(seeds)
        frontier = set(seeds)
        for _ in range(radius):
            nxt = set()
            for n in frontier:
                nxt |= self.neighbors(n)
            frontier = nxt - seen
            seen |= nxt
        g = SceneAffordanceGraph()
        for nid in seen:
            if nid in self.nodes:
                n = self.nodes[nid]
                g.add_node(nid, n.ntype, **n.attrs)
        for e in self.edges:
            if e.src in seen and e.dst in seen:
                g.add_edge(e.src, e.rel, e.dst)
        return g

    def path(self, src: str, dst: str, max_depth: int = 6):
        """Shortest node path between two nodes (undirected), or None."""
        if src not in self._adj or dst not in self._adj:
            return None
        q = deque([(src, [src])])
        seen = {src}
        while q:
            cur, p = q.popleft()
            if cur == dst:
                return p
            if len(p) > max_depth:
                continue
            for nb in self._adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, p + [nb]))
        return None

    def explain(self, agent_id: str) -> dict:
        """Traceability: agent's top need -> actions that satisfy it -> props/zones that
        afford those actions. The decision's causal chain as graph structure."""
        an = f"agent:{agent_id}"
        if an not in self.nodes:
            return {}
        need_field = self.nodes[an].attrs.get("top_need", "")
        options = []
        for e in self.in_edges(f"need:{need_field}", "satisfies"):     # action --satisfies--> need
            action = e.src
            providers = [pe.src for pe in self.in_edges(action, "affords")]   # props
            providers += [se.src for se in self.in_edges(action, "supports")]  # zones
            options.append({"action": self.nodes[action].attrs.get("name", action),
                            "providers": providers})
        return {"agent": self.nodes[an].attrs.get("name", agent_id),
                "top_need": need_field, "options": options}

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n.id, "ntype": n.ntype, "attrs": n.attrs} for n in self.nodes.values()],
            "edges": [{"src": e.src, "rel": e.rel, "dst": e.dst} for e in self.edges],
        }

    def stats(self) -> dict:
        by_type, by_rel = {}, {}
        for n in self.nodes.values():
            by_type[n.ntype] = by_type.get(n.ntype, 0) + 1
        for e in self.edges:
            by_rel[e.rel] = by_rel.get(e.rel, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "by_type": by_type, "by_rel": by_rel}

    # ── build from a SceneModel (the projection) ─────────────────────────────
    @classmethod
    def build(cls, scene) -> "SceneAffordanceGraph":
        g = cls()

        # zones + zone-level supported actions
        for z in scene.zones.values():
            zid = g.add_node(f"zone:{z.id}", "zone", label=z.label,
                             zone_type=z.zone_type, zone_function=z.zone_function)
            for fn in z.zone_function:
                for a in ACTIONS_BY_ZONEFUNC.get(fn, []):
                    g.add_edge(zid, "supports", g.add_node(f"action:{a}", "action", name=a))

        # props: contains, affords, and (action -> satisfies -> need) via need_effects
        for o in scene.objects.values():
            pid = g.add_node(f"prop:{o.id}", "prop", object_type=o.object_type,
                             state=o.state, zone=o.zone_id)
            if f"zone:{o.zone_id}" in g.nodes:
                g.add_edge(f"zone:{o.zone_id}", "contains", pid)
            for aff in o.affordances:
                an = g.add_node(f"action:{aff.action}", "action", name=aff.action)
                g.add_edge(pid, "affords", an)
                for fld, dv in aff.need_effects.items():
                    beneficial = dv > 0 if fld == "energy" else dv < 0
                    if beneficial:
                        g.add_edge(an, "satisfies", g.add_node(f"need:{fld}", "need", field=fld))

        # agents: in-zone + their pressing needs + the top need (for explain())
        tracked = ("thirst", "hunger", "bladder", "energy", "loneliness",
                   "stress", "curiosity", "status", "groupAffinity", "taskProgress")
        for a in scene.agents.values():
            tf, tp = top_need(a.needs)
            aid = g.add_node(f"agent:{a.id}", "agent", name=a.name, role=a.role,
                             zone=a.current_zone, top_need=tf, top_pressure=round(tp, 1))
            if f"zone:{a.current_zone}" in g.nodes:
                g.add_edge(aid, "in", f"zone:{a.current_zone}")
            for fld in tracked:
                if a.needs.pressure(fld) >= 40:      # only pressing needs, keeps the graph legible
                    g.add_edge(aid, "needs", g.add_node(f"need:{fld}", "need", field=fld))

        # active object events -> the zones/props they affect (dynamic layer)
        for t in scene.tasks:
            ev = g.add_node(f"event:{t['object']}", "event", etype=t.get("event", ""))
            if f"zone:{t['zone']}" in g.nodes:
                g.add_edge(ev, "affects", f"zone:{t['zone']}")
            if f"prop:{t['object']}" in g.nodes:
                g.add_edge(ev, "concerns", f"prop:{t['object']}")

        return g
