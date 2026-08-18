"""
encode.py — the unified ECGP heterogeneous encoder (SCHEMA_ECGP.md rev. 3.1 §1–§5, §11).

EcgpWorld (effective state) → {node feats per type, typed edges + inverses, per-agent joint options}.
Numeric only (no text embeddings). Self/continue options point at the agent's OWN node; groups are
never an option target. All spatial features are normalized by the per-scene scale S = max(nav-bbox
diagonal, EPS_SCALE). Output shapes are asserted by tests/test_roundtrip.py against §11.
"""
import math
import numpy as np

from .. import vocab as V
from .options import build_options
from ..teacher import need_pressure as NP
from .. import social as cs


def _emotion(a):
    return [(a.valence + 1.0) / 2.0, a.arousal, a.panic / 100.0]


def _density_by_agent(agents, r):
    """Local density for every agent in one grid-hash pass, O(N·k) rather than the O(N^2) that calling a
    per-agent brute-force all-pairs distance would cost. The quantity is identical (same radius, same
    exclude-self, same min(1, n/8) squash); test_scale_density_matches_brute_force proves the
    equivalence. Same pattern as the "nearby" edge pass below."""
    out = {a.id: 0.0 for a in agents}
    if len(agents) < 2 or r <= 1e-9:
        return out
    grid = {}
    for a in agents:
        grid.setdefault((int(a.pos[0] // r), int(a.pos[1] // r)), []).append(a)
    for a in agents:
        cx, cy = int(a.pos[0] // r), int(a.pos[1] // r)
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for b in grid.get((cx + dx, cy + dy), ()):
                    if b.id != a.id and math.hypot(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]) <= r:
                        n += 1
        out[a.id] = min(1.0, n / 8.0)
    return out


# Proximity radii are fractions of the scene scale S. On larger evaluation worlds a scale-relative
# radius would make per-agent neighbourhoods grow with the environment, giving O(N^2) proximity edges
# (871 nearby edges per agent at a crowd of 5000). The scale used for PROXIMITY ONLY is therefore capped
# at the training extent's nav-bbox diagonal. Every fixed-extent world has S < S_PROX_CAP, so the cap is
# a bit-identical no-op there (asserted by tests/test_prox_cap_freeze.py) and engages only on larger
# worlds. Feature normalization — positions, path costs, group distances — still uses the true S.
S_PROX_CAP = math.hypot(16.0, 10.0)


def encode(world, hierarchy=False, option_feat_v2=False) -> dict:
    S = world.scene_scale()
    Sp = min(S, S_PROX_CAP)                  # proximity-only scale (see S_PROX_CAP above)
    active = bool(world.overlays.active_patch_ids())
    global_of, types, type_local = {}, [], []
    feats = {t: [] for t in V.NODE_TYPES}
    node_records = []

    def add(nid, ntype, feat, rec=None):
        gid = len(types)
        global_of[nid] = gid
        types.append(ntype)
        type_local.append((ntype, len(feats[ntype])))
        feats[ntype].append(np.asarray(feat, np.float32))
        node_records.append({"id": nid, "type": ntype, **(rec or {})})
        return gid

    # ── agent nodes ──
    density_by_agent = _density_by_agent(list(world.agents.values()), Sp * 0.15)
    for a in world.agents.values():
        g = world.groups.get(a.group_id)
        if g and g.members:
            mem = [world.agents[m] for m in g.members if m in world.agents]
            cx = sum(m.pos[0] for m in mem) / len(mem); cy = sum(m.pos[1] for m in mem) / len(mem)
            dc = math.hypot(a.pos[0] - cx, a.pos[1] - cy) / S
            leader = world.agents.get(g.leader_id)
            dl = (math.hypot(a.pos[0] - leader.pos[0], a.pos[1] - leader.pos[1]) / S) if leader else 0.0
            has_g = 1.0
        else:
            dc = dl = 0.0; has_g = 0.0
        dens = density_by_agent[a.id]
        f = ([NP.pressure(n, a.needs[n]) for n in V.NEEDS_V1]     # correct per-need pressure (item A)
             + [a.ocean[o] for o in V.OCEAN_V1]
             + _emotion(a)
             + V.onehot(a.social_role, V.SOCIAL_ROLES_V1)
             + V.onehot(a.group_role, V.GROUP_ROLES_V1)
             + [dens, dens, 0.0]                                  # local_density, congestion, collision_risk
             + [min(1.0, dc), min(1.0, dl), has_g]
             + V.onehot(a.last_action, V.ACTIONS_V1)
             + [1.0 if a.is_staff else 0.0, 1.0 if active else 0.0])
        add(f"agent:{a.id}", "agent", f, {"name": a.name})

    # ── zone nodes ──
    for z in world.zones.values():
        has_dis = any(k[0] == z.id and world.overlays.is_disabled(k[0], k[1])
                      for k in world.overlays._disabled)
        has_blk = any(world.overlays.is_blocked_edge(z.id, n) for n in z.adjacency)
        is_haz = world.overlays.is_hazard(z.id)
        is_tgt = world.overlays.attraction(z.id) != 0.0 or is_haz
        f = (V.onehot(z.zone_function, V.ZONE_FUNCTIONS_V1)
             + [0.0, 1.0 if has_dis else 0.0, 1.0 if has_blk else 0.0,
                1.0 if is_haz else 0.0, 1.0 if is_tgt else 0.0])
        add(f"zone:{z.id}", "zone", f, {"zone_function": z.zone_function})

    # ── object nodes ──
    for o in world.objects.values():
        st = world.effective_object_state(o)
        cap = max(1, o.capacity)
        f = (V.onehot(world.object_class(o), V.OBJECT_CLASS_V1)
             + V.multihot(world.object_affordances(o), V.AFFORDANCES_V1)
             + V.onehot(st if st in V.OBJECT_STATES_V1 else "UNK", V.OBJECT_STATES_V1)
             + [1.0 if o.capacity_ok() else 0.0, o.occupancy / cap,
                cap / V.CAP_MAX, min(1.0, o.usage_count / 10.0)])
        add(f"object:{o.id}", "object", f, {"zone": o.zone_id})

    # ── group nodes ──
    for g in world.groups.values():
        f = (V.onehot(g.group_type, V.GROUP_TYPES_V1)
             + [g.cohesion, g.follow_strength, g.group_stress / 100.0,
                min(1.0, len(g.members) / 10.0), 1.0 if g.group_goal else 0.0])
        add(f"group:{g.id}", "group", f, {})

    # ── event nodes ──
    for e in world.events.values():
        if e.id not in world.overlays.patches:
            continue
        f = [e.ttl / V.TTL_MAX, 1.0 if e.is_evacuation else 0.0, e.severity, len(e.op_ids) / V.OPS_MAX]
        add(f"event:{e.id}", "event", f, {})

    # ── operation nodes ──
    for op in world.operations.values():
        if op.event_id not in world.overlays.patches:
            continue
        f = (V.onehot(op.op_kind, V.EVENT_OP_KINDS_V1)
             + [max(-1.0, min(1.0, op.magnitude / 100.0)), op.severity,
                world.overlays.patches[op.event_id]["ttl"] / V.TTL_MAX, 1.0 if op.active else 0.0])
        add(f"op:{op.id}", "operation", f, {})

    # ── edges (typed, with inverses) ──
    edge_idx = {r: [] for r in V.RELATIONS_V1}
    edge_w = {r: [] for r in V.RELATIONS_V1}

    def edge(rel, src, dst, w=1.0):
        if src in global_of and dst in global_of:
            edge_idx[rel].append((global_of[src], global_of[dst]))
            edge_w[rel].append(float(w))

    def edge_pair(rel, inv, src, dst, w=1.0):
        edge(rel, src, dst, w); edge(inv, dst, src, w)

    for a in world.agents.values():
        edge_pair("located_in", "contains_agent", f"agent:{a.id}", f"zone:{a.zone}")
        if a.group_id:
            edge_pair("member_of", "has_member", f"agent:{a.id}", f"group:{a.group_id}")
    for o in world.objects.values():
        edge_pair("object_in_zone", "contains_object", f"object:{o.id}", f"zone:{o.zone_id}")
    # ── hierarchy relations, opt-in via hierarchy=True. Off by default so the v1 schema (exactly 23
    # relation keys, pinned by test_roundtrip.py) is undisturbed for existing callers and checkpoints.
    # PART_OF/HAS_PART comes from the `parent_id` grounding on SmartObject; NEAR is computed from object
    # positions alone. SUPPORTS/ON and PROVIDES/PROVIDED_BY are wired but stay empty until the grounding
    # pass sets `supports_object_id`/`provider_object_id` — a known gap.
    if hierarchy:
        for r in V.HIERARCHY_RELATIONS_V2:
            edge_idx.setdefault(r, []); edge_w.setdefault(r, [])
        objs = list(world.objects.values())
        for o in objs:
            parent = getattr(o, "parent_id", None)
            if parent and parent in world.objects:
                edge_pair("part_of", "has_part", f"object:{o.id}", f"object:{parent}")
            supports = getattr(o, "supports_object_id", None)      # not yet populated by any grounding pass
            if supports and supports in world.objects:
                edge_pair("on", "has_on", f"object:{supports}", f"object:{o.id}")
            provider = getattr(o, "provider_object_id", None)      # not yet populated by any grounding pass
            if provider and provider in world.objects:
                edge_pair("provided_by", "provides", f"object:{o.id}", f"object:{provider}")
        NEAR_R = 0.6 * Sp
        for i, oa in enumerate(objs):
            for ob in objs[i + 1:]:
                if oa.zone_id != ob.zone_id:
                    continue
                d = math.hypot(oa.pos[0] - ob.pos[0], oa.pos[1] - ob.pos[1])
                if d <= NEAR_R:
                    edge("near", f"object:{oa.id}", f"object:{ob.id}")
                    edge("near", f"object:{ob.id}", f"object:{oa.id}")
    for g in world.groups.values():
        if g.leader_id:
            for m in g.members:
                if m != g.leader_id:
                    edge_pair("follows", "followed_by", f"agent:{m}", f"agent:{g.leader_id}",
                              g.follow_strength)
        if g.group_goal:
            edge_pair("group_goal", "goal_of_group", f"group:{g.id}", f"zone:{g.group_goal}", g.cohesion)
    for e in world.events.values():
        if e.id not in world.overlays.patches:
            continue
        for opid in e.op_ids:
            op = world.operations.get(opid)
            if not op:
                continue
            edge_pair("has_operation", "operation_of", f"event:{e.id}", f"op:{op.id}")
            tt = op.target_type
            if tt == "zone":
                edge_pair("event_affects_zone", "zone_affected_by_event", f"op:{op.id}",
                          f"zone:{op.target_id}", op.magnitude / 100.0)
            elif tt == "object":
                edge_pair("event_affects_object", "object_affected_by_event", f"op:{op.id}",
                          f"object:{op.target_id}", op.magnitude / 100.0)
            elif tt == "agent":
                edge_pair("event_affects_agent", "agent_affected_by_event", f"op:{op.id}",
                          f"agent:{op.target_id}", op.magnitude / 100.0)
            elif tt == "group":
                edge_pair("event_affects_group", "group_affected_by_event", f"op:{op.id}",
                          f"group:{op.target_id}", op.magnitude / 100.0)

    # nearby (proximity) + social_relation (meaningful ties), both bidirectional. Built in O(N·k) from a
    # spatial hash plus one pass over the sparse relationship table rather than an O(N^2) all-pairs loop,
    # so the encoder scales to large crowds. The edge set is identical to the brute-force version:
    # social_relation for every meaningful relationship at any distance, nearby for every other
    # within-radius pair.
    agents = list(world.agents.values())
    R = 0.2 * Sp
    related = set()                                      # meaningful pairs → social_relation (skip in nearby)
    for rel in world.relationships.values():
        if not (rel.persistent or rel.relationship_type != "stranger" or rel.affinity >= 0.15):
            continue                                     # non-meaningful ties fall through to nearby, as before
        a_id, b_id = rel.agent_a, rel.agent_b
        if a_id in world.agents and b_id in world.agents:
            related.add((a_id, b_id) if a_id <= b_id else (b_id, a_id))
            for s, d in ((a_id, b_id), (b_id, a_id)):
                edge("social_relation", f"agent:{s}", f"agent:{d}", rel.affinity)
    if R > 1e-9:                                          # nearby via a grid hash: only same/adjacent cells
        grid = {}
        for a in agents:
            grid.setdefault((int(a.pos[0] // R), int(a.pos[1] // R)), []).append(a)
        for a in agents:
            cx, cy = int(a.pos[0] // R), int(a.pos[1] // R)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for b in grid.get((cx + dx, cy + dy), ()):
                        if b.id <= a.id:                 # visit each unordered pair once
                            continue
                        key = (a.id, b.id) if a.id <= b.id else (b.id, a.id)
                        if key in related:               # meaningful pair already got social_relation
                            continue
                        dist = math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1])
                        if dist <= R:
                            prox = 1.0 - dist / (R + 1e-9)
                            for s, d in ((a, b), (b, a)):
                                edge("nearby", f"agent:{s.id}", f"agent:{d.id}", prox)
    for z in world.zones.values():
        for n in z.adjacency:
            if n in world.zones and not world.overlays.is_blocked_edge(z.id, n):
                edge("zone_adjacent", f"zone:{z.id}", f"zone:{n}")

    # ── per-agent joint options ──
    agents_enc = []
    for a in world.agents.values():
        opts = build_options(world, a)
        tg, tt, act, ofeat, feas = [], [], [], [], []
        for o in opts:
            nid = (f"agent:{a.id}" if o.target_type == "self"
                   else f"{o.target_type}:{o.target_id}")
            o.target_global = global_of.get(nid, global_of[f"agent:{a.id}"])
            tg.append(o.target_global)
            tt.append(V.TARGET_TYPES_V1.index(o.target_type))
            act.append(V.ACTIONS_V1.index(o.action) if o.action in V.ACTIONS_V1
                       else V.ACTIONS_V1.index("UNK"))
            pnorm = min(o.path_distance / (V.PATH_FACTOR * S), 1.0) if o.path_distance >= 0 else -1.0
            enorm = min(o.eta / (V.PATH_FACTOR * S / V.NOMINAL_SPEED), 1.0) if o.eta >= 0 else -1.0
            struct = [o.feasible, o.nav_reachable, pnorm, enorm, o.action_compatible,
                      o.event_relevant, o.capacity_ok, o.same_group, o.affinity]
            if option_feat_v2:
                # Append the three extra Option fields options.py populates (functional_role one-hot over
                # OBJECT_ROLE_V2, expected_need_reduction, hierarchy_complete). Off by default so the v1
                # tensor width (OPTION_FEAT_DIM = 23) is unchanged for existing checkpoints; a caller opts
                # in explicitly, the same convention as `hierarchy=`.
                struct = struct + V.onehot(getattr(o, "functional_role", "UNK"), V.OBJECT_ROLE_V2) + [
                    getattr(o, "expected_need_reduction", 0.0), getattr(o, "hierarchy_complete", 1.0)]
            ofeat.append(struct + V.onehot(o.action, V.ACTIONS_V1))
            feas.append(o.feasible)
        dim = V.OPTION_FEAT_DIM_V2 if option_feat_v2 else V.OPTION_FEAT_DIM
        agents_enc.append({
            "agent_id": a.id, "agent_global": global_of[f"agent:{a.id}"],
            "opt_target_global": np.asarray(tg, np.int64),
            "opt_target_type": np.asarray(tt, np.int64),
            "opt_action": np.asarray(act, np.int64),
            "opt_feat": np.asarray(ofeat, np.float32).reshape(len(opts), dim),
            "opt_feasible": np.asarray(feas, np.float32),
            "options": opts,
        })

    return {
        "scene_scale": S,
        "n_nodes": len(types), "types": types, "type_local": type_local, "global_of": global_of,
        "feats": {t: (np.stack(v) if v else np.zeros((0, V.NODE_FEATURE_DIM[t]), np.float32))
                  for t, v in feats.items()},
        "edges": {r: (np.asarray(p, np.int64).T if p else np.zeros((2, 0), np.int64))
                  for r, p in edge_idx.items()},
        "edge_w": {r: np.asarray(w, np.float32) for r, w in edge_w.items()},
        "agents": agents_enc, "node_records": node_records,
    }
