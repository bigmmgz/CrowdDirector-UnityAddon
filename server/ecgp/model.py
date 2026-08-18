"""
model.py — the ECGP neural policy (SCHEMA_ECGP.md rev. 3.1 §3–§5, §8).

Heterogeneous graph → per-agent distribution over JOINT decision-options:
  1. typed input projection: one Linear per node type (6) → shared hidden H;
  2. shared heterogeneous message passing over the 23 relations (mean-aggregated per relation,
     edge-weighted, self-loop + residual + LayerNorm);
  3. option pointer scorer: for each option, score [h_agent ‖ h_target ‖ h_agent*h_target ‖ opt_feat]
     — targets are REAL graph nodes (self = the agent's own node), never raw IDs, never a group node;
  4. hard feasibility masking (opt_feasible → -inf) BEFORE softmax;
  5. one predicted distribution over the agent's options (+ an optional urgency regression head).

No text embeddings; no group-output head (deferred). Consumes the encoder's enc dict directly, so a
single-scene enc (batch of 1) and a collated multi-scene enc (batch.py) use the same forward.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vocab as V

NEG_INF = -1e9


def masked_log_softmax(logits, feasible):
    """log-softmax over FEASIBLE options only; infeasible get -inf (→ prob exactly 0)."""
    mask = feasible > 0.5
    if not bool(mask.any()):
        mask = torch.ones_like(mask)                    # degenerate guard (never in practice: self.idle)
    logits = logits.masked_fill(~mask, NEG_INF)
    return F.log_softmax(logits, dim=0)


class RelationalLayer(nn.Module):
    """One heterogeneous message-passing round: per-relation linear, mean-aggregated to destinations,
    edge-weighted, plus a self transform; residual + LayerNorm."""
    def __init__(self, h, relations):
        super().__init__()
        self.self_lin = nn.Linear(h, h)
        self.rel = nn.ModuleDict({r: nn.Linear(h, h, bias=False) for r in relations})
        self.norm = nn.LayerNorm(h)
        self.h = h

    def forward(self, x, edges, n, edge_w):
        agg = self.self_lin(x)
        for r, ei in edges.items():
            if r not in self.rel or ei.shape[1] == 0:
                continue
            src, dst = ei[0], ei[1]
            msg = self.rel[r](x)[src]
            w = edge_w.get(r)
            if w is not None:
                msg = msg * w.unsqueeze(-1)
            summed = torch.zeros(n, self.h, device=x.device).index_add(0, dst, msg)
            deg = torch.zeros(n, device=x.device).index_add(
                0, dst, torch.ones(dst.shape[0], device=x.device))
            agg = agg + summed / deg.clamp(min=1.0).unsqueeze(-1)
        return self.norm(F.relu(agg) + x)


class ECGPNet(nn.Module):
    def __init__(self, hidden=V.HIDDEN, layers=2, relation_names=None, option_feat_dim=None):
        """`relation_names` and `option_feat_dim` default to the v1 values (V.RELATIONS_V1,
        V.OPTION_FEAT_DIM), so existing call sites and checkpoints are unaffected. The hierarchy-aware
        architecture passes `relation_names=V.RELATIONS_V2, option_feat_dim=V.OPTION_FEAT_DIM_V2`
        explicitly; `ecgp.v21_config.V21_MODEL_CONFIG` is that configuration."""
        super().__init__()
        self.h = hidden
        self.relation_names = list(relation_names) if relation_names is not None else list(V.RELATIONS_V1)
        self.option_feat_dim = option_feat_dim if option_feat_dim is not None else V.OPTION_FEAT_DIM
        self.node_types = list(V.NODE_TYPES)
        self.type_proj = nn.ModuleDict(
            {t: nn.Linear(V.NODE_FEATURE_DIM[t], hidden) for t in self.node_types})
        self.layers = nn.ModuleList([RelationalLayer(hidden, self.relation_names) for _ in range(layers)])
        # scorer input = [h_agent ‖ h_target ‖ h_agent*h_target ‖ opt_feat] = 3*hidden + option_feat_dim
        # (V.POINTER_INPUT_DIM == 215 at hidden=64 with the v1 option features; the expression also
        # covers capacity sweeps and the wider hierarchy-aware option tensor)
        scorer_in = 3 * hidden + self.option_feat_dim
        self.scorer = nn.Sequential(nn.Linear(scorer_in, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))
        self.urgency_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                          nn.Linear(hidden // 2, 1))

    # enc dict (numpy) → contextual node embeddings [N, H]
    def embed(self, enc, device):
        n = enc["n_nodes"]
        x = torch.zeros(n, self.h, device=device)
        globals_of_type = {t: [] for t in self.node_types}
        for g, (t, _l) in enumerate(enc["type_local"]):
            globals_of_type[t].append(g)
        for t in self.node_types:
            f = enc["feats"][t]
            if f.shape[0]:
                proj = self.type_proj[t](torch.as_tensor(f, dtype=torch.float32, device=device))
                idx = torch.as_tensor(globals_of_type[t], dtype=torch.long, device=device)
                x = x.index_copy(0, idx, proj)
        edges = {r: torch.as_tensor(ei, dtype=torch.long, device=device)
                 for r, ei in enc["edges"].items()}
        edge_w = {r: torch.as_tensor(w, dtype=torch.float32, device=device)
                  for r, w in enc.get("edge_w", {}).items()}
        for layer in self.layers:
            x = layer(x, edges, n, edge_w)
        return x

    def score_agent(self, x, a, device):
        """Return (logits[O], log_probs[O], urgency_scalar) for one agent record."""
        ha = x[int(a["agent_global"])]
        cg = torch.as_tensor(a["opt_target_global"], dtype=torch.long, device=device)
        hc = x[cg]                                                   # [O, H]
        hae = ha.unsqueeze(0).expand(hc.shape[0], -1)
        ofeat = torch.as_tensor(a["opt_feat"], dtype=torch.float32, device=device)
        inp = torch.cat([hae, hc, hae * hc, ofeat], dim=-1)         # [O, 3H+23]
        logits = self.scorer(inp).squeeze(-1)                       # [O]
        feas = torch.as_tensor(a["opt_feasible"], dtype=torch.float32, device=device)
        logp = masked_log_softmax(logits, feas)
        urg = self.urgency_head(ha).squeeze(-1)
        return logits, logp, urg

    def forward(self, enc, device="cpu"):
        x = self.embed(enc, device)
        return [(a, *self.score_agent(x, a, device)) for a in enc["agents"]]

    @torch.no_grad()
    def predict(self, enc, device="cpu"):
        """agent_id → (probs[O], argmax_index)."""
        out = {}
        for a, _logits, logp, _urg in self.forward(enc, device):
            p = logp.exp()
            out[a["agent_id"]] = (p.cpu().numpy(), int(p.argmax().item()))
        return out
