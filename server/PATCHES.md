# Divergence from `python-server/`

This tree is extracted from the research server, not a rewrite. Everything is a byte-for-byte
copy except the two path defaults below, both of which assumed a checkout layout a release does
not have. Keeping the list this short is deliberate — it is what makes re-syncing with upstream a
diff rather than a merge.

### `crowd_director_server.py` — EventLog location

```diff
-EVENTLOG = Path(__file__).resolve().parent.parent / "CrowdSim" / "EventLog"
+EVENTLOG = Path(os.environ.get("CROWDDIRECTOR_EVENTLOG",
+                               Path(__file__).resolve().parent / "EventLog"))
```

The original resolves to a sibling `CrowdSim/` Unity project. A release has no such sibling, so it
now defaults to `server/EventLog` and honours `CROWDDIRECTOR_EVENTLOG`. Point that variable at the
old location to reproduce the original behaviour exactly.

### `ecgp/runtime/live_bridge.py` — default checkpoint

```diff
-                                                 "crowddirect_v3", "outputs", "cd_v3_seed0", "best.pt"))
+                                                 "model", "full_graph_seed0", "best.pt"))
```

The original points into `crowddirect_v3/outputs/`, a training output directory. The bundled
checkpoint is byte-identical — both are SHA-256 `d23bcb77…`, verified before extraction — so this
changes where the file is found, not which weights load. `ECGP_CKPT` still overrides.

### Not copied

`assetgen/` (9 modules — the LPC/Gemini art pipeline) is omitted. It is reached through a single
guarded import and the server already degrades with `[lpc] runtime character generation disabled`,
which is the expected log line here. It is left out because this is a director-only release and
because the LPC assets are CC-BY-SA-3.0 / GPL-3.0, which a permissively licensed add-on cannot
carry.
