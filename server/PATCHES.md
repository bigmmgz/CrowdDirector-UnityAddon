# Divergence from the research server

This tree is extracted from the research server, not a rewrite. Everything is a byte-for-byte copy
except the changes below, each of which existed because the original assumed a checkout layout that a
standalone release does not have. Keeping this list short and explicit is what makes re-syncing with
upstream a diff rather than a merge.

### `crowd_director_server.py` — EventLog location

```diff
-EVENTLOG = Path(__file__).resolve().parent.parent / "CrowdSim" / "EventLog"
+EVENTLOG = Path(os.environ.get("CROWDDIRECTOR_EVENTLOG",
+                               Path(__file__).resolve().parent / "EventLog"))
```

The original resolves to a sibling `CrowdSim/` Unity project. It now defaults to `server/EventLog` and
honours `CROWDDIRECTOR_EVENTLOG`; point that at the old location to reproduce the original exactly.

### `policy/runtime/live_bridge.py` — default checkpoint

```diff
-                                                 "crowddirect_v3", "outputs", "cd_v3_seed0", "best.pt"))
+                                                 "model", "full_graph_seed0", "best.pt"))
```

The original points into a training output directory. The bundled checkpoint is byte-identical — both
are SHA-256 `d23bcb77…`, verified before extraction — so this changes where the file is found, not
which weights load. `ECGP_CKPT` still overrides.

### `scene_spec.py` — smart-object catalog location

```diff
-_ASSETS = os.path.join(os.path.dirname(__file__), "..", "CrowdSim", "Assets", "StreamingAssets", "AssetDataset")
+_ASSETS = os.environ.get("CROWDDIRECTOR_ASSETS",
+                         os.path.join(os.path.dirname(__file__), "assets"))
```

The original reads the catalog out of the Unity project's `StreamingAssets`. Without a sibling
`CrowdSim/` the path silently resolved to nothing and the affordance index loaded **zero** objects,
which leaves the director with nothing to send agents to. `server/assets/` now carries
`smart_object_catalog_limezu.json` (45 smart objects, 18 affordances). Override with
`CROWDDIRECTOR_ASSETS`; select a different set with `ECGP_PROP_SET`.

### Not included

**`assetgen/`** (9 modules — the LPC/Gemini art pipeline). Reached through a single guarded import;
the server already degrades with `[lpc] runtime character generation disabled`, which is the expected
log line here. Left out because this is a director-only release and because the LPC assets are
CC-BY-SA-3.0 / GPL-3.0, obligations that would pass to anything redistributing them.

**`main.py`**. A research entry point that executed `scripts/infer.py`, which is not part of the
director path and does not ship. The server entry point is `crowd_director_server.py`, which
`start.sh` and `start.bat` invoke.
