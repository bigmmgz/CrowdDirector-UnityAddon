# Minimal Crowd

A complete runnable demo in two scripts. Capsules, no art — it shows the contract, not a look.

1. Start the server: `server/start.bat` (Windows) or `server/start.sh`. **No API key needed** — the
   demo loads a prebuilt scene rather than designing one.
2. New empty scene → empty GameObject → add **Minimal Crowd Demo**.
3. Press Play.

The camera, the client, the zone floors and the 28 agents are all built at runtime, so there is
nothing to wire up and no scene file to open.

## What to look at

`MinimalCrowdAgent` is the smallest useful `ICrowdAgent`: it reports its position, zone and needs,
and walks where it is told. Replace `Update()` with your own NavMesh or steering and everything else
carries over.

`MinimalCrowdDemo` is the factory. The server mints agent ids when a scene loads, so agents are
created from the spec — note that `Bind()` makes the agent report `spec.Id` as its `AgentId`. An
agent reporting anything else is never directed.

## Recording a demo

Use `prebuiltScene = "demo"` rather than `description`. It loads the same floor plan every time, so
takes are comparable, and it costs nothing per run.

Try, in order: let it settle for ten seconds, then **Fire alarm** (evacuation is deterministic and
immediate), then **All clear**, then an instruction like *"the coffee machine is broken"* and watch
the drinkers reroute.

Set `logActions` on the client to print each decision with the policy's own stated reason.
