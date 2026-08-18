# CrowdDirector

Natural-language crowd direction for Unity. Describe a scene in a sentence, type instructions at it
while it runs — *"the coffee machine is broken"*, *"everyone leave"* — and a trained graph policy
decides what each agent does, every tick, on your machine.

You keep navigation, collision, reservations and animation. The add-on only decides **which target**
and **which action**, which is precisely what it was trained to do alongside deterministic execution
systems.

```csharp
public class MyAgent : MonoBehaviour, ICrowdAgent
{
    public string AgentId => name;
    public Vector2 Position => transform.position;
    public CrowdNeeds Needs => _needs;
    // ...

    public void DirectorMoveToZone(string zoneId, string reason) => _nav.SetDestination(ZoneCentre(zoneId));
    public void DirectorRest(string reason) => _anim.Play("Sit");
}
```

## How it is put together

```
Unity  ── ICrowdAgent state ──►  sidecar server  ──►  graph policy   (local, no API call)
       ◄──── per-agent actions ──                ──►  Claude        (scene setup + instructions only)
```

The **per-tick director makes no API call.** A 442k-parameter relational graph policy runs locally at
roughly 0.7 ms per agent decision. An `ANTHROPIC_API_KEY` is needed only for two authoring features:
generating a scene from a description, and interpreting free-text instructions. Without a key, a
scene you have already built still runs, and the deterministic instruction parser still handles the
common phrasings.

## Getting started

**1. Start the sidecar.**

```bash
cd server
./start.sh          # Windows: start.bat
```

First run creates a virtualenv and installs PyTorch (a few hundred MB, CPU-only, once). It then
serves on `ws://localhost:8765`. To use scene generation, set a key first:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # Windows: set ANTHROPIC_API_KEY=sk-ant-...
```

**2. Add the Unity package.** Window → Package Manager → **+** → *Add package from disk* → pick
`unity/package.json`.

**3. Wire it up.** Put `CrowdDirectorClient` on a GameObject, implement `ICrowdAgent` on your agents,
and register them:

```csharp
director.RegisterAgent(myAgent);
director.GenerateScene("a busy hospital waiting room at visiting hour");
director.DescribeEvent("the vending machine is out of order");
```

## What ships

```
server/     the director stack, extracted from the research server
  ecgp/       the graph policy, its runtime, and the capability compiler
  dsag/       smart objects, affordances, needs
  model/      full_graph_seed0/best.pt - the released checkpoint
  PATCHES.md  every divergence from the original, and why
unity/      the UPM client package
  Runtime/    ICrowdAgent, CrowdNeeds, CrowdDirectorClient
```

The server is an **extraction, not a rewrite** — the modules are byte-for-byte copies of the code
that produced the published results, with two path defaults changed so it runs outside a checkout.
[server/PATCHES.md](server/PATCHES.md) lists them. The bundled checkpoint is SHA-256 identical to the
one the paper reports.

## Honest limits

**A Python process must be running.** That is the trade this design makes. It cannot be shipped
inside a built game, so it suits editor-time work, research and prototyping — not a title you
release. A dependency-free C# port exists in prototype and reproduces the policy exactly, but it is
not finished.

**Desktop only.** `ClientWebSocket` rules out WebGL, and the sidecar rules out mobile and console.

**No art.** This is a director. Your agents keep their own visuals. The LPC character pipeline from
the research project is deliberately not included: it is director-irrelevant and its assets are
CC-BY-SA-3.0 / GPL-3.0, which would be inherited by anything shipping them.

**`PersonalityType`, not `AgentType`.** The scene generator renames `AgentType` freely between
generations. Key your own logic on `PersonalityType`, which comes from a fixed vocabulary.

**Needs are a closed set.** The nine fields on `CrowdNeeds` match `ecgp.vocab.NEEDS_V1` exactly.
Renaming one silently drops it from every decision rather than raising.
