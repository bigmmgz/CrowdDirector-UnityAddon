# CrowdDirector

Natural-language crowd direction for Unity, powered by a trained relational graph policy.

![Unity](https://img.shields.io/badge/Unity-2021.3%2B-black?logo=unity&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

Describe a space in a sentence and CrowdDirector lays out its zones and populates it. While it runs,
type instructions at the crowd — *"the coffee machine is broken"*, *"staff leave first"* — and every
agent responds according to its own needs, role and relationships.

A 442,000-parameter graph policy makes the per-agent decisions locally, at roughly **0.7 ms per agent
decision** with **no API call per tick**. Your navigation, collision handling and animation stay
exactly as they are: CrowdDirector chooses *what* each agent should do and *where*, and leaves the
execution to you.

---

## Features

- **Instruction-driven** — plain-English direction compiled against the live scene into a persistent
  directive, not a one-shot prompt.
- **Needs-driven agents** — a nine-value need model (hunger, thirst, bladder, energy, stress,
  loneliness, group affinity, status, curiosity) with Maslow-style tier ordering.
- **Social behaviour** — per-pair familiarity, trust and tension evolve into acquaintance, friendship
  or avoidance, and feed back into who talks to whom.
- **Emergencies** — evacuation is deterministic and instant, bypassing the language layer entirely.
- **Scene generation** — zones, agent types, personalities and starting needs from one sentence.
- **Scales** — decision cost is linear in agent count and independent of scene complexity.
- **Engine-agnostic execution** — implement one interface and keep your own movement stack.

---

## Requirements

| | |
|---|---|
| Unity | 2021.3 or later |
| Python | 3.10 or later, for the director server |
| Platforms | Windows, macOS, Linux — Editor and standalone |
| Dependencies | `com.unity.nuget.newtonsoft-json` (resolved automatically) |

An `ANTHROPIC_API_KEY` enables scene generation and free-text instructions. The per-agent director
runs locally and does not use it.

---

## Installation

### 1. Install the Unity package

**Package Manager → + → Add package from git URL:**

```
https://github.com/bigmmgz/CrowdDirector-UnityAddon.git?path=/unity
```

Or add it to `Packages/manifest.json` directly:

```json
{
  "dependencies": {
    "com.crowddirector.client": "https://github.com/bigmmgz/CrowdDirector-UnityAddon.git?path=/unity"
  }
}
```

### 2. Start the director server

```bash
git clone https://github.com/bigmmgz/CrowdDirector-UnityAddon.git
cd CrowdDirector-UnityAddon/server

export ANTHROPIC_API_KEY=sk-ant-...     # Windows: set ANTHROPIC_API_KEY=sk-ant-...
./start.sh                              # Windows: start.bat
```

The first run creates a virtual environment and installs PyTorch. It then serves on
`ws://localhost:8765`, which is where the Unity client connects by default.

---

## Quick start

Add a **Crowd Director Client** component to any GameObject, then implement `ICrowdAgent` on your
agents:

```csharp
using UnityEngine;
using UnityEngine.AI;
using CrowdDirector;

public class Villager : MonoBehaviour, ICrowdAgent
{
    public string AgentId         => _id;
    public string AgentName       => _name;
    public string AgentType       => "visitor";
    public string PersonalityType => "casual_young";

    public Vector2   Position      => transform.position;
    public string    CurrentZoneId => _zone;
    public CrowdNeeds Needs        => _needs;

    public void DirectorMoveToZone(string zoneId, string reason)
        => _agent.SetDestination(CrowdScene.ZoneCentre(zoneId));

    public void DirectorRest(string reason)  => _animator.Play("Sit");
    public void DirectorIdle(string reason)  => _agent.ResetPath();
    // ...
}
```

Register them and direct the crowd:

```csharp
var director = GetComponent<CrowdDirectorClient>();

foreach (var villager in FindObjectsOfType<Villager>())
    director.RegisterAgent(villager);

director.GenerateScene("a busy hospital waiting room at visiting hour");
director.DescribeEvent("the vending machine is out of order");
director.TriggerEvent("fire_alarm", "Smoke in the east wing");
```

---

## How it works

```
Unity                          Director server                    Policy
─────                          ───────────────                    ──────
ICrowdAgent state  ──────────►  scene graph        ──────────►  graph policy
                                                                (local, per tick)
per-agent actions  ◄──────────  target + action    ◄──────────
```

Each tick, the client reports agent positions, needs and relationships. The server rebuilds a
heterogeneous scene graph — agents, zones, objects, groups and active events — enumerates the
candidate `(target, action)` pairs available to each agent, and the policy scores them. Structurally
impossible options are masked before scoring, so an agent is never told to use an object that is
full, disabled or unreachable.

Instructions take a separate path. They are compiled against the live scene's actual capabilities
into a persistent directive, so an instruction referring to something the scene does not contain is
reported as unsupported rather than silently approximated. Once compiled, the directive persists
across ticks until cleared.

---

## API reference

### `CrowdDirectorClient`

| Member | Description |
|---|---|
| `RegisterAgent(ICrowdAgent)` | Add an agent to the directed crowd |
| `UnregisterAgent(ICrowdAgent)` | Remove an agent and notify the server |
| `GenerateScene(string)` | Build zones and agent types from a description |
| `DescribeEvent(string)` | Issue a free-text instruction |
| `TriggerEvent(string, string)` | `fire_alarm`, `closing_time`, `music_starts`, `new_arrival`, `all_clear` |
| `ClearEvent()` | Cancel the active event and any standing orders |
| `Connect()` / `Disconnect()` | Manage the connection manually |
| `IsConnected`, `IsSceneReady`, `AgentCount` | State |

**Inspector:** `serverUrl`, `autoConnect`, `tickInterval` (default 3 s), `stateInterval`
(default 1.5 s), `logActions`.

**Events:** `Connected`, `Disconnected`, `SceneReady`, `ActionsReceived`, `ServerError`.

### `ICrowdAgent`

| Member | Description |
|---|---|
| `AgentId` | Stable unique identifier — required |
| `AgentName` | Display name |
| `AgentType` | Scene-specific role, e.g. `"barista"` |
| `PersonalityType` | Fixed-vocabulary type, e.g. `"solo_worker"` |
| `Position`, `CurrentZoneId`, `Needs` | Live state, reported each interval |
| `EncounterCounts`, `Relationships`, `Friends` | Social state — return `null` to opt out |
| `DirectorMoveToZone(zoneId, reason)` | Walk to a zone |
| `DirectorGroupMove(zoneId, reason)` | Move as a group |
| `DirectorStartConversation(targetId, reason)` | Approach and talk |
| `DirectorRest(reason)` / `DirectorIdle(reason)` | Rest, or stand down |

`AgentType` is regenerated for every scene; key your own logic on `PersonalityType`, which comes from
a fixed vocabulary.

### `CrowdNeeds`

Nine `float` fields on a 0–100 scale — `hunger`, `thirst`, `bladder`, `energy`, `stress`,
`loneliness`, `groupAffinity`, `status`, `curiosity`. High means pressing, except `energy`, where low
means tired. `CrowdNeeds.Default` gives a reasonable starting point.

---

## Configuration

The server reads its settings from the environment:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables scene generation and free-text instructions |
| `ECGP_CKPT` | bundled | Path to the policy checkpoint |
| `ECGP_TEMP` | `1.0` | Sampling temperature; lower is greedier |
| `ECGP_SAMPLE` | `1` | Set `0` for deterministic arg-max selection |
| `ECGP_AMBIENT` | `1` | Ambient world events (spills, deliveries, arrivals) |
| `CROWDDIRECTOR_EVENTLOG` | `server/EventLog` | Audit trail and file-drop event inbox |

---

## Repository layout

```
server/          director server and the trained policy
  ecgp/            graph policy, runtime and instruction compiler
  dsag/            smart objects, affordances, needs
  model/           the released checkpoint
unity/           Unity client package (UPM)
  Runtime/         ICrowdAgent, CrowdNeeds, CrowdDirectorClient
```
