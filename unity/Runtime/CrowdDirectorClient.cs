// CrowdDirectorClient.cs — the Unity end of the CrowdDirector sidecar.
//
// Drop this on one GameObject, register your ICrowdAgent implementations, and it keeps the local
// server informed of scene state and applies the decisions that come back. The policy itself runs
// in the sidecar; nothing here makes an API call.
//
// Two details are load-bearing, and both were bugs before they were fixes:
//
//   * ClientWebSocket permits exactly ONE in-flight SendAsync. A button-triggered send colliding
//     with the periodic state update is silently dropped, not raised. Every send is therefore
//     serialised through _sendLock, and no send path may bypass SendJson().
//
//   * The receive loop runs off the main thread, and Unity APIs may only be touched on it. Decoded
//     messages are queued and drained in Update(), so an ICrowdAgent implementation can safely move
//     transforms and start coroutines.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace CrowdDirector
{
    [AddComponentMenu("CrowdDirector/Crowd Director Client")]
    public class CrowdDirectorClient : MonoBehaviour
    {
        [Header("Server")]
        [Tooltip("The sidecar's address. Run server/start.bat or start.sh first.")]
        public string serverUrl = "ws://localhost:8765";

        [Tooltip("Connect automatically on Start.")]
        public bool autoConnect = true;

        [Header("Cadence")]
        [Tooltip("Seconds between decision ticks. The policy costs about 0.7 ms per agent, so this "
               + "is a choice about how often agents may change their minds, not a budget limit.")]
        public float tickInterval = 3.0f;

        [Tooltip("Seconds between state reports to the server.")]
        public float stateInterval = 1.5f;

        [Header("Debug")]
        public bool logActions = false;

        // ── events you can subscribe to ───────────────────────────────────────────────
        public event Action Connected;
        public event Action<string> Disconnected;
        public event Action<JObject> SceneReady;
        public event Action<IReadOnlyList<CrowdDirectorAction>> ActionsReceived;
        public event Action<string> ServerError;

        public bool IsConnected { get { return _socket != null && _socket.State == WebSocketState.Open; } }
        public bool IsSceneReady { get { return _sceneReady; } }
        public int AgentCount { get { return _agents.Count; } }

        readonly Dictionary<string, ICrowdAgent> _agents = new Dictionary<string, ICrowdAgent>();
        readonly ConcurrentQueue<string> _inbound = new ConcurrentQueue<string>();
        readonly SemaphoreSlim _sendLock = new SemaphoreSlim(1, 1);

        const string ClosedSentinel = "__closed";

        ClientWebSocket _socket;
        CancellationTokenSource _cts;
        float _tickTimer, _stateTimer;
        bool _sceneReady;
        string _pendingEventHint = "";

        // ── lifecycle ─────────────────────────────────────────────────────────────────

        void Start()
        {
            if (autoConnect) Connect();
        }

        void Update()
        {
            string json;
            while (_inbound.TryDequeue(out json))
            {
                try { Dispatch(json); }
                catch (Exception e) { Debug.LogError("[CrowdDirector] dispatch failed: " + e.Message); }
            }

            if (!IsConnected || !_sceneReady) return;

            _tickTimer -= Time.deltaTime;
            if (_tickTimer <= 0f)
            {
                _tickTimer = tickInterval;
                SendJson(new { type = "request_tick" });
            }

            _stateTimer -= Time.deltaTime;
            if (_stateTimer <= 0f)
            {
                _stateTimer = stateInterval;
                SendStateUpdate();
            }
        }

        void OnDestroy() { Disconnect(); }

        // ── agent registry ────────────────────────────────────────────────────────────

        /// <summary>Register an agent so it is reported to the server and can be directed.</summary>
        public void RegisterAgent(ICrowdAgent agent)
        {
            if (agent == null) return;
            if (string.IsNullOrEmpty(agent.AgentId))
            {
                // An empty id becomes a null key inside the server's relationship bookkeeping and
                // throws there instead of here, which is far harder to trace back.
                Debug.LogError("[CrowdDirector] refusing to register an agent with no AgentId.");
                return;
            }
            _agents[agent.AgentId] = agent;
        }

        public void UnregisterAgent(ICrowdAgent agent)
        {
            if (agent == null || string.IsNullOrEmpty(agent.AgentId)) return;
            if (_agents.Remove(agent.AgentId) && IsConnected)
                SendJson(new { type = "agent_despawned", agent_id = agent.AgentId });
        }

        // ── connection ────────────────────────────────────────────────────────────────

        public async void Connect()
        {
            if (IsConnected) return;
            Disconnect();

            _socket = new ClientWebSocket();
            _cts = new CancellationTokenSource();
            try
            {
                await _socket.ConnectAsync(new Uri(serverUrl), _cts.Token);
                Debug.Log("[CrowdDirector] connected to " + serverUrl);
                if (Connected != null) Connected();
                ReceiveLoop();
            }
            catch (Exception e)
            {
                Debug.LogError("[CrowdDirector] could not reach " + serverUrl +
                               " - is the sidecar running? (server/start.bat)\n" + e.Message);
                if (Disconnected != null) Disconnected(e.Message);
            }
        }

        public void Disconnect()
        {
            _sceneReady = false;
            if (_cts != null) { _cts.Cancel(); _cts.Dispose(); _cts = null; }
            if (_socket != null) { _socket.Dispose(); _socket = null; }
        }

        async void ReceiveLoop()
        {
            byte[] buffer = new byte[64 * 1024];
            StringBuilder sb = new StringBuilder();
            try
            {
                while (IsConnected && !_cts.IsCancellationRequested)
                {
                    WebSocketReceiveResult r = await _socket.ReceiveAsync(
                        new ArraySegment<byte>(buffer), _cts.Token);

                    if (r.MessageType == WebSocketMessageType.Close)
                    {
                        _inbound.Enqueue(ClosedSentinel);
                        return;
                    }

                    sb.Append(Encoding.UTF8.GetString(buffer, 0, r.Count));
                    if (!r.EndOfMessage) continue;          // a frame, not yet a whole message

                    _inbound.Enqueue(sb.ToString());
                    sb.Length = 0;
                }
            }
            catch (OperationCanceledException) { /* Disconnect() */ }
            catch (Exception e)
            {
                Debug.LogWarning("[CrowdDirector] receive loop ended: " + e.Message);
                _inbound.Enqueue(ClosedSentinel);
            }
        }

        // ── sending ───────────────────────────────────────────────────────────────────

        /// <summary>The ONLY send path. Serialised, because ClientWebSocket allows one in-flight
        /// SendAsync and silently drops a colliding one.</summary>
        async void SendJson(object payload)
        {
            if (!IsConnected) return;
            byte[] bytes = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(payload));

            await _sendLock.WaitAsync();
            try
            {
                if (!IsConnected) return;
                await _socket.SendAsync(new ArraySegment<byte>(bytes),
                                        WebSocketMessageType.Text, true, _cts.Token);
            }
            catch (Exception e) { Debug.LogWarning("[CrowdDirector] send failed: " + e.Message); }
            finally { _sendLock.Release(); }
        }

        void SendStateUpdate()
        {
            List<object> data = new List<object>(_agents.Count);
            foreach (KeyValuePair<string, ICrowdAgent> kv in _agents)
            {
                ICrowdAgent a = kv.Value;
                if (a == null) continue;
                Vector2 p = a.Position;
                data.Add(new
                {
                    id = a.AgentId,
                    name = a.AgentName,
                    agent_type = a.AgentType,
                    personality = a.PersonalityType,
                    x = p.x,
                    y = p.y,
                    needs = a.Needs.ToDictionary(),
                    current_zone = a.CurrentZoneId,
                    encounter_counts = a.EncounterCounts,
                    relationships = a.Relationships,
                    friends = a.Friends,
                });
            }

            if (!string.IsNullOrEmpty(_pendingEventHint))
            {
                SendJson(new { type = "update_state", agents = data, event_hint = _pendingEventHint });
                _pendingEventHint = "";
            }
            else
            {
                SendJson(new { type = "update_state", agents = data });
            }
        }

        // ── the calls you make ────────────────────────────────────────────────────────

        /// <summary>Design a scene from a description. This is the one call that needs
        /// ANTHROPIC_API_KEY set on the server; the per-tick director does not.</summary>
        public void GenerateScene(string description)
        {
            _sceneReady = false;
            SendJson(new { type = "generate_scene", description, render_mode = "library" });
        }

        /// <summary>Free-text instruction - "the coffee machine is broken", "everyone leave".
        /// Compiled against the live scene into a persistent directive.</summary>
        public void DescribeEvent(string description)
        {
            SendJson(new { type = "describe_event", description });
            if (_sceneReady) _tickTimer = 0.5f;            // act on it promptly
        }

        /// <summary>One of: fire_alarm, closing_time, music_starts, new_arrival, all_clear.</summary>
        public void TriggerEvent(string eventType, string message)
        {
            SendJson(new { type = "trigger_event", event_type = eventType, message });
            if (_sceneReady) _tickTimer = 0.5f;
        }

        public void ClearEvent() { SendJson(new { type = "clear_event" }); }

        // ── receiving ─────────────────────────────────────────────────────────────────

        void Dispatch(string json)
        {
            if (json == ClosedSentinel)
            {
                _sceneReady = false;
                Debug.LogWarning("[CrowdDirector] connection closed.");
                if (Disconnected != null) Disconnected("closed");
                return;
            }

            JObject msg = JObject.Parse(json);
            string type = (string)msg["type"];

            switch (type)
            {
                case "scene_ready":
                    _sceneReady = true;
                    _tickTimer = 0.5f;
                    _stateTimer = 0f;
                    if (SceneReady != null) SceneReady(msg);
                    break;

                case "actions":
                    ApplyActions(msg["actions"] as JArray);
                    break;

                case "error":
                    string m = (string)msg["message"];
                    if (string.IsNullOrEmpty(m)) m = "unknown server error";
                    Debug.LogError("[CrowdDirector] server: " + m);
                    if (ServerError != null) ServerError(m);
                    break;

                // generating / event_interpreted / sprite_* / lpc_* are demo-side concerns and are
                // deliberately ignored by a director-only client.
            }
        }

        void ApplyActions(JArray actions)
        {
            if (actions == null) return;
            List<CrowdDirectorAction> applied = new List<CrowdDirectorAction>(actions.Count);

            foreach (JToken t in actions)
            {
                CrowdDirectorAction a = new CrowdDirectorAction();
                a.AgentId = (string)t["agent_id"];
                a.Action = CrowdDirectorAction.Parse((string)t["action"]);
                a.ZoneId = (string)t["zone_id"];
                a.TargetAgentId = (string)t["target_agent_id"];
                a.Reason = (string)t["reason"];

                if (string.IsNullOrEmpty(a.AgentId)) continue;

                ICrowdAgent agent;
                if (!_agents.TryGetValue(a.AgentId, out agent) || agent == null) continue;

                if (logActions) Debug.Log("[CrowdDirector] " + a);

                try
                {
                    switch (a.Action)
                    {
                        case CrowdAction.MoveToZone: agent.DirectorMoveToZone(a.ZoneId, a.Reason); break;
                        case CrowdAction.GroupMove: agent.DirectorGroupMove(a.ZoneId, a.Reason); break;
                        case CrowdAction.Meet:
                        case CrowdAction.StartConversation:
                            agent.DirectorStartConversation(a.TargetAgentId, a.Reason); break;
                        case CrowdAction.Rest: agent.DirectorRest(a.Reason); break;
                        default: agent.DirectorIdle(a.Reason); break;
                    }
                    applied.Add(a);
                }
                catch (Exception e)
                {
                    Debug.LogError("[CrowdDirector] agent '" + a.AgentId + "' threw applying " +
                                   a.Action + ": " + e.Message);
                }
            }

            if (ActionsReceived != null) ActionsReceived(applied);
        }
    }
}
