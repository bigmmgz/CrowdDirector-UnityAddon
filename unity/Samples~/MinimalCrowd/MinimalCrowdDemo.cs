// MinimalCrowdDemo.cs — a complete runnable demo in one component.
//
// Drop this on an empty GameObject in an empty scene and press Play. It builds the camera, the
// client, the zone floors and the crowd at runtime, so there is no scene file to author and nothing
// to wire in the inspector.
//
// It loads the prebuilt "demo" scene, which needs NO API key: the server reads it from
// prebuilt_scenes/demo.json rather than designing one. That also makes it repeatable, which is what
// you want when recording several takes of the same thing.

using System.Collections.Generic;
using UnityEngine;

namespace CrowdDirector.Samples
{
    [AddComponentMenu("CrowdDirector/Samples/Minimal Crowd Demo")]
    public class MinimalCrowdDemo : MonoBehaviour, ICrowdAgentFactory
    {
        [Header("Server")]
        public string serverUrl = "ws://localhost:8765";

        [Tooltip("A prebuilt scene key. 'demo' ships with the server and needs no API key. Leave "
               + "blank to generate one from `description` instead, which does need a key.")]
        public string prebuiltScene = "demo";

        [Tooltip("Used only when `prebuiltScene` is blank.")]
        public string description = "a busy hospital waiting room at visiting hour";

        CrowdDirectorClient _client;
        readonly List<GameObject> _zoneFloors = new List<GameObject>();

        void Start()
        {
            SetUpCamera();

            _client = gameObject.AddComponent<CrowdDirectorClient>();
            _client.serverUrl = serverUrl;
            _client.autoConnect = false;          // connect after the factory is assigned
            _client.AgentFactory = this;
            _client.logActions = true;

            _client.Connected += OnConnected;
            _client.SceneReady += OnSceneReady;
            _client.ServerError += m => Debug.LogError("[demo] server: " + m);

            _client.Connect();
        }

        static void SetUpCamera()
        {
            Camera cam = Camera.main;
            if (cam == null)
            {
                GameObject go = new GameObject("Main Camera");
                go.tag = "MainCamera";
                cam = go.AddComponent<Camera>();
            }
            cam.orthographic = true;
            cam.orthographicSize = 7f;                     // the world rect is X [-8,8], Y [-5,5]
            cam.transform.position = new Vector3(0f, 0f, -10f);
            cam.backgroundColor = new Color(0.13f, 0.14f, 0.17f);
            cam.clearFlags = CameraClearFlags.SolidColor;
        }

        void OnConnected()
        {
            if (!string.IsNullOrEmpty(prebuiltScene)) _client.LoadScene(prebuiltScene);
            else _client.GenerateScene(description);
        }

        void OnSceneReady(CrowdScene scene)
        {
            foreach (GameObject g in _zoneFloors) Destroy(g);
            _zoneFloors.Clear();

            foreach (CrowdZone z in scene.Zones)
            {
                GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Quad);
                Destroy(floor.GetComponent<Collider>());
                floor.name = "zone:" + z.Id;
                floor.transform.position = new Vector3(z.Centre.x, z.Centre.y, 1f);
                floor.transform.localScale = new Vector3(z.Bounds.width, z.Bounds.height, 1f);

                Color c = z.Color;
                c.a = 1f;
                floor.GetComponent<Renderer>().material.color = c * 0.55f;
                _zoneFloors.Add(floor);
            }

            Debug.Log("[demo] " + scene.Name + ": " + scene.Zones.Count + " zones, " +
                      _client.AgentCount + " agents");
        }

        // ── ICrowdAgentFactory ────────────────────────────────────────────────────────

        public ICrowdAgent CreateAgent(CrowdAgentSpec spec, CrowdScene scene)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            Destroy(go.GetComponent<Collider>());
            go.transform.localScale = new Vector3(0.34f, 0.34f, 0.34f);
            go.transform.position = new Vector3(spec.StartPosition.x, spec.StartPosition.y, 0f);

            MinimalCrowdAgent a = go.AddComponent<MinimalCrowdAgent>();
            a.Bind(spec, scene);                  // reports spec.Id as its AgentId - required
            return a;
        }

        public void DestroyAgent(ICrowdAgent agent)
        {
            MonoBehaviour mb = agent as MonoBehaviour;
            if (mb != null) Destroy(mb.gameObject);
        }

        // ── a small control panel, so a recording has something to show ───────────────

        string _instruction = "the coffee machine is broken";

        void OnGUI()
        {
            const int w = 340;
            GUILayout.BeginArea(new Rect(12, 12, w, 260), GUI.skin.box);

            GUILayout.Label(_client != null && _client.IsConnected
                ? "Connected  -  " + _client.AgentCount + " agents"
                : "Not connected. Start the server: server/start.bat");

            GUILayout.Space(6);
            GUILayout.Label("Instruction");
            _instruction = GUILayout.TextField(_instruction);

            if (GUILayout.Button("Send instruction") && _client != null)
                _client.DescribeEvent(_instruction);

            GUILayout.Space(6);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Fire alarm") && _client != null)
                _client.TriggerEvent("fire_alarm", "Smoke in the east wing");
            if (GUILayout.Button("All clear") && _client != null)
                _client.TriggerEvent("all_clear", "The alarm was a false one");
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Closing time") && _client != null)
                _client.TriggerEvent("closing_time", "The venue is closing");
            if (GUILayout.Button("Reload scene") && _client != null)
                _client.LoadScene(prebuiltScene);
            GUILayout.EndHorizontal();

            GUILayout.EndArea();
        }
    }
}
