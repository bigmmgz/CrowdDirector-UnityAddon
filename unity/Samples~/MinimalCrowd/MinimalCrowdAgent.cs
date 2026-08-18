// MinimalCrowdAgent.cs — the smallest useful ICrowdAgent.
//
// A capsule that walks to whatever zone the director sends it to. No NavMesh, no animation, no art:
// the point of the sample is to show the contract, not to look good. Swap the movement for your own
// and the rest carries over unchanged.

using System.Collections.Generic;
using UnityEngine;

namespace CrowdDirector.Samples
{
    public class MinimalCrowdAgent : MonoBehaviour, ICrowdAgent
    {
        public float moveSpeed = 1.6f;

        CrowdAgentSpec _spec;
        CrowdScene _scene;
        CrowdNeeds _needs;
        Vector2 _target;
        string _zone;
        bool _resting;

        public void Bind(CrowdAgentSpec spec, CrowdScene scene)
        {
            _spec = spec;
            _scene = scene;
            _needs = spec.InitialNeeds;
            _target = spec.StartPosition;
            _zone = null;

            name = spec.Id + " (" + spec.Name + ")";
            GetComponent<Renderer>().material.color = spec.Color;
        }

        // ── what the server is told each interval ─────────────────────────────────────

        public string AgentId { get { return _spec.Id; } }
        public string AgentName { get { return _spec.Name; } }
        public string AgentType { get { return _spec.AgentType; } }
        public string PersonalityType { get { return _spec.PersonalityType; } }

        public Vector2 Position { get { return transform.position; } }
        public string CurrentZoneId { get { return _zone; } }
        public CrowdNeeds Needs { get { return _needs; } }

        // Social state is optional. Returning null opts out; relationships then stay at their
        // initial values and the policy leans on needs and roles instead.
        public Dictionary<string, int> EncounterCounts { get { return null; } }
        public Dictionary<string, object> Relationships { get { return null; } }
        public IEnumerable<string> Friends { get { return null; } }

        // ── what the director asks of it ──────────────────────────────────────────────

        public void DirectorMoveToZone(string zoneId, string reason)
        {
            _resting = false;
            GoTo(zoneId);
        }

        public void DirectorGroupMove(string zoneId, string reason)
        {
            _resting = false;
            GoTo(zoneId);
        }

        public void DirectorStartConversation(string targetAgentId, string reason)
        {
            // A real implementation would walk to the other agent and play a talk animation.
            _resting = false;
        }

        public void DirectorRest(string reason) { _resting = true; }

        public void DirectorIdle(string reason) { _resting = false; }

        void GoTo(string zoneId)
        {
            if (string.IsNullOrEmpty(zoneId) || _scene == null) return;
            CrowdZone z;
            if (!_scene.TryGetZone(zoneId, out z)) return;

            _zone = zoneId;
            // spread arrivals out so a directed group does not stack on one point
            _target = z.Centre + new Vector2(Random.Range(-z.Bounds.width, z.Bounds.width),
                                             Random.Range(-z.Bounds.height, z.Bounds.height)) * 0.3f;
        }

        void Update()
        {
            if (!_resting)
            {
                Vector2 p = Vector2.MoveTowards(transform.position, _target, moveSpeed * Time.deltaTime);
                transform.position = new Vector3(p.x, p.y, 0f);
            }

            // A crude needs model so the director has something changing to react to. Your own
            // simulation replaces this; the director only reads the values.
            float dt = Time.deltaTime * 0.6f;
            _needs.hunger = Mathf.Clamp(_needs.hunger + dt * 0.8f, 0f, 100f);
            _needs.thirst = Mathf.Clamp(_needs.thirst + dt * 1.0f, 0f, 100f);
            _needs.bladder = Mathf.Clamp(_needs.bladder + dt * 0.5f, 0f, 100f);
            _needs.energy = Mathf.Clamp(_needs.energy + (_resting ? dt * 2f : -dt * 0.4f), 0f, 100f);
        }
    }
}
