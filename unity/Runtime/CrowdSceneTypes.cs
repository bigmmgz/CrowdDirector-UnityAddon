// CrowdSceneTypes.cs — what the server hands back when a scene is ready.
//
// The SERVER owns the roster. It mints agent ids during scene generation (`agent_0`, `agent_1`, …)
// from the cast the scene designer chose, and every later message — state in, actions out — is keyed
// on those ids. So the client's job is to instantiate one of your agents per spec it is given, not to
// announce agents it already has: an id the server did not mint is silently dropped at both ends.

using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace CrowdDirector
{
    /// <summary>One agent the server expects to exist, delivered with the scene.</summary>
    public struct CrowdAgentSpec
    {
        /// <summary>Server-assigned. Your agent MUST report this back as its AgentId.</summary>
        public string Id;

        public string Name;
        public string AgentType;
        public string PersonalityType;
        public Vector2 StartPosition;
        public Color Color;
        public CrowdNeeds InitialNeeds;

        public static CrowdAgentSpec FromJson(JToken t)
        {
            CrowdAgentSpec s = new CrowdAgentSpec();
            s.Id = (string)t["id"];
            s.Name = (string)t["name"];
            s.AgentType = (string)t["agent_type"];
            s.PersonalityType = (string)t["personality_type"] ?? (string)t["personality"];
            s.StartPosition = new Vector2((float?)t["x"] ?? 0f, (float?)t["y"] ?? 0f);

            Color c;
            s.Color = ColorUtility.TryParseHtmlString((string)t["color"] ?? "", out c) ? c : Color.white;

            CrowdNeeds n = CrowdNeeds.Default;
            JToken needs = t["needs"];
            if (needs != null)
            {
                n.hunger = (float?)needs["hunger"] ?? n.hunger;
                n.thirst = (float?)needs["thirst"] ?? n.thirst;
                n.bladder = (float?)needs["bladder"] ?? n.bladder;
                n.energy = (float?)needs["energy"] ?? n.energy;
                n.stress = (float?)needs["stress"] ?? n.stress;
                n.loneliness = (float?)needs["loneliness"] ?? n.loneliness;
                n.groupAffinity = (float?)needs["groupAffinity"] ?? n.groupAffinity;
                n.status = (float?)needs["status"] ?? n.status;
                n.curiosity = (float?)needs["curiosity"] ?? n.curiosity;
            }
            s.InitialNeeds = n;
            return s;
        }
    }

    /// <summary>A room in the generated floor plan. Zones arrive gap-free and tile the world rect.</summary>
    public struct CrowdZone
    {
        public string Id;
        public string Label;
        public string ZoneType;

        /// <summary>World-space rectangle. X is in [-8, 8] and Y in [-5, 5] by default.</summary>
        public Rect Bounds;

        public Color Color;

        public Vector2 Centre { get { return Bounds.center; } }

        public static CrowdZone FromJson(JToken t)
        {
            CrowdZone z = new CrowdZone();
            z.Id = (string)t["id"];
            z.Label = (string)t["label"];
            z.ZoneType = (string)t["zone_type"];
            z.Bounds = new Rect((float?)t["x"] ?? 0f, (float?)t["y"] ?? 0f,
                                (float?)t["w"] ?? 1f, (float?)t["h"] ?? 1f);

            Color c;
            z.Color = ColorUtility.TryParseHtmlString((string)t["color"] ?? "", out c) ? c : Color.gray;
            return z;
        }
    }

    /// <summary>The scene as generated: its zones and the cast to instantiate.</summary>
    public sealed class CrowdScene
    {
        public string Name;
        public string Theme;
        public string Description;
        public readonly List<CrowdZone> Zones = new List<CrowdZone>();
        public readonly List<CrowdAgentSpec> Agents = new List<CrowdAgentSpec>();

        readonly Dictionary<string, CrowdZone> _byId =
            new Dictionary<string, CrowdZone>(System.StringComparer.Ordinal);

        public bool TryGetZone(string zoneId, out CrowdZone zone)
        {
            return _byId.TryGetValue(zoneId ?? "", out zone);
        }

        /// <summary>Zone centre, or Vector2.zero when the id is unknown.</summary>
        public Vector2 ZoneCentre(string zoneId)
        {
            CrowdZone z;
            return TryGetZone(zoneId, out z) ? z.Centre : Vector2.zero;
        }

        public static CrowdScene FromJson(JObject msg)
        {
            CrowdScene s = new CrowdScene();
            s.Name = (string)msg["scene_name"];
            s.Theme = (string)msg["theme"];
            s.Description = (string)msg["description"];

            JArray zones = msg["zones"] as JArray;
            if (zones != null)
                foreach (JToken t in zones)
                {
                    CrowdZone z = CrowdZone.FromJson(t);
                    s.Zones.Add(z);
                    if (!string.IsNullOrEmpty(z.Id)) s._byId[z.Id] = z;
                }

            JArray agents = msg["agents"] as JArray;
            if (agents != null)
                foreach (JToken t in agents)
                    s.Agents.Add(CrowdAgentSpec.FromJson(t));

            return s;
        }
    }
}
