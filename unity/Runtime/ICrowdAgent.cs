// ICrowdAgent.cs — what you implement to be directed.
//
// The add-on decides WHICH target and WHICH action; you keep navigation, collision, reservations
// and animation. That split is deliberate: the policy was trained against deterministic execution
// systems, so replacing your movement code with its own would change the very thing it learned to
// rely on.
//
// The read-only members are sent to the server every state interval. The Director* methods are
// called on the main thread when a decision arrives.

using System.Collections.Generic;
using UnityEngine;

namespace CrowdDirector
{
    public interface ICrowdAgent
    {
        // ── identity, reported to the server ──────────────────────────────────────────

        /// <summary>Stable, unique, and never null — a null id throws inside the server's
        /// relationship bookkeeping rather than being skipped.</summary>
        string AgentId { get; }

        string AgentName { get; }

        /// <summary>Free-text role for this scene, e.g. "barista". The scene generator invents
        /// these, so never key your own art or logic on it - use PersonalityType.</summary>
        string AgentType { get; }

        /// <summary>From the controlled vocabulary the policy was trained on, e.g. "solo_worker",
        /// "social_group". Stable across generations, unlike AgentType.</summary>
        string PersonalityType { get; }

        // ── live state, reported every state interval ─────────────────────────────────

        Vector2 Position { get; }
        string CurrentZoneId { get; }
        CrowdNeeds Needs { get; }

        /// <summary>How many times this agent has met each other agent. Return null if you are
        /// not modelling encounters; relationships then stay at their initial values.</summary>
        Dictionary<string, int> EncounterCounts { get; }

        /// <summary>Per-pair relationship state. Return null to opt out.</summary>
        Dictionary<string, object> Relationships { get; }

        IEnumerable<string> Friends { get; }

        // ── commands, called on the main thread ───────────────────────────────────────

        void DirectorMoveToZone(string zoneId, string reason);
        void DirectorGroupMove(string zoneId, string reason);
        void DirectorStartConversation(string targetAgentId, string reason);
        void DirectorRest(string reason);
        void DirectorIdle(string reason);
    }
}
