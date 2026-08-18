// CrowdDirectorAction.cs — one decision for one agent, as it arrives from the server.

namespace CrowdDirector
{
    public enum CrowdAction
    {
        Idle,
        MoveToZone,
        GroupMove,
        StartConversation,
        Rest,
        Meet,
    }

    public struct CrowdDirectorAction
    {
        public string AgentId;
        public CrowdAction Action;

        /// <summary>Destination zone, for MoveToZone and GroupMove. Null otherwise.</summary>
        public string ZoneId;

        /// <summary>The other party, for StartConversation and Meet. Null otherwise.</summary>
        public string TargetAgentId;

        /// <summary>The policy's stated reason. Useful for debug overlays and logging.</summary>
        public string Reason;

        public static CrowdAction Parse(string wire)
        {
            switch (wire)
            {
                case "move_to_zone": return CrowdAction.MoveToZone;
                case "group_move": return CrowdAction.GroupMove;
                case "start_conversation": return CrowdAction.StartConversation;
                case "meet": return CrowdAction.Meet;
                case "rest": return CrowdAction.Rest;
                default: return CrowdAction.Idle;
            }
        }

        public override string ToString()
        {
            string target = ZoneId ?? TargetAgentId ?? "-";
            return AgentId + ": " + Action + " -> " + target +
                   (string.IsNullOrEmpty(Reason) ? "" : " (" + Reason + ")");
        }
    }
}
