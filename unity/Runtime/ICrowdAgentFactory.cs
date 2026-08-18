// ICrowdAgentFactory.cs — how your agents get created for a generated scene.
//
// A scene decides its own cast: how many agents, of which types, with which personalities and
// starting needs. The server mints their ids and keys every subsequent message on them, so agents
// must be created FROM the scene rather than announced to it. Implement this, assign it on the
// client, and one Create call arrives per agent whenever a scene becomes ready.

namespace CrowdDirector
{
    public interface ICrowdAgentFactory
    {
        /// <summary>
        /// Instantiate one agent for this spec. The returned object MUST report `spec.Id` as its
        /// AgentId — that is the identity the server directs. Return null to skip this agent.
        /// </summary>
        ICrowdAgent CreateAgent(CrowdAgentSpec spec, CrowdScene scene);

        /// <summary>
        /// Tear down an agent created earlier, called when a new scene replaces the current one.
        /// Destroying the GameObject is the usual implementation.
        /// </summary>
        void DestroyAgent(ICrowdAgent agent);
    }
}
