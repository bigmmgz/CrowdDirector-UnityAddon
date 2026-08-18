// CrowdNeeds.cs — the nine-value need vector the policy was trained on.
//
// The names and the order are FROZEN: they match ecgp.vocab.NEEDS_V1 exactly. The server keys on
// these strings, so renaming a field here silently drops that need from every decision rather than
// raising anything. Values run 0-100, where high means the need is pressing (except energy, where
// low means tired).

using System.Collections.Generic;

namespace CrowdDirector
{
    [System.Serializable]
    public struct CrowdNeeds
    {
        public float hunger;
        public float thirst;
        public float bladder;
        public float energy;
        public float stress;
        public float loneliness;
        public float groupAffinity;
        public float status;
        public float curiosity;

        /// <summary>A sensible mid-range starting point for a freshly spawned agent.</summary>
        public static CrowdNeeds Default
        {
            get
            {
                CrowdNeeds n = new CrowdNeeds();
                n.hunger = 30f; n.thirst = 30f; n.bladder = 20f;
                n.energy = 80f; n.stress = 20f; n.loneliness = 30f;
                n.groupAffinity = 50f; n.status = 50f; n.curiosity = 50f;
                return n;
            }
        }

        public Dictionary<string, float> ToDictionary()
        {
            Dictionary<string, float> d = new Dictionary<string, float>(9);
            d["hunger"] = hunger;
            d["thirst"] = thirst;
            d["bladder"] = bladder;
            d["energy"] = energy;
            d["stress"] = stress;
            d["loneliness"] = loneliness;
            d["groupAffinity"] = groupAffinity;
            d["status"] = status;
            d["curiosity"] = curiosity;
            return d;
        }
    }
}
