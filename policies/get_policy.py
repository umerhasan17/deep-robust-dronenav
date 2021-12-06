from habitat_baselines.rl.ppo import PointNavBaselinePolicy
from policies.baseline_midlevel_policy import PointNavBaselineMidLevelPolicy
from policies.midlevel_map_policy import PointNavDRRNPolicy


def get_current_policy_object(policy_name):
    policy_objects = dict(
        Baseline=PointNavBaselinePolicy,
        BaselineMidlevel=PointNavBaselineMidLevelPolicy,
        DRDN=PointNavDRRNPolicy,
    )

    return policy_objects[policy_name]
