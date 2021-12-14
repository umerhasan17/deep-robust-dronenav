from habitat_baselines.rl.ppo import PointNavBaselinePolicy
from policies.baseline_midlevel_policy import PointNavBaselineMidLevelPolicy
from policies.DRRN_policy import PointNavDRRNPolicy
from policies.actual_map_policy import PointNavDRRNActualMapPolicy


def get_current_policy_object(policy_name):
    policy_objects = dict(
        Baseline=PointNavBaselinePolicy,
        BaselineMidLevel=PointNavBaselineMidLevelPolicy,
        DRRN=PointNavDRRNPolicy,
        DRRNActualMap=PointNavDRRNActualMapPolicy,
    )

    return policy_objects[policy_name]
