"""Finds all the specs of policies that we can test with.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/utils.py

"""
from typing import List, Optional

import numpy as np
import posggym

import posggym_agents as pga
from posggym_agents.agents.registration import PolicySpec
from tests.conftest import env_name_prefix


def try_make_policy(policy_spec: PolicySpec) -> Optional[pga.Policy]:
    """Tries to make the policy showing if it is possible."""
    try:
        if policy_spec.env_id is None:
            env = posggym.make("MultiAccessBroadcastChannel-v0")
        else:
            env = posggym.make(policy_spec.env_id)

        if policy_spec.valid_agent_ids:
            agent_id = policy_spec.valid_agent_ids[0]
        else:
            agent_id = env.possible_agents[0]

        return pga.make(policy_spec, env.model, agent_id)
    except (
        ImportError,
        posggym.error.DependencyNotInstalled,
        posggym.error.MissingArgument,
    ) as e:
        pga.logger.warn(f"Not testing {policy_spec.id} due to error: {e}")
    return None


# Tries to make all policies to test with
_all_testing_initialised_policies: List[Optional[pga.Policy]] = [
    try_make_policy(policy_spec)
    for policy_spec in pga.registry.values()
    if env_name_prefix is None or policy_spec.id.startswith(env_name_prefix)
]
all_testing_initialised_policies: List[pga.Policy] = [
    policy for policy in _all_testing_initialised_policies if policy is not None
]

# All testing posggym-agents policy specs
all_testing_policy_specs: List[PolicySpec] = [
    policy.spec
    for policy in all_testing_initialised_policies
    if policy.spec is not None
]


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.

    Arguments
    ---------
    a: first data structure
    b: second data structure
    prefix: prefix for failed assertion message for types and dicts

    """
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"
        for k in a:
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b, prefix)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b, prefix)
    else:
        assert a == b
