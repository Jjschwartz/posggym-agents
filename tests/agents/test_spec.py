"""Tests that posggym_agents.spec works as expected.

Reference:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_spec.py
"""
import re

import posggym
import pytest

import posggym_agents as pga


def test_generic_spec():
    spec = pga.spec("Random-v0")
    assert spec.id == "Random-v0"
    assert spec is pga.registry["Random-v0"]


def test_generic_env_spec():
    spec = pga.spec("MultiAccessBroadcastChannel-v0/Random-v0")
    assert spec.id == "Random-v0"
    assert spec is pga.registry["Random-v0"]


def test_env_spec():
    spec = pga.spec("LevelBasedForaging-5x5-n2-f4-v2/Heuristic1-v0")
    assert spec.id == "LevelBasedForaging-5x5-n2-f4-v2/Heuristic1-v0"
    assert spec is pga.registry["LevelBasedForaging-5x5-n2-f4-v2/Heuristic1-v0"]


def test_generic_spec_kwargs():
    env = posggym.make("MultiAccessBroadcastChannel-v0")
    action_dist = {0: 0.3, 1: 0.7}
    policy = pga.make(
        "DiscreteFixedDistributionPolicy-v0", env.model, env.agents[0], dist=action_dist
    )
    assert policy.spec is not None
    assert policy.spec.kwargs["dist"] == action_dist


def test_generic_spec_missing_lookup():
    pga.register(id="Test1-v0", entry_point="no-entry-point")
    pga.register(id="Test1-v15", entry_point="no-entry-point")
    pga.register(id="Test1-v9", entry_point="no-entry-point")
    pga.register(id="Other1-v100", entry_point="no-entry-point")

    with pytest.raises(
        pga.error.DeprecatedPolicy,
        match=re.escape(
            "Policy version v1 for `Test1` is deprecated. Please use `Test1-v15` "
            "instead."
        ),
    ):
        pga.spec("Test1-v1")

    with pytest.raises(
        pga.error.UnregisteredPolicy,
        match=re.escape(
            "Policy version `v1000` for policy `Test1` doesn't exist. "
            "It provides versioned policies: [ `v0`, `v9`, `v15` ]."
        ),
    ):
        pga.spec("Test1-v1000")

    with pytest.raises(
        pga.error.UnregisteredPolicy,
        match=re.escape("Policy Unknown1 doesn't exist. "),
    ):
        pga.spec("Unknown1-v1")


def test_env_spec_missing_lookup():
    env_id = "MultiAccessBroadcastChannel-v0"
    pga.register(id=f"{env_id}/Test1-v0", entry_point="no-entry-point")
    pga.register(id=f"{env_id}/Test1-v15", entry_point="no-entry-point")
    pga.register(id=f"{env_id}/Test1-v9", entry_point="no-entry-point")
    pga.register(id=f"{env_id}/Other1-v100", entry_point="no-entry-point")

    with pytest.raises(
        pga.error.DeprecatedPolicy,
        match=re.escape(
            f"Policy version v1 for `{env_id}/Test1` is deprecated. Please use "
            f"`{env_id}/Test1-v15` instead."
        ),
    ):
        pga.spec(f"{env_id}/Test1-v1")

    with pytest.raises(
        pga.error.UnregisteredPolicy,
        match=re.escape(
            f"Policy version `v1000` for policy `{env_id}/Test1` doesn't exist. "
            "It provides versioned policies: [ `v0`, `v9`, `v15` ]."
        ),
    ):
        pga.spec(f"{env_id}/Test1-v1000")

    with pytest.raises(
        pga.error.UnregisteredPolicy,
        match=re.escape(f"Policy Unknown1 doesn't exist for env ID {env_id}. "),
    ):
        pga.spec(f"{env_id}/Unknown1-v1")


def test_spec_malformed_lookup():
    expected_error_msg = (
        "Malformed policy ID: “Random-v0”. "
        "(Currently all IDs must be of the form [env-id/](policy-name)-v(version) "
        "(env-id may be optional, depending on the policy))."
    )
    with pytest.raises(
        pga.error.Error,
        match=f"^{re.escape(expected_error_msg)}$",
    ):
        pga.spec("“Random-v0”")


def test_spec_default_lookups():
    env_id = "MultiAccessBroadcastChannel-v0"
    pga.register(id=f"{env_id}/Test3", entry_point="no-entry-point")
    pga.register(id="Test4", entry_point="no-entry-point")

    with pytest.raises(
        pga.error.DeprecatedPolicy,
        match=re.escape(
            f"Policy version `v0` for policy `{env_id}/Test3` doesn't exist. "
            f"It provides the default version {env_id}/Test3`."
        ),
    ):
        pga.spec(f"{env_id}/Test3-v0")

    assert pga.spec(f"{env_id}/Test3") is not None

    with pytest.raises(
        pga.error.DeprecatedPolicy,
        match=re.escape(
            "Policy version `v0` for policy `Test4` doesn't exist. "
            "It provides the default version Test4`."
        ),
    ):
        pga.spec("Test4-v0")

    assert pga.spec("Test4") is not None

    with pytest.raises(
        pga.error.DeprecatedPolicy,
        match=re.escape(
            "Policy version `v0` for policy `Test4` doesn't exist. "
            "It provides the default version Test4`."
        ),
    ):
        pga.spec(f"{env_id}/Test4-v0")

    assert pga.spec(f"{env_id}/Test4") is not None
