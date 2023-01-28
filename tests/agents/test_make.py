"""Tests that `posggym_agents.make` works as expected.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_make.py

"""
import re
import warnings

import posggym
import pytest

import posggym_agents as pga
from posggym_agents.agents.random_policies import (
    DiscreteFixedDistributionPolicy,
    RandomPolicy,
)


TEST_ENV_ID = "MultiAccessBroadcastChannel-v0"
TEST_ENV_ID_UNV = "MultiAccessBroadcastChannel"


@pytest.fixture(scope="function")
def register_make_testing_policies():
    """Registers testing policies for `posggym_agents.make`"""
    pga.register(
        "GenericTestPolicy-v0",
        entry_point=RandomPolicy,
    )
    pga.register(
        f"{TEST_ENV_ID}/EnvTestPolicy-v1",
        entry_point=RandomPolicy,
    )
    pga.register(
        f"{TEST_ENV_ID}/EnvTestPolicy-v3",
        entry_point=RandomPolicy,
    )
    pga.register(
        f"{TEST_ENV_ID}/EnvTestPolicy-v5",
        entry_point=RandomPolicy,
    )

    pga.register(
        f"{TEST_ENV_ID}/EnvUnversionedTestPolicy",
        entry_point=RandomPolicy,
    )

    pga.register(
        "GenericArgumentTestPolicy-v0",
        entry_point=DiscreteFixedDistributionPolicy,
        kwargs={"dist": None},
    )
    pga.register(
        f"{TEST_ENV_ID}/EnvArgumentTestPolicy-v0",
        entry_point=DiscreteFixedDistributionPolicy,
        kwargs={"dist": None},
    )

    yield

    del pga.registry["GenericTestPolicy-v0"]
    del pga.registry[f"{TEST_ENV_ID}/EnvTestPolicy-v1"]
    del pga.registry[f"{TEST_ENV_ID}/EnvTestPolicy-v3"]
    del pga.registry[f"{TEST_ENV_ID}/EnvTestPolicy-v5"]
    del pga.registry[f"{TEST_ENV_ID}/EnvUnversionedTestPolicy"]
    del pga.registry["GenericArgumentTestPolicy-v0"]
    del pga.registry[f"{TEST_ENV_ID}/EnvArgumentTestPolicy-v0"]


def test_make_generic():
    env_id = "MultiAccessBroadcastChannel-v0"
    env = posggym.make(env_id, disable_env_checker=True)

    policy = pga.make("Random-v0", env.model, env.agents[0])
    assert policy.spec.id == "Random-v0"

    policy = pga.make(f"{env_id}/Random-v0", env.model, env.agents[0])
    assert policy.spec.id == "Random-v0"

    env.close()


def test_make_env():
    env_id = "LevelBasedForaging-5x5-n2-f4-v2"
    env = posggym.make(env_id, disable_env_checker=True)
    policy_id = f"{env_id}/Heuristic1-v0"
    policy = pga.make(policy_id, env.model, env.agents[0])
    assert policy.spec.id == policy_id
    env.close()


def test_make_kwargs(register_make_testing_policies):
    env = posggym.make(TEST_ENV_ID)
    dist = {0: 0.3, 1: 0.7}
    policy = pga.make(
        "GenericArgumentTestPolicy-v0",
        env.model,
        env.agents[0],
        dist=dist,
    )
    assert policy.spec is not None
    assert policy.spec.id == "GenericArgumentTestPolicy-v0"
    assert policy.dist == dist

    policy = pga.make(
        f"{TEST_ENV_ID}/GenericArgumentTestPolicy-v0",
        env.model,
        env.agents[0],
        dist=dist,
    )
    assert policy.spec is not None
    assert policy.spec.id == "GenericArgumentTestPolicy-v0"
    assert policy.dist == dist

    policy = pga.make(
        f"{TEST_ENV_ID}/EnvArgumentTestPolicy-v0",
        env.model,
        env.agents[0],
        dist=dist,
    )
    assert policy.spec is not None
    assert policy.spec.id == f"{TEST_ENV_ID}/EnvArgumentTestPolicy-v0"
    assert policy.dist == dist

    env.close()


@pytest.mark.parametrize(
    "policy_id_input, policy_id_suggested",
    [
        ("random-v0", "Random"),
        ("RAnDom-v0", "Random"),
        ("Discretefixeddistributionpolicy-v10", "DiscreteFixedDistributionPolicy"),
        ("MultiAccessBroadcastChnnel-v0/EnvTestPolicy-v1", TEST_ENV_ID_UNV),
        ("MultiAccessBroadcastChnnel-v0/EnvUnversionedTestPolicy", TEST_ENV_ID_UNV),
        (f"{TEST_ENV_ID}/EnvUnversioneTestPolicy", "EnvUnversionedTestPolicy"),
        (f"{TEST_ENV_ID}/EnvTesPolicy", "EnvTestPolicy"),
    ],
)
def test_policy_suggestions(
    register_make_testing_policies, policy_id_input, policy_id_suggested
):
    env = posggym.make(TEST_ENV_ID)
    with pytest.raises(
        pga.error.UnregisteredPolicy, match=f"Did you mean: `{policy_id_suggested}`?"
    ):
        pga.make(policy_id_input, env.model, env.agents[0])


@pytest.mark.parametrize(
    "policy_id_input, suggested_versions, default_version",
    [
        ("Random-v12", "`v0`", False),
        (f"{TEST_ENV_ID}/EnvTestPolicy-v6", "`v1`, `v3`, `v5`", False),
        (f"{TEST_ENV_ID}/EnvUnversionedTestPolicy-v6", "", True),
    ],
)
def test_env_version_suggestions(
    register_make_testing_policies,
    policy_id_input,
    suggested_versions,
    default_version,
):
    env = posggym.make(TEST_ENV_ID)
    if default_version:
        with pytest.raises(
            pga.error.DeprecatedPolicy,
            match="It provides the default version",
        ):
            pga.make(policy_id_input, env.model, env.agents[0])
    else:
        with pytest.raises(
            pga.error.UnregisteredPolicy,
            match=f"It provides versioned policies: \\[ {suggested_versions} \\]",
        ):
            pga.make(policy_id_input, env.model, env.agents[0])


def test_make_deprecated():
    # Making policy version that is no longer supported will raise an error
    # Note: making an older version (i.e. not the latest version) will only raise a
    #       warning if the older version is still supported (i.e. is in the registry)
    pga.register(
        "DummyPolicy-v1",
        entry_point=RandomPolicy,
    )

    env = posggym.make(TEST_ENV_ID)
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            pga.error.Error,
            match=re.escape(
                "Policy version v0 for `DummyPolicy` is deprecated. Please use "
                "`DummyPolicy-v1` instead."
            ),
        ):
            pga.make("DummyPolicy-v0", env.model, env.agents[0])

    del pga.registry["DummyPolicy-v1"]


def test_make_latest_versioned_env(register_make_testing_policies):
    env = posggym.make(TEST_ENV_ID)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            f"Using the latest versioned policy `{TEST_ENV_ID}/EnvTestPolicy-v5` "
            f"instead of the unversioned policy `{TEST_ENV_ID}/EnvTestPolicy`."
        ),
    ):
        policy = pga.make(f"{TEST_ENV_ID}/EnvTestPolicy", env.model, env.agents[0])
    assert policy.spec is not None
    assert policy.spec.id == f"{TEST_ENV_ID}/EnvTestPolicy-v5"
