"""Test that `posggym_agents.register` works as expected.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_register.py

"""
import re
from typing import Optional

import pytest

import posggym_agents as pga


@pytest.mark.parametrize(
    "policy_id, env_id, policy_name, version",
    [
        (
            "MyAwesomeEnv-v0/MyAwesomePolicy-v0",
            "MyAwesomeEnv-v0",
            "MyAwesomePolicy",
            0,
        ),
        ("MyAwesomePolicy-v0", None, "MyAwesomePolicy", 0),
        ("MyAwesomePolicy", None, "MyAwesomePolicy", None),
        ("MyAwesomePolicy-vfinal-v0", None, "MyAwesomePolicy-vfinal", 0),
        ("MyAwesomePolicy-vfinal", None, "MyAwesomePolicy-vfinal", None),
        ("MyAwesomePolicy--", None, "MyAwesomePolicy--", None),
        ("MyAwesomePolicy-v", None, "MyAwesomePolicy-v", None),
    ],
)
def test_register(
    policy_id: str, env_id: Optional[str], policy_name: str, version: Optional[int]
):
    pga.register(policy_id, entry_point="no-entry-point")
    assert pga.spec(policy_id).id == policy_id

    full_name = f"{policy_name}"
    if env_id:
        full_name = f"{env_id}/{full_name}"
    if version is not None:
        full_name = f"{full_name}-v{version}"

    assert full_name in pga.agents.registry

    del pga.agents.registry[policy_id]


@pytest.mark.parametrize(
    "policy_id",
    [
        "“Random-v0”",
        "MyNotSoAwesomePolicy-vNone\n",
        "MyEnvID///MyNotSoAwesomePolicy-vNone",
    ],
)
def test_register_error(policy_id):
    with pytest.raises(
        pga.error.Error, match=f"^Malformed policy ID: {policy_id}"
    ):
        pga.register(policy_id, "no-entry-point")


def test_register_versioned_unversioned():
    # Register versioned then unversioned
    versioned_policy = "MyPolicy-v0"
    pga.register(versioned_policy, "no-entry-point")
    assert pga.agents.spec(versioned_policy).id == versioned_policy

    unversioned_policy = "MyPolicy"
    with pytest.raises(
        pga.error.RegistrationError,
        match=re.escape(
            "Can't register the unversioned policy `MyPolicy` when the versioned"
            " policy `MyPolicy-v0` of the same name already exists."
        ),
    ):
        pga.register(unversioned_policy, "no-entry-point")

    # Clean everything
    del pga.agents.registry[versioned_policy]

    # Register unversioned then versioned
    pga.register(unversioned_policy, "no-entry-point")
    assert pga.agents.spec(unversioned_policy).id == unversioned_policy
    with pytest.raises(
        pga.error.RegistrationError,
        match=re.escape(
            "Can't register the versioned policy `MyPolicy-v0` when the "
            "unversioned policy `MyPolicy` of the same name already exists."
        ),
    ):
        pga.register(versioned_policy, "no-entry-point")

    # Clean everything
    del pga.agents.registry[unversioned_policy]
