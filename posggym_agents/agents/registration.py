"""Functions and classes for registering and loading implemented agents.

Based on Farama Foundation Gymnasium,
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/envs/registration.py

"""
from __future__ import annotations

import copy
import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple, Optional

import posggym.model as M

from posggym_agents import error, logger
from posggym_agents.policy import Policy


# [env-name/](policy-id)-v(version)
# env-name is group 1, policy_id is group 2, version is group 3
POLICY_ID_RE: re.Pattern = re.compile(
    # r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")
    r"^(?:(?P<env-id>[\w:-]+)\/)?(?:(?P<pi-name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


class PolicyEntryPoint(Protocol):
    """Entry point function for instantiating a new policy instance."""

    def __call__(
        self, model: M.POSGModel, agent_id: M.AgentID, policy_id: str, **kwargs
    ) -> Policy:
        ...


def parse_policy_id(policy_id: str) -> Tuple[str | None, str, int | None]:
    """Parse policy ID string format.

    env-name is group 1, policy-name is group 2, version is group 3

    [env-name/](policy-name)-v(version)

    Arguments
    ---------
    policy_id: The policy id to parse

    Returns
    -------
    env_id: The environment ID
    pi_name: The policy name
    version: The policy version

    Raises
    ------
    Error: If the policy id does not a valid environment regex

    """
    match = POLICY_ID_RE.fullmatch(policy_id)
    if not match:
        raise error.Error(
            f"Malformed environment ID: {policy_id}. (Currently all IDs must be of the "
            "form [env-name/](policy-name)-v(version) (env-name may be optional, "
            "depending on the policy)."
        )
    env_id, pi_name, version = match.group("env-id", "pi-name", "version")
    if version is not None:
        version = int(version)

    return env_id, pi_name, version


def get_policy_id(env_id: str | None, policy_name: str, version: int | None) -> str:
    """Get the full policy ID given a name and (optional) version and env-name.

    Inverse of :meth:`parse_policy_id`.

    Arguments
    ---------
    env_id: The environment ID
    policy_name: The policy name
    version: The policy version

    Returns
    -------
    policy_id: The policy id

    """
    full_name = policy_name
    if version is not None:
        full_name += f"-v{version}"
    if env_id is not None:
        full_name = env_id + "/" + full_name
    return full_name


@dataclass
class PolicySpec:
    """A specification for a particular agent policy.

    Used to register agent policies that can then be dynamically loaded using
    posggym_agents.make.

    Arguments
    ---------
    id: The official policy ID of the agent policy
    entry_point: The Python entrypoint for initializing an instance of the agent policy
    valid_agent_ids: Optional AgentIDs in environment that policy is compatible with. If
        None then assumes policy can be used for any agent in the environment.
    kwargs: Additional kwargs, if any, to pass to the agent initializing

    """

    id: str
    entry_point: PolicyEntryPoint
    valid_agent_ids: List[M.AgentID] | None = field(default=None)

    # Environment Arguments
    kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Extract the namespace, name and version from id.

        Is called after spec is created.
        """
        # Initialize env_id, policy_name, version
        self.env_id, self.policy_name, self.version = parse_policy_id(self.id)


def _check_env_id_exists(env_id: str | None):
    """Check if a env ID exists. If it doesn't, print a helpful error message."""
    if env_id is None:
        return
    env_ids = {spec_.env_id for spec_ in registry.values() if spec_.env_id is not None}
    if env_id in env_ids:
        return

    suggestion = (
        difflib.get_close_matches(env_id, env_ids, n=1) if len(env_ids) > 0 else None
    )
    suggestion_msg = (
        f"Did you mean: `{suggestion[0]}`?"
        if suggestion
        else f"Have you installed the proper package for {env_id}?"
    )

    raise error.EnvIDNotFound(f"Environment ID {env_id} not found. {suggestion_msg}")


def _check_name_exists(env_id: str | None, policy_name: str):
    """Check if policy exists for given env id. If not, print helpful error message."""
    _check_env_id_exists(env_id)
    names = {
        spec_.policy_name.lower(): spec_.policy_name
        for spec_ in registry.values()
        if spec_.env_id == env_id
    }

    if policy_name in names.values():
        return

    suggestion = difflib.get_close_matches(policy_name.lower(), names, n=1)
    namespace_msg = f" for env ID {env_id}" if env_id else ""
    suggestion_msg = f"Did you mean: `{names[suggestion[0]]}`?" if suggestion else ""

    raise error.NameNotFound(
        f"Policy {policy_name} doesn't exist{namespace_msg}. {suggestion_msg}"
    )


def _check_version_exists(env_id: str | None, policy_name: str, version: int | None):
    """Check if policy version exists for env ID. Print helpful error message if not.

    This is a complete test whether an policy ID is valid, and will provide the best
    available hints.

    Arguments
    ---------
    env_id: The environment ID
    policy_name: The policy name
    version: The policy version

    Raises
    ------
    DeprecatedPolicy: The policy doesn't exist but a default version does or the
        policy version is deprecated
    VersionNotFound: The ``version`` used doesn't exist

    """
    if get_policy_id(env_id, policy_name, version) in registry:
        return

    _check_name_exists(env_id, policy_name)
    if version is None:
        return

    message = (
        f"Policy version `v{version}` for policy "
        f"`{get_policy_id(env_id, policy_name, None)}` doesn't exist."
    )

    policy_specs = [
        spec_
        for spec_ in registry.values()
        if spec_.env_id == env_id and spec_.policy_name == policy_name
    ]
    policy_specs = sorted(policy_specs, key=lambda spec_: int(spec_.version or -1))

    default_spec = [spec_ for spec_ in policy_specs if spec_.version is None]

    if default_spec:
        message += f" It provides the default version {default_spec[0].id}`."
        if len(policy_specs) == 1:
            raise error.DeprecatedPolicy(message)

    # Process possible versioned environments
    versioned_specs = [spec_ for spec_ in policy_specs if spec_.version is not None]

    latest_spec = max(
        versioned_specs, key=lambda spec: spec.version, default=None  # type: ignore
    )
    if latest_spec is None or latest_spec.version is None:
        return

    if version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{spec_.version}`" for spec_ in policy_specs)
        message += f" It provides versioned policies: [ {version_list_msg} ]."
        raise error.VersionNotFound(message)

    if version < latest_spec.version:
        raise error.DeprecatedPolicy(
            f"Policy version v{version} for "
            f"`{get_policy_id(env_id, policy_name, None)}` "
            f"is deprecated. Please use `{latest_spec.id}` instead."
        )


def find_highest_version(env_id: str | None, policy_name: str) -> int | None:
    """Finds the highest registered version of the policy in the registry."""
    version: list[int] = [
        spec_.version
        for spec_ in registry.values()
        if spec_.env_id == env_id
        and spec_.policy_name == policy_name
        and spec_.version is not None
    ]
    return max(version, default=None)


# Global registry of policies. Meant to be accessed through `register` and `make`
registry: dict[str, PolicySpec] = {}


def _check_spec_register(spec: PolicySpec):
    """Checks whether spec is valid to be registered.

    Helper function for `register`.
    """
    global registry
    latest_versioned_spec = max(
        (
            spec_
            for spec_ in registry.values()
            if spec_.env_id == spec.env_id
            and spec_.policy_name == spec.policy_name
            and spec_.version is not None
        ),
        key=lambda spec_: int(spec_.version),  # type: ignore
        default=None,
    )

    unversioned_spec = next(
        (
            spec_
            for spec_ in registry.values()
            if spec_.env_id == spec.env_id
            and spec_.policy_name == spec.policy_name
            and spec_.version is None
        ),
        None,
    )

    if unversioned_spec is not None and spec.version is not None:
        raise error.RegistrationError(
            "Can't register the versioned policy "
            f"`{spec.id}` when the unversioned policy "
            f"`{unversioned_spec.id}` of the same name already exists."
        )
    elif latest_versioned_spec is not None and spec.version is None:
        raise error.RegistrationError(
            "Can't register the unversioned policy "
            f"`{spec.id}` when the versioned policy "
            f"`{latest_versioned_spec.id}` of the same name "
            f"already exists. Note: the default behavior is "
            f"that `posggym_agents.make` with the unversioned policy "
            f"will return the latest versioned policy"
        )


def get_all_env_policies(
    env_id: str, _registry: Dict = registry, include_generic_policies: bool = True
) -> List[PolicySpec]:
    """Get all PolicySpecs that are associated with a given environment ID.

    Arguments
    ---------
    env_id: The ID of the environment
    _registry: The policy registry
    include_generic_policies: whether to also return policies that are valid for all
        environments (e.g. the random-v0 policy)

    Returns
    -------
    policy_specs: list of specs for policies associated with given environment.

    """
    return [
        pi_spec
        for pi_spec in _registry.values()
        if (
            pi_spec.env_id == env_id
            or (include_generic_policies and pi_spec.env_id is None)
        )
    ]


def register(
    id: str,
    entry_point: PolicyEntryPoint,
    **kwargs,
):
    """Register a policy with posggym-agents.

    The `id` parameter corresponds to the unique identifier of the policy, with the
    syntax as follows:

        `(env-id)/(policy-name)-v(version)`

    where `env-id` is optional if the policy is valid for all environments
    (e.g. the uniform random policy).

    It takes arbitrary keyword arguments, which are passed to the `PolicySpec`
    constructor.

    Arguments
    ---------
    id: The policy id
    entry_point: The entry point for creating the policy
    **kwargs: arbitrary keyword arguments which are passed to the policy constructor

    """
    global registry
    new_spec = PolicySpec(
        id=id,
        entry_point=entry_point,
        **kwargs,
    )
    register_spec(new_spec, **kwargs)


def register_spec(
    spec: PolicySpec,
    **kwargs,
):
    """Register a policy spec with posggym-agents.

    The `id` parameter of the spec corresponds to the unique identifier of the policy,
    with the syntax as follows:

        `(env-id)/(policy-name)-v(version)`

    where `env-id` is optional if the policy is valid for all environments
    (e.g. the uniform random policy).

    It takes arbitrary keyword arguments, which are passed to the `PolicySpec`
    constructor.

    Arguments
    ---------
    spec: The policy spec
    **kwargs: arbitrary keyword arguments which are passed to the policy constructor

    """
    global registry
    _check_spec_register(spec)
    if spec.id in registry:
        logger.warn(f"Overriding policy {spec.id} already in registry.")
    registry[spec.id] = spec


def make(
    id: str | PolicySpec, model: M.POSGModel, agent_id: M.AgentID, **kwargs
) -> Policy:
    """Create an policy according to the given ID.

    To find all available policies use `posggym_agents.agents.registry.keys()` for
    all valid ids.

    Arguments
    ---------
    id: Name of the policy or a policy spec.
    model: The model for the environment the policy will be interacting with.
    agent_id: The ID of the agent the policy will be used for.
    kwargs: Additional arguments to pass to the policy constructor.

    Returns
    -------
    policy: An instance of the policy.

    Raises
    ------
    Error: If the ``id`` doesn't exist then an error is raised

    """
    if isinstance(id, PolicySpec):
        spec_: Optional[PolicySpec] = id
    else:
        spec_ = registry.get(id)

        env_id, policy_name, version = parse_policy_id(id)
        latest_version = find_highest_version(env_id, policy_name)
        if (
            version is not None
            and latest_version is not None
            and latest_version > version
        ):
            logger.warn(
                f"The policy {id} is out of date. You should consider "
                f"upgrading to version `v{latest_version}`."
            )

        if version is None and latest_version is not None:
            version = latest_version
            new_env_id = get_policy_id(env_id, policy_name, version)
            spec_ = registry.get(new_env_id)  # type: ignore
            logger.warn(
                f"Using the latest versioned environment `{new_env_id}` "
                f"instead of the unversioned environment `{id}`."
            )

    if spec_ is None:
        raise error.Error(f"No registered env with id: {id}")

    if spec_.valid_agent_ids and agent_id not in spec_.valid_agent_ids:
        raise error.Error(
            f"Attempted to initialize policy with ID={spec_.id} with invalid "
            f"agent ID '{agent_id}'. Valid agent IDs for this policy are: "
            f"{spec_.valid_agent_ids}."
        )

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if spec_.entry_point is None:
        raise error.Error(f"{spec_.id} registered but entry_point is not specified")
    else:
        assert callable(spec_.entry_point)
        policy_creator = spec_.entry_point

    try:
        policy = policy_creator(model, agent_id, spec_.id, **kwargs)
    except TypeError as e:
        raise e

    # Copies the policy creation specification and kwargs to add to the
    # policy's specification details
    spec_ = copy.deepcopy(spec_)
    spec_.kwargs = _kwargs
    policy.spec = spec_

    return policy


def spec(id: str) -> PolicySpec:
    """Retrieve the spec for the given policy from the global registry.

    Arguments
    ---------
    id: the policy id.

    Returns
    -------
    spec: the policy spec from the global registry.

    Raises
    ------
    Error: if policy with given ``id`` doesn't exist in global registry.

    """
    spec_ = registry.get(id)
    if spec_ is None:
        env_id, policy_name, version = parse_policy_id(id)
        _check_version_exists(env_id, policy_name, version)
        raise error.Error(f"No registered policy with id: {id}")
    else:
        assert isinstance(spec_, PolicySpec)
        return spec_


def pprint_registry(
    _registry: dict = registry,
    num_cols: int = 3,
    exclude_env_ids: List[str] | None = None,
    disable_print: bool = False,
) -> str | None:
    """Pretty print the policies in the registry.

    Arguments
    ---------
    _registry: Policy registry to be printed.
    num_cols: Number of columns to arrange policies in, for display.
    exclude_env_ids: Exclude any policies for environments with thee IDs from being
        printed.
    disable_print: Whether to return a string of all the policy IDs instead of printing
        it to console.

    Returns
    -------
    return_str: formatted str representation of registry, if ``disable_print=True``,
        otherwise returns ``None``.

    """
    # Defaultdict to store policy names according to env_id.
    env_policies = defaultdict(lambda: [])
    max_justify = float("-inf")
    for spec in _registry.values():
        env_id, _, _ = parse_policy_id(spec.id)
        if env_id is None:
            env_id = "Generic"
        env_policies[env_id].append(spec.id)
        max_justify = max(max_justify, len(spec.id))

    # Iterate through each environment and print policies alphabetically.
    return_str = ""
    for env_id, policies in env_policies.items():
        # Ignore namespaces to exclude.
        if exclude_env_ids is not None and env_id in exclude_env_ids:
            continue
        return_str += f"{'=' * 5} {env_id} {'=' * 5}\n"
        # Reference: https://stackoverflow.com/a/33464001
        for count, item in enumerate(sorted(policies), 1):
            return_str += (
                item.ljust(max_justify) + " "
            )  # Print column with justification.
            # Once all rows printed, switch to new column.
            if count % num_cols == 0 or count == len(policies):
                return_str = return_str.rstrip(" ") + "\n"
        return_str += "\n"

    if disable_print:
        return return_str

    print(return_str, end="")
    return None
