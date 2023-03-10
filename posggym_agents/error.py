"""posggym-agent specific errors.

Adapted from Farama Foundation gymnasium, copied here to so that error source path is
reported correctly so as to avoid any confusion.
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/error.py

"""


class Error(Exception):
    """Base posggym-agent error."""


class Unregistered(Error):
    """Raised when user requests item from registry that doesn't exist."""


class UnregisteredPolicy(Unregistered):
    """Raised when user requests policy from registry that doesn't exist."""


class EnvIDNotFound(UnregisteredPolicy):
    """Raised when user requests policy from registry with env-id that doesn't exist."""


class EnvArgsIDNotFound(UnregisteredPolicy):
    """Raised when user requests policy from registry with env-args that don't exist."""


class NameNotFound(UnregisteredPolicy):
    """Raised when user requests policy from registry where name doesn't exist."""


class VersionNotFound(UnregisteredPolicy):
    """Raised when user requests policy from registry where version doesn't exist."""


class DeprecatedPolicy(Error):
    """Raised when user requests policy from registry with old version.

    I.e. if the version number is older than the latest version env with the same
    name.
    """


class RegistrationError(Error):
    """Raised when the user attempts to register an invalid policy.

    For example, an unversioned policy when a versioned env exists.
    """


class UnseedablePolicy(Error):
    """Raised when the user tries to seed an policy that does not support seeding."""


class DependencyNotInstalled(Error):
    """Raised when the user has not installed a dependency."""


class MissingArgument(Error):
    """Raised when a required argument in the initializer is missing."""


class InvalidBound(Error):
    """Raised when the clipping an array with invalid upper and/or lower bound."""


class InvalidFile(Error):
    """Raised when trying to access and invalid posggym-agents file."""


class DownloadError(Error):
    """Raised when error occurred while trying to download posggym-agents file."""
