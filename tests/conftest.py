def pytest_addoption(parser):  # noqa
    parser.addoption(
        "--env_id_prefix",
        action="store",
        default=None,
        help=(
            "name prefix of environments to test policies for (default is to "
            "test all registered policies in all their environments)."
        ),
    )


def pytest_generate_tests(metafunc):  # noqa
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    if "env_id_prefix" in metafunc.fixturenames:
        metafunc.parametrize(
            "env_id_prefix", [metafunc.config.getoption("env_id_prefix")]
        )


# name prefix of environments to test
# Usage: pytest <test files> --env_name_prefix <name>
#
# This limits the policies to be tested to those associated with environments whose ID
# starts with <name>.
# Will test all registered policies if not specified.
env_name_prefix = None


def pytest_configure(config):
    """Configure pytest."""
    global env_name_prefix
    env_name_prefix = config.getoption("--env_id_prefix")
