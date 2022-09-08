
def pytest_addoption(parser):   # noqa
    parser.addoption(
        "--env_name_prefix",
        action="store",
        default=None,
        help=(
            "name prefix of environments to test policies for (default is to "
            "test all registered policies in all their environments)."
        )
    )


def pytest_generate_tests(metafunc):   # noqa
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    if "env_name_prefix" in metafunc.fixturenames:
        metafunc.parametrize(
            "env_name_prefix", [metafunc.config.getoption("env_name_prefix")]
        )
