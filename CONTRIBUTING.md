# POSGGym-Agents Contribution Guidelines

At this time we are currently accepting the current forms of contributions:

- Bug reports (keep in mind that changing environment behavior should be minimized as that requires releasing a new version of the environment and makes results hard to compare across versions)
- Pull requests for bug fixes
- Documentation improvements
- New agents

## Development

This section contains technical instructions & hints for the contributors.

### Installation

Clone the repo then you can install POSGGym-Agents locally using `pip`  by navigating to the `posggym-agents` root directory (the one containing the `setup.py` file), and running:

```
pip install -e .
```

Or use the following to install `posggym-agents` with all dependencies:

```
pip install -e .[all]
```

And the following to install dependencies for running tests:

```
pip install -e .[testing]
```

### Type checking

This project uses `mypy` for type checking. For instructions on installation see official [instructions](https://mypy.readthedocs.io/en/latest/getting_started.html#installing-and-running-mypy).
Once `mypy` is installed it can be run locally by running ``mypy --package posggym-agents`` from the root project directory.

### Code style

For code style posggym uses `black`. See the [black website](https://black.readthedocs.io/en/stable/) for install instructions.

### Docstrings

For documentation this project uses `pydocstyle`.

### Running tests

The project comes with a number of tests using [pytest](https://docs.pytest.org/en/latest/getting-started.html#install-pytest). These can be run locally with `pytest` from the `posggym-agents/tests` folder.
