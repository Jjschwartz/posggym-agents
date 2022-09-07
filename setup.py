import sys
from setuptools import setup, find_packages

try:
    import torch
except ImportError:
    print(
        "BA-POSGMCP depends on the pytorch library. For installation "
        "instructions visit https://pytorch.org/"
    )
    sys.exit(1)

try:
    import ray.rllib
except ImportError:
    print(
        "BA-POSGMCP depends on the ray RLlib library. For installation "
        "instructions visit https://docs.ray.io/en/latest/rllib/index.html. "
        "Only tested with version 1.12, which is installable via pip with: "
        'pip install "ray[rllib]"==1.12'
    )
    sys.exit(1)

extras = {
    "test": ["pytest>=6.2"],
}

extras['all'] = [item for group in extras.values() for item in group]


setup(
    name='posggym-agents',
    version='0.0.1',
    url="https://github.com/Jjschwartz/posggym-agents/",
    description=(
        "A collection of agents and agent training code for posggym."
    ),
    author="Jonathon Schwartz",
    author_email="Jonathon.Schwartz@anu.edu.au",
    license="MIT",
    packages=[
        package for package in find_packages()
        if package.startswith('posggym_agents')
    ],
    install_requires=[
        'posggym',
    ],
    extras_require=extras,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    zip_safe=False
)
