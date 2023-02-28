# POSGGym Agents

POSGGym-Agents is a collection of agent policies and policy training code for [POSGGym](https://github.com/RDLLab/posggym) environments. It is designed to provide easy-to-use policy implementations that can be used for testing against.

> **_NOTE:_** This project is under active development and you are currently looking at version 0.1.2 of this project which works with the old posggym API (version 0.1.0). Stay tuned for a newer version that is compatible with the newer posggym API.


## Installation

This project depends on the [PyTorch](https://pytorch.org/) and [Ray RLlib](https://docs.ray.io/en/releases-1.12.0/rllib/index.html) libraries. Specifically pytorch version `>= 1.11` and rllib version `1.12`. We recommend install `torch` before installing the POSGGym-Agents package to ensure you get the version of `torch` that works with your CPU/GPU setup.

This version of POSGGym-Agents (version=`0.1.0`) can be installed by cloning this repo and and installing locally using `pip`. This can be done in two ways:

1. Using pip to install from repo directly

```
# via http
pip install git+https://github.com/Jjschwartz/posggym-agents.git@v0.1.2

```

2. Cloning repo and installing from local repo (do this if you plan to edit the code)

```
git clone --depth 1 --branch v0.1.1 git@github.com:Jjschwartz/posggym-agents.git
cd posggym-agents
pip install -e .
```


## API usage

POSGGym-Agents models each agent as a python `policy` class, which at it's simplest accepts an observation and returns the next action. Here's an example using one of the K-Level Reasoning policies in the `Driving7x7RoundAbout-n2-v0` environment:


```python
import posggym
import posggym_agents as pga
env = posggym.make("Driving7x7RoundAbout-n2-v0")

agents = [pga.make("Driving7x7RoundAbout-n2-v0/klr_k1_seed0-v0", env.model, i) for i in env.agents]

obs = env.reset(seed=42)
for i, agent in agents.items():
	agent.reset()


for t in range(50):
	actions = tuple(agents[i].step(obs[i]) for i in env.agents)
	obs, rewards, done, info = env.step(actions)

	if done:
		obs = env.reset()
		for i, agent in agents.items():
		    agent.reset()

env.close()
```

## Agents

The project currently has agents implemented for the following POSGGym environments:

- Driving
- Level Based Foraging
- Predator Prey
- Pursuit Evasion


Note, agents may only be implemented for specific version of each problem.


## Authors

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


## License

`MIT` © 2022, Jonathon Schwartz


## Versioning

The POSGGym library uses [semantic versioning](https://semver.org/).
