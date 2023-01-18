# POSGGym Agents

POSGGym-Agents is a collection of agent policies and policy training code for [POSGGym](https://github.com/RDLLab/posggym) environments. It is designed to provide easy-to-use policy implementations that can be used for testing against.

## Installation

This project depends on the [PyTorch](https://pytorch.org/) and [Ray RLlib](https://docs.ray.io/en/releases-1.12.0/rllib/index.html) libraries. Specifically pytorch version `>= 1.11` and rllib version `1.12`. We recommend install `torch` before installing the POSGGym-Agents package to ensure you get the version of `torch` that works with your CPU/GPU.

You can install the latest version of POSGGym-Agents using `pip install posggym-agents` or this version specifically with `pip install posggym-agents==0.1.0`.

## API usage

POSGGym-Agents models each agent as a python `policy` class, which at it's simplest accepts an observation and returns the next action. Here's an example using one of the K-Level Reasoning policies in the `Driving7x7RoundAbout-n2-v0` environment:


```python
import posggym
import posggym_agents as pga
env = posggym.make("Driving-7x7RoundAbout-n2-v0")

agents = [pga.make("Driving-7x7RoundAbout-n2-v0/klr_k1_seed0-v0", env.model, i) for i in env.agents]

obs, info = env.reset(seed=42)
for i, agent in agents.items():
	agent.reset()


for t in range(50):
	actions = {i: agents[i].step(obs[i]) for i in env.agents}
	obs, rewards, termination, truncated, done, info = env.step(actions)

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


## Authors

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


## License

`MIT` © 2022, Jonathon Schwartz


## Versioning

The POSGGym library uses [semantic versioning](https://semver.org/).
