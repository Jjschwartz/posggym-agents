# POSGGym Agents

POSGGym-Agents is a collection of agent policies and policy training code for [POSGGym](https://github.com/RDLLab/posggym) environments. The goal of the library is to provide a diverse set of policies and a simple API that makes it easy to import and use policies in your own research.

## Installation

This project depends on the [PyTorch](https://pytorch.org/) and [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) libraries. Specifically pytorch version `>= 1.11` and rllib version `>=2.3`. We recommend install `torch` before installing the POSGGym-Agents package to ensure you get the version of `torch` that works with your CPU/GPU.

You can install the latest version of POSGGym-Agents using `pip install posggym-agents`. Note that this will install the base library without all the agent policy files, which will be downloaded from the posggym-agents github repo upon request (i.e. when `posggym_agents.make` is called), the policy file is then stored locally with the posggym-agents code for reuse.

## API usage

POSGGym-Agents models each agent as a python `policy` class, which at it's simplest accepts an observation and returns the next action. Here's an example using one of the K-Level Reasoning policies in the `PursuitEvasion-v0` environment:


```python
import posggym
import posggym_agents as pga
env = posggym.make("PursuitEvasion-v0", grid="16x16")

policies = {
    '0': pga.make("PursuitEvasion-v0/grid=16x16/klr_k1_seed0_i0-v0", env.model, '0'),
    '1': pga.make("PursuitEvasion-v0/grid=16x16/klr_k1_seed0_i1-v0", env.model, '1')
}

obs, info = env.reset(seed=42)
for i, policy in policies.items():
    policy.reset(seed=7)

for t in range(100):
    actions = {i: policies[i].step(obs[i]) for i in env.agents}
    obs, rewards, termination, truncated, all_done, info = env.step(actions)

    if all_done:
        obs, info = env.reset()
        for i, policy in agents.items():
            policy.reset()

env.close()
for policy in policies.values():
    policy.close()
```

In the above code we initialize two of the implemented policies for the `PursuitEvasion-v0` environment by calling the `posggym_agents.make` function and passing in the full policy ID of each policy, the `posggym.Env` environmnent model and the agent ID of the agent the policy will be used for in the environment (this ensures it uses the correct environment properties such as action and observation space).

The policy ID is made up of four parts:

1. `env_id` - the ID of the environment the policy is for: `PursuitEvasion-v0`
2. `env_args_id` - a string representation of the environment arguments used in the version of the environment the policy is for: `grid=16x16`
3. `policy_name` - the name of the policy: `klr_k1_seed0_i0` and `klr_k1_seed0_i1`
4. `version` - the version of the policy: `v0`

The `env_id` and `env_args_id` may be omitted depending on the policy. If the policy is environment agnostic (e.g. the `Random-v0` policy works for any environment) then both the `env_id` and `env_args_id` can be omitted. While if the policy is environment specific, but works for all variations of the environment or the environment has only a single variation (it doesn't have any parameters) then the `env_args_id` can be omitted (e.g. `PursuitEvasion-v0/shortestpath-v0`).


## List of Policies

The project currently has policies implemented for the following POSGGym environments:

- Driving
- Level Based Foraging
- Predator Prey
- Pursuit Evasion

The full list of policies can be obtained using the following code:

```python
import posggym_agents as pga
pga.pprint_registry()
# will display all policies organized by `env_id/env_args_id/`
```


## Authors

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


## License

`MIT` © 2022, Jonathon Schwartz
