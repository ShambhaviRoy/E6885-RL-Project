# Gym environment for CityFlow

## Basics

`gym_cityflow` adds a custom environment from CityFlow following this [tutorial](https://github.com/openai/gym/blob/master/docs/creating-environments.md).
 
```
gym_cityflow/
  README.md
  setup.py
  test.py
  gym_cityflow/
    __init__.py
    envs/
      __init__.py
      fcityflow_1x1.py
      1x1_config/
        config.json
        flow.json
        roadnet.json
```

## Installation 
 
 `pip install -e .`
 
```python
import gym
import gym_cityflow

env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
```

CityFlow `config.json`, `flow.json`, and `roadnet.json` are from [CityFlow/examples](https://github.com/cityflow-project/CityFlow/tree/master/examples)

---

## Test Run with DQN
```python
import gym
import gym_cityflow
import numpy as np
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
```

## How to add new environments to `gym`

The explanation below is derived from this [cartpole example](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) which is a part of OpenAI
s gym.

1. Subclass your environment under `gym.Env`

```python
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class MyEnv(gym.Env):
    metadata = {'render.modes':['human']}
...
```

2. override: `__init__(self)`, `step(self, action)`, `reset(self)`, `render(self, mode='human')`, `close(self)`

```python
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class MyEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self):

    def step(self, action):
        """
        Source: https://gym.openai.com/docs/        

        Args:
            action: action for the agent to take

        Returns:
            tuple of 4 valuess: (state, reward, is_done, info).
            
            state (Object): observation from the agent 
            reward (float): reward
            done (boolean): whether the environment has reached a terminal state
            info (dict): additional information for debugging. MAY NOT be used during evaluation, only 
            used during debugging.
        """

    def reset(self):   
        """
        Retruns:
            state (Object): Initial observation
        """

    def render(self, mode='human'):
        print("hello world")

    def close(self):
        """
        stops the environment
        """
```
