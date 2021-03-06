import gym
import gym_cityflow
import numpy as np
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    model = DQN(MlpPolicy, env, verbose=1)
    log_interval = 10
    total_episodes = 100
    model.learn(total_timesteps=env.steps_per_episode*total_episodes, log_interval=log_interval)
    model.save("deepq_1x1")

    model = DQN.load("deepq_1x1", env=env)
    obs = env.reset()
    total_reward = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #print(obs,rewards,dones,info)
        total_reward +=rewards
        if dones:
          time = env.results()
          print('Results: ', time)
          print('reward: ',total_reward)
          break