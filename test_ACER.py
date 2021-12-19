import gym
import gym_cityflow
import numpy as np
from stable_baselines.deepq.policies import MlpPolicy
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import  A2C, ACKTR
#from stable_baselines.common import make_vec_env
#tried both ACER and ACKTR, poor performance. DQN was best
if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    steps_per_episode = env.steps_per_episode
    #env = DummyVecEnv([lambda: env])
    #env = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False,clip_obs=10.)
    model = ACKTR("MlpPolicy", VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False,clip_obs=10.), verbose=1)
    log_interval = 10
    total_episodes = 100
    model.learn(total_timesteps=steps_per_episode*total_episodes, log_interval=log_interval)
    model.save("acer_1x1")

    model = ACKTR.load("a2c_1x1", env=VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False,clip_obs=10.))
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