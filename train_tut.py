from agent import Agent
import gymnasium as gym
import numpy as np

env = gym.make("CarRacing-v2")

agent = Agent(lr_actor=2.5e-5, lr_critic=2.5e-4, tau=1e-3, env=env)
