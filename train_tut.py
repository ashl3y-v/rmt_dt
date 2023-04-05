from agent import Agent
import gymnasium as gym
import numpy as np

env = gym.make("CarRacing-v2")

agent = Agent(lr_actor=2.5E-5, lr_critic=2.5E-4, tau=1E-3, env=env)
