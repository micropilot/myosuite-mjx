import time 

import torch 
from brax.io import model

from myosuite.utils import gym
from myosuite.agents.brax.flax_to_torch import (
    TorchModel
)

params = model.load_params('policies/brax')
model = TorchModel(params)
model.eval()

env = gym.make('myoFingerReachRandom-v0').unwrapped
print (env.action_space)

state, _ = env.reset()
print("State:", state)

for _ in range(32):
    state = torch.tensor(state, dtype=torch.float32)
    action = model(state)
    action = action.detach().numpy()
    obs, rew, done, _, info = env.step(action)
    env.mj_render()
    time.sleep(0.01)
    print (rew, done)
    if done:
        break

    


# if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_file', type=str, default='./data/reach_one_hand/torch_model.pt')
    # parser.add_argument('--task', type=str, default='reach')
    # parser.add_argument('--render', action='store_true', default=False)
    # parser.add_argument('--with_full_model', action='store_true', default=False)
    # args = parser.parse_args()

    # main(args)