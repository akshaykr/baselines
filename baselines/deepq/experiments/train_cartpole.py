import sys
sys.path.append('/Users/akshay/code/baselines/')
sys.path.append('/Users/akshay/projects/state_decoding/exp_contextual/python/')
import gym
from gym.envs.registration import register

from baselines import deepq

register(
    id = 'Grid-v0',
    entry_point='Environments:GridWorld',
)

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("Grid-v0")
    env.init(env_config={'horizon':2, 'dimension':2, 'swap':0.1})

    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=10000*env.horizon,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        prioritized_replay=True,
        print_freq=100,
        callback=None
    )


if __name__ == '__main__':
    main()
