from smac.env.multiagentenv import MultiAgentEnv
import numpy as np
from typing import Optional
import gym
from gym.envs.registration import register
import lbforaging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ForagingEnv(MultiAgentEnv):

    def __init__(self,
                 field_size: int,
                 players: int,
                 max_food: int,
                 force_coop: bool,
                 partially_observe: bool,
                 sight: int,
                 is_print: bool,
                 seed: int, 
                 need_render: bool):
        self.n_agents = players
        self.max_food = max_food
        self.n_actions = 6
        self._total_steps = 0
        self._episode_steps = 0
        self.NN = 0
        self.is_print = is_print
        self.need_render = need_render
        np.random.seed(seed)

        self.episode_limit = 50

        self.agent_score = np.zeros(players)

        register(
        id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v0".format(field_size, players, max_food,
                                                                "-coop" if force_coop else "",
                                                                "-{}s".format(sight) if partially_observe else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": players,
            "max_player_level": 3,
            "field_size": (field_size, field_size),
            "max_food": max_food,
            "sight": sight if partially_observe else field_size,
            "max_episode_steps": 50,
            "force_coop": force_coop,
        },
        )
        env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v0".format(field_size, players, max_food,
                                                              "-coop" if force_coop else "",
                                                              "-{}s".format(sight) if partially_observe else "")
        print(env_id)
        # env = gym.make("Foraging-8x8-2p-1f-coop-v0")
        self.env = gym.make(env_id)
        self.env.seed(seed)

    def step(self, actions):
        # print('actions', actions.shape)
        """ Returns reward, terminated, info """
        self._total_steps += 1
        self._episode_steps += 1

        # For visualization, not complete in this version
        # if self.is_print:
        #     actionLog = open('./pics/actions.log', mode = 'a+', encoding='utf-8')
        #     actionLog.write('t_steps: %d\n' % self._episode_steps)
        #     actionLog.write('actions: %s\n' % str(actions.cpu().numpy()))

        # if self.need_render:
        #     fig = plt.figure()
        #     data = self.env.render(mode='rgb_array')
        #     plt.imshow(data)
        #     plt.axis('off')
        #     if not os.path.exists("./pics"):
        #         os.makedirs("./pics")
        #     fig.savefig("pics/game-{}.png".format(self.NN), bbox_inches='tight')
        #     self.NN += 1
        self.obs, rewards, dones, info, self.food_state, self.player_state = self.env.step(actions.cpu().numpy())

        # print('actions', actions.shape)
        # assert actions.shape[0] == 2
        # self.obs, rewards, dones, info = self.env.step(actions)
        self.agent_score += rewards
        # self.agent_score -= 0.002 / self.n_agents

        # reward = np.sum(rewards, axis=1)
        reward = np.sum(rewards)
        # step penalty
        reward -= 0.002
        terminated = np.all(dones)
        # TODO:
        # if reward > 0:
        #     terminated = True

        return reward, terminated, info

    def get_obs(self):
        # print('Im in get_obs')
        """ Returns all agent observations in a list """
        # print('obs', self.obs)
        return self.obs

    def get_obs_agent(self, agent_id):
        # print('Im in get_obs_agent')
        """ Returns observation for agent_id """
        return np.array(self.obs[agent_id])

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env._get_observation_space().shape[0]

    def get_state(self):
        state = self.player_state
        state = np.concatenate([state, self.food_state])
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        # print('self.env._obs_length', self.env._obs_length)
        return 3 * self.n_agents + 3 * self.max_food

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        res = [0] * self.n_actions
        t = self.env._valid_actions[self.env.players[agent_id]]
        for i in range(len(t)):
            res[t[i].value] = 1
        return res

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self._episode_steps = 0
        self.agent_score = np.zeros(self.n_agents)
        # self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.obs, self.food_state, self.player_state = self.env.reset()
        return self.get_obs(), self.get_state()

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "agent_score": self.agent_score,
        }
        return stats
