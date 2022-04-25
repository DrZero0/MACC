import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixerHidden(nn.Module):
    def __init__(self, args):
        super(QMixerHidden, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        if args.env == 'sc2':
            self.n_enemies = self.args.state_info_dic[self.args.env_args['map_name']]['n_enemies']
            self.nf_al = self.args.state_info_dic[self.args.env_args['map_name']]['nf_al']
            self.nf_en = self.args.state_info_dic[self.args.env_args['map_name']]['nf_en']
        elif args.env == 'foraging':
            self.n_enemies = args.env_args['max_food']
            self.nf_al = 3
            self.nf_en = 3
        elif args.env == 'stag_hunt':
            self.n_enemies = args.env_args['n_stags'] + args.env_args['n_hare']
            self.nf_al = 5
            self.nf_en = 5
        if self.args.inseparable:
            self.state_dim = int(np.prod(args.state_shape + self.n_agents * (self.args.rnn_hidden_dim_ally - self.nf_al) + self.args.rnn_hidden_state - self.n_enemies * self.nf_en))
        else:
            self.state_dim = int(np.prod(args.state_shape + self.n_agents * (self.args.rnn_hidden_dim_ally - self.nf_al) + self.n_enemies * (self.args.rnn_hidden_dim_enemy - self.nf_en)))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
