import torch as th
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.distributions import kl_divergence
from tensorboardX import SummaryWriter

from utils.sparsemax import Sparsemax

writer = None
def init_writer(log_name):
    global writer
    writer = SummaryWriter(log_dir=log_name)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_dim, softTemperature, dim_q=None, dim_k=None, dim_v=None, verbose=False, isSoftmax=False):
        super(MultiHeadAttention, self).__init__()
        assert (n_dim % n_heads) == 0, "n_heads must divide n_dim"
        attn_dim = n_dim // n_heads
        self.attn_dim = attn_dim
        self.n_heads = n_heads
        self.verbose = verbose
        self.temperature=attn_dim ** 0.5 / softTemperature
        self.isSoftmax = isSoftmax
        if dim_q is None:
            dim_q = n_dim
        if dim_k is None:
            dim_k = dim_q
        if dim_v is None:
            dim_v = dim_k

        self.fc_q = nn.Linear(dim_q, n_dim, bias=False)
        self.fc_k = nn.Linear(dim_k, n_dim, bias=False)
        self.fc_v = nn.Linear(dim_v, n_dim)
        self.fc_final = nn.Linear(n_dim, n_dim)

    def forward(self, h_q, h_k, h_v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        bs = h_q.shape[0]
        q = self.fc_q(h_q).view(bs, -1, self.n_heads, self.attn_dim).transpose(1, 2)
        k_T = self.fc_k(h_k).view(bs, -1, self.n_heads, self.attn_dim).permute(0, 2, 3, 1)
        v = self.fc_v(h_v).view(bs, -1, self.n_heads, self.attn_dim).transpose(1, 2)
        alpha = th.matmul(q / self.temperature, k_T)
        if self.isSoftmax:
            alpha = F.softmax(alpha, dim=-1)
        else:
            sparsemax = Sparsemax(dim=-1)
            alpha = sparsemax(alpha)
        if self.verbose:
            self.alpha = alpha.squeeze(2).detach()
        res = th.matmul(alpha, v).transpose(1, 2).reshape(bs, -1, self.attn_dim * self.n_heads)
        res = self.fc_final(res)
        return res

class MACCAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MACCAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # enemy info
        if args.env == 'sc2':
            self.n_enemies = args.state_info_dic[self.args.env_args['map_name']]['n_enemies']
        elif args.env == 'foraging':
            self.n_enemies = args.env_args['max_food']
        elif args.env == 'stag_hunt':
            self.n_enemies = args.env_args['n_stags'] + args.env_args['n_hare']
        if args.mixer == 'vdn' and args.env == 'sc2':
            self.nf_en = 5 * args.state_info_dic[self.args.env_args['map_name']]['nf_en']
        elif args.mixer == 'vdn' and args.env == 'foraging':
            self.nf_en = 15
        elif args.mixer == 'vdn' and args.env == 'stag_hunt':
            self.nf_en = 25
        else:
            self.nf_en = args.rnn_hidden_dim_enemy

        if self.args.inseparable:
            self.n_enemies = 1
            self.nf_en = args.rnn_hidden_state

        self.latent_dim = args.latent_dim
        self.bs = 0

        self.embed_fc_input_size = input_shape
        NN_HIDDEN_SIZE = args.NN_HIDDEN_SIZE
        activation_func = nn.LeakyReLU()

        # print('input shape:', input_shape) # input_shape: 96
        # self.embed_net: encoder to generate parameters of n Guassian distributions
        # args.latent_dim: dimension of Guassian distribution (*2: mu + var)
        self.embed_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
                                       nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       activation_func, 
                                       nn.Linear(NN_HIDDEN_SIZE, self.n_enemies * args.latent_dim * 2))

        # self.inference_net: variational estimator q, input: oj, tau_i
        # self.inference_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + args.rnn_hidden_dim, NN_HIDDEN_SIZE),
        #                                    nn.BatchNorm1d(NN_HIDDEN_SIZE),
        #                                    activation_func,
        #                                    nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))
        self.latent = th.rand(args.n_agents, self.n_enemies * args.latent_dim * 2)  # (n, n*(mu+var))
        # self.latent_infer = th.rand(args.n_agents, args.latent_dim * 2)  # (n, (mu+var))

        if self.args.inseparable:
            self.decoder = nn.Sequential(nn.Linear(args.latent_dim, 32),
                                        activation_func, 
                                        nn.Linear(32, args.rnn_hidden_state))
        else:
            self.decoder = nn.Sequential(nn.Linear(args.latent_dim, 32),
                                        activation_func, 
                                        nn.Linear(32, args.rnn_hidden_dim_enemy))

        self.atten = MultiHeadAttention(args.num_heads, args.attn_dim, self.args.softTemperature, args.rnn_hidden_dim, args.rnn_hidden_dim, args.latent_dim, verbose=True, isSoftmax=args.isSoftmax)
        # embedding of the predicted observation of other agents for attention
        self.embedding_for_attention = nn.Linear(args.latent_dim, args.rnn_hidden_dim)
        
        # self.pre_q_net = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + args.attn_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_latent(self, bs):
        self.bs = bs
        return None

    def forward(self, inputs, enemy_states, hidden_state, train_mode=False):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.view(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        # h_detach = h.clone().detach()

        # self.latent: parameters of n Guassian distributions (mu+var)
        self.latent = self.embed_net(h)
        if train_mode:
            self.latent[:, -self.n_enemies * self.latent_dim:] = th.clamp(
                th.exp(self.latent[:, -self.n_enemies * self.latent_dim:]),
                min=self.args.var_floor)  # var

            # latent_embed: reshape self.latent
            latent_embed = self.latent.reshape(self.bs * self.n_agents, self.n_enemies * self.latent_dim * 2)

            gaussian_embed = D.Normal(latent_embed[:, :self.n_enemies * self.latent_dim],
                                    (latent_embed[:, self.n_enemies * self.latent_dim:]) ** (1 / 2))
            # latent: c
            latent = gaussian_embed.rsample()
            # latent = th.zeros_like(latent)
            # print(latent.shape)
        else:
            latent = self.latent.reshape(self.bs * self.n_agents, self.n_enemies * self.latent_dim * 2)[:, :self.n_enemies * self.latent_dim]

        latent_i = latent.reshape(self.bs * self.n_agents * self.n_enemies, self.latent_dim)
        # TODO: embedding, attention
        other_hidden = self.embedding_for_attention(latent_i).view(self.bs * self.n_agents, self.n_enemies, -1)
        # mask = th.zeros((self.n_agents, 1, self.n_agents))
        # for i in range(self.n_agents):
        #     mask[i, :, i] = 1
        # ===================上面两行注释掉就是包括自己==========================
        # mask = mask.repeat(self.bs, 1, 1).to(latent.device)
        
        h_agent_alone = h.view(self.bs * self.n_agents, 1, -1)

        # TODO
        # inputs_interactive, weights = self.atten(h_agent_alone, other_hidden, other_hidden)
        inputs_interactive = self.atten(h_agent_alone, other_hidden, latent.view(self.bs * self.n_agents, self.n_enemies, -1))

        # n_agents * batch_size * embed_dim => batch_size * n_agents * embed_dim 
        #                                   => (batch_size * n_agents) * embed_dim
        # to catenate with original inputs
        inputs_interactive = th.reshape(inputs_interactive, (-1, self.args.attn_dim))
        # inputs_interactive = inputs_interactive.to(self.args.device)
        q = self.fc2(th.cat((h, inputs_interactive), dim=-1))

        loss = th.tensor(0.0).to(self.args.device)
        recon_loss = th.tensor(0.0).to(self.args.device)
        sim_loss = th.tensor(0.0).to(self.args.device)
        entropy_loss = None

        if train_mode:
            standard_n_dist = D.Normal(th.zeros_like(latent_embed[:, :self.n_enemies * self.latent_dim]), 
                                       th.ones_like(latent_embed[:, self.n_enemies * self.latent_dim:]))
            regu_loss = kl_divergence(gaussian_embed, standard_n_dist).sum(-1).mean()
            # h_true = h_detach
            # h_true = h_true.view(self.bs, self.n_agents, -1)
            # h_true = h_true.repeat(1, self.n_agents, 1)
            # h_true = h_true.view(self.bs * self.n_agents * self.n_agents, -1)
            # obs_true = obs_true[:, agent_i:agent_i + 1, :].repeat(1, self.n_agents, 1)
            # obs_true = obs_true.view(self.bs * self.n_agents, -1)
            # recon_loss += (F.mse_loss(h_hat, h_true) + self.args.vae_beta * regu_loss)
            recon_loss += self.args.vae_beta * regu_loss

            # enemy similarity loss TODO: check its validity
            # latent_mean = latent_i.reshape(self.bs, self.n_agents, self.n_enemies, self.latent_dim).permute(0, 2, 1, 3)
            # for i in range(self.n_enemies):
            #     temp1 = latent_mean[:, i, :, :].mean(-2, keepdim=True).repeat(1, self.n_agents, 1).reshape(-1, self.latent_dim)
            #     temp2 = latent_mean[:, i, :, :].reshape(-1, self.latent_dim)
            #     sim_loss += F.mse_loss(temp1, temp2)
            # sim_loss /= self.n_enemies

            # q_hat = self.pre_q_net(h_hat)
            s_hat = self.decoder(latent_i)
            s_true = enemy_states
            s_true = s_true.repeat(1, self.n_agents, 1)
            s_true = s_true.view(self.bs * self.n_agents * self.n_enemies, -1)
            recon_loss += F.mse_loss(s_hat, s_true)

            # entropy_loss = self.calculate_entropy(self.atten.alpha)
            if False:
                gaussian_embed_list = []
                # generate n Gaussian distributions
                for agent_i in range(self.n_agents):
                    gaussian_embed = D.Normal(latent_embed[:, agent_i * self.latent_dim:(agent_i + 1) * self.latent_dim],
                                            (latent_embed[:, (self.n_agents + agent_i) * self.latent_dim:
                                                            (self.n_agents + agent_i + 1) * self.latent_dim]) ** (1 / 2))
                    gaussian_embed_list.append(gaussian_embed)
                for agent_i in range(self.n_agents):
                    # infer_fc_input_fix = inputs.view(self.bs, self.n_agents, -1)
                    h = h.view(self.bs, self.n_agents, -1)
                    # h_i = h[:, agent_i:agent_i + 1, :].repeat(1, self.n_agents, 1)
                    # infer_fc_input = th.cat([h_i, infer_fc_input_fix], dim=-1)
                    h_j = h.view(self.bs, self.n_agents, -1)[:, agent_i:agent_i + 1, :].repeat(1, self.n_agents, 1)
                    h_i = h
                    infer_fc_input = th.cat([h_j, h_i], dim=-1)
                    # print(infer_fc_input_fix.shape, infer_fc_input.shape)  # torch.Size([32, 5, 96]) torch.Size([32, 5, 192])
                    # infer_fc_input: input of variational estimator q (tau_i: h_i, o_j:infer_fc_input_fix)
                    infer_fc_input = infer_fc_input.view(self.bs * self.n_agents, -1)
                    # print(self.embed_fc_input_size * 2, infer_fc_input.shape)
                    self.latent_infer = self.inference_net(infer_fc_input)
                    self.latent_infer[:, -self.latent_dim:] = th.clamp(th.exp(self.latent_infer[:, -self.latent_dim:]),
                                                                    min=self.args.var_floor)
                    # gaussian_infer: Gaussian distribution generated by q
                    gaussian_infer = D.Normal(self.latent_infer[:, :self.latent_dim],
                                            (self.latent_infer[:, self.latent_dim:]) ** (1 / 2))

                    # mi_loss: compute KL divergence of p and q
                    mi_loss += kl_divergence(gaussian_embed_list[agent_i], gaussian_infer).sum(dim=-1).mean()
                    # print(mi_loss)

        logs = {}

        recon_loss *= self.args.recon_loss_weight
        sim_loss *=  self.args.similarity_loss_weight

        loss = recon_loss + sim_loss

        # h = h.view(self.bs * self.n_agents, -1)
        # q = self.fc3(th.cat([h, latent], dim=-1))
        # q = q_alone + q_interactive
        return q, h, {"reg_loss": loss, "recon_loss": recon_loss, "sim_loss": sim_loss, "entropy_loss": entropy_loss, "weights": logs}
    
    def calculate_entropy(self, thres):
        entropy_loss = th.tensor(0.0).to(self.args.device)
        thres = thres.view(self.bs, self.n_agents, self.args.num_heads, -1)
        for i in range(self.args.num_heads):
            for j in range(self.n_agents):
                agents_p = thres[:, j, i, ].unsqueeze(2)
                agents_log = th.log10(agents_p)
                entropy_loss -= th.mean(th.matmul(agents_p.transpose(1, 2), agents_log))

        return entropy_loss / self.n_agents / self.args.num_heads
