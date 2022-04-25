import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_hidden import QMixerHidden
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop

class LatentQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "qmix_hidden":
                self.mixer = QMixerHidden(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        # modify st, to store history information in st
        if args.inseparable:
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
            self.fc_al = nn.Linear(self.nf_al + self.args.n_agents, args.rnn_hidden_dim_ally).to(self.args.device)
            self.rnn_al = nn.GRUCell(args.rnn_hidden_dim_ally, args.rnn_hidden_dim_ally).to(self.args.device)
            self.fc_en = nn.Linear(self.nf_en * self.n_enemies, args.rnn_hidden_state).to(self.args.device)
            self.rnn_en = nn.GRUCell(args.rnn_hidden_state, args.rnn_hidden_state).to(self.args.device)
        else:
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
            self.fc_al = nn.Linear(self.nf_al + self.args.n_agents, args.rnn_hidden_dim_ally).to(self.args.device)
            self.rnn_al = nn.GRUCell(args.rnn_hidden_dim_ally, args.rnn_hidden_dim_ally).to(self.args.device)
            self.fc_en = nn.Linear(self.nf_en, args.rnn_hidden_dim_enemy).to(self.args.device)
            self.rnn_en = nn.GRUCell(args.rnn_hidden_dim_enemy, args.rnn_hidden_dim_enemy).to(self.args.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        self.mac.init_latent(batch.batch_size)

        # init hidden state for rnn in st
        if self.args.inseparable:
            self.hidden_fc_al = self.fc_al.weight.new(1, self.args.rnn_hidden_dim_ally).zero_()[0].unsqueeze(0).expand(batch.batch_size, self.n_agents, -1)
            self.hidden_fc_en = self.fc_en.weight.new(1, self.args.rnn_hidden_state).zero_()[0].unsqueeze(0).expand(batch.batch_size, 1, -1)
        else:
            self.hidden_fc_al = self.fc_al.weight.new(1, self.args.rnn_hidden_dim_ally).zero_()[0].unsqueeze(0).expand(batch.batch_size, self.n_agents, -1)
            self.hidden_fc_en = self.fc_en.weight.new(1, self.args.rnn_hidden_dim_enemy).zero_()[0].unsqueeze(0).expand(batch.batch_size, self.n_enemies, -1)
        
        if self.args.mixer is not None:
            if self.args.mixer == "vdn":
                self.new_states = th.zeros(batch.batch_size, batch.max_seq_length, batch["state"].size()[2] + self.n_agents * (5 * self.nf_al + self.n_agents - self.nf_al) + self.n_enemies * (5 * self.nf_en - self.nf_en)).to(self.args.device)
                for i in range(self.n_agents):
                    self.new_states[:, :, i * (5 * self.nf_al + self.n_agents) + 5 * self.nf_al + i] = 1
            else:
                if self.args.inseparable:
                    self.new_states = th.zeros(batch.batch_size, batch.max_seq_length, batch["state"].size()[2] + self.n_agents * (self.args.rnn_hidden_dim_ally - self.nf_al) + self.args.rnn_hidden_state - self.n_enemies * self.nf_en).to(self.args.device)
                else:
                    self.new_states = th.zeros(batch.batch_size, batch.max_seq_length, batch["state"].size()[2] + self.n_agents * (self.args.rnn_hidden_dim_ally - self.nf_al) + self.n_enemies * (self.args.rnn_hidden_dim_enemy - self.nf_en)).to(self.args.device)

        reg_loss = 0
        recon_loss = 0
        sim_loss = 0
        entropy_loss = 0
        reg_flag = False
        recon_flag = False
        mi_flag = False
        entropy_flag = False
        weights = []
        for t in range(batch.max_seq_length):
            # modify st, get new state
            if self.args.mixer is not None:
                if self.args.mixer == "vdn":
                    inputs_al = batch["state"][:, t, :self.nf_al * self.n_agents].reshape(batch.batch_size, self.n_agents, -1)
                    inputs_en = batch["state"][:, t, self.nf_al * self.n_agents:self.nf_al * self.n_agents + self.n_enemies * self.nf_en].reshape(batch.batch_size, self.n_enemies, self.nf_en)
                    for i in range(self.n_agents):
                        if t > 0:
                            self.new_states[:, t, i * (5 * self.nf_al + self.n_agents) + self.nf_al:i * (5 * self.nf_al + self.n_agents) + 5 * self.nf_al] = \
                                self.new_states[:, t - 1, i * (5 * self.nf_al + self.n_agents):i * (5 * self.nf_al + self.n_agents) + 4 * self.nf_al].clone()
                        self.new_states[:, t, i * (5 * self.nf_al + self.n_agents):i * (5 * self.nf_al + self.n_agents) + self.nf_al] = \
                            batch["state"][:, t, i * self.nf_al:(i + 1) * self.nf_al]
                    for i in range(self.n_enemies):
                        if t > 0:
                            self.new_states[:, t, self.n_agents * (5 * self.nf_al + self.n_agents) + i * (5 * self.nf_en) + self.nf_en:self.n_agents * (5 * self.nf_al + self.n_agents) + i * (5 * self.nf_en) + 5 * self.nf_en] = \
                                self.new_states[:, t - 1, self.n_agents * (5 * self.nf_al + self.n_agents) + i * (5 * self.nf_en):self.n_agents * (5 * self.nf_al + self.n_agents) + i * (5 * self.nf_en) + 4 * self.nf_en].clone()
                        self.new_states[:, t, self.n_agents * (5 * self.nf_al + self.n_agents) + i * (5 * self.nf_en):self.n_agents * (5 * self.nf_al + self.n_agents) + i * (5 * self.nf_en) + self.nf_en] = \
                            batch["state"][:, t, self.n_agents * self.nf_al + i * self.nf_en:self.n_agents * self.nf_al + (i + 1) * self.nf_en]
                    self.new_states[:, t, self.n_agents * (5 * self.nf_al + self.n_agents) + self.n_enemies * 5 * self.nf_en:] =\
                         batch["state"][:, t, self.nf_al * self.n_agents + self.n_enemies * self.nf_en:].reshape(batch.batch_size, -1)
                else:
                    if self.args.inseparable:
                        inputs_al = batch["state"][:, t, :self.nf_al * self.n_agents].reshape(batch.batch_size, self.n_agents, -1)
                        inputs_al = th.cat((inputs_al, th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch.batch_size, -1, -1)), -1)
                        inputs_al = inputs_al.reshape(batch.batch_size * self.n_agents, -1)
                        inputs_en = batch["state"][:, t, self.nf_al * self.n_agents:self.nf_al * self.n_agents + self.n_enemies * self.nf_en].reshape(batch.batch_size, self.n_enemies * self.nf_en)
                        
                        x_al = F.relu(self.fc_al(inputs_al))
                        h_in_al = self.hidden_fc_al.view(-1, self.args.rnn_hidden_dim_ally)
                        self.hidden_fc_al = self.rnn_al(x_al, h_in_al)
                        h_al = self.hidden_fc_al.view(-1, self.n_agents * self.args.rnn_hidden_dim_ally)
                        self.new_states[:, t, :self.n_agents * self.args.rnn_hidden_dim_ally] = h_al

                        x_en = F.relu(self.fc_en(inputs_en))
                        h_in_en = self.hidden_fc_en.view(-1, self.args.rnn_hidden_state)
                        self.hidden_fc_en = self.rnn_en(x_en, h_in_en)
                        h_en = self.hidden_fc_en.view(-1, self.args.rnn_hidden_state)
                        self.new_states[:, t, self.n_agents * self.args.rnn_hidden_dim_ally:self.n_agents * self.args.rnn_hidden_dim_ally + self.args.rnn_hidden_state] = h_en
                        self.new_states[:, t, self.n_agents * self.args.rnn_hidden_dim_ally + self.args.rnn_hidden_state:] = batch["state"][:, t, self.nf_al * self.n_agents + self.n_enemies * self.nf_en:].reshape(batch.batch_size, -1)
                    else:
                        inputs_al = batch["state"][:, t, :self.nf_al * self.n_agents].reshape(batch.batch_size, self.n_agents, -1)
                        inputs_al = th.cat((inputs_al, th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch.batch_size, -1, -1)), -1)
                        inputs_al = inputs_al.reshape(batch.batch_size * self.n_agents, -1)
                        inputs_en = batch["state"][:, t, self.nf_al * self.n_agents:self.nf_al * self.n_agents + self.n_enemies * self.nf_en].reshape(batch.batch_size * self.n_enemies, self.nf_en)
                        
                        x_al = F.relu(self.fc_al(inputs_al))
                        h_in_al = self.hidden_fc_al.view(-1, self.args.rnn_hidden_dim_ally)
                        self.hidden_fc_al = self.rnn_al(x_al, h_in_al)
                        h_al = self.hidden_fc_al.view(-1, self.n_agents * self.args.rnn_hidden_dim_ally)
                        self.new_states[:, t, :self.n_agents * self.args.rnn_hidden_dim_ally] = h_al

                        x_en = F.relu(self.fc_en(inputs_en))
                        h_in_en = self.hidden_fc_en.view(-1, self.args.rnn_hidden_dim_enemy)
                        self.hidden_fc_en = self.rnn_en(x_en, h_in_en)
                        h_en = self.hidden_fc_en.view(-1, self.n_enemies * self.args.rnn_hidden_dim_enemy)
                        self.new_states[:, t, self.n_agents * self.args.rnn_hidden_dim_ally:self.n_agents * self.args.rnn_hidden_dim_ally + self.n_enemies * self.args.rnn_hidden_dim_enemy] = h_en
                        self.new_states[:, t, self.n_agents * self.args.rnn_hidden_dim_ally + self.n_enemies * self.args.rnn_hidden_dim_enemy:] = batch["state"][:, t, self.nf_al * self.n_agents + self.n_enemies * self.nf_en:].reshape(batch.batch_size, -1)
            
            if self.args.inseparable:
                agent_outs, loss = self.mac.forward(batch, self.hidden_fc_en, t=t, train_mode=True)
            else:
                agent_outs, loss = self.mac.forward(batch, self.hidden_fc_en, t=t, train_mode=True)
            if loss["reg_loss"] is not None:
                reg_loss += loss["reg_loss"]
                reg_flag = True
            if loss["recon_loss"] is not None:
                recon_loss += loss["recon_loss"]
                recon_flag = True
            if loss["sim_loss"] is not None:
                sim_loss += loss["sim_loss"]
                mi_flag = True
            if loss["entropy_loss"] is not None:
                entropy_loss += loss["entropy_loss"]
                entropy_flag = True
            mac_out.append(agent_outs)

        reg_loss /= batch.max_seq_length
        recon_loss /= batch.max_seq_length
        sim_loss /= batch.max_seq_length
        entropy_loss /= batch.max_seq_length

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        self.target_mac.init_latent(batch.batch_size)  # (bs,n,latent_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix (use new state)
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, self.new_states[:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, self.new_states[:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if hasattr(self.args, 'gdx_weight'):
            loss = loss * self.args.gdx_weight / th.sqrt(reg_loss + 1e-8) + reg_loss
        else:
            loss += reg_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            if reg_flag:
                self.logger.log_stat("reg_loss", reg_loss.item(), t_env)
            if recon_flag:
                self.logger.log_stat("recon_loss", recon_loss.item(), t_env)
            if mi_flag:
                self.logger.log_stat("sim_loss", sim_loss.item(), t_env)
            if entropy_flag:
                self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            
            self.log_stats_t = t_env

    def _log_from_mac_out(self, logs, t):
        if len(logs) == 0: 
            return 
        keys = list(logs[0].keys())
        for k in keys:
            value = th.stack([l[k] for l in logs])
            self.logger.log_histogram(k, value, t)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
