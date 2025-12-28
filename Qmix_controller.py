import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class RNN(nn.Module):
    """QMIX中的RNN网络，每个客户端一个"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # GRU层
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def init_hidden(self):
        # 初始化隐藏状态
        return torch.zeros(1, self.hidden_dim)
    
    def forward(self, x, hidden_state):
        # x: [batch_size, input_dim]
        # hidden_state: [batch_size, hidden_dim]
        
        # GRU更新隐藏状态
        h = self.gru(x, hidden_state)
        
        # 输出动作Q值
        q = self.fc(h)
        
        return q, h

class MixingNetwork(nn.Module):
    """QMIX混合网络，整合所有客户端的Q值"""
    def __init__(self, num_agents, state_dim, hidden_dim):
        super(MixingNetwork, self).__init__()
        
        # 超参数网络生成权重
        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * num_agents)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        
        # 超参数网络生成偏置
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)
        
        # 确保权重非负
        self.hyper_w1.weight.data.normal_(0, 0.1)
        self.hyper_w2.weight.data.normal_(0, 0.1)
        self.hyper_b1.weight.data.normal_(0, 0.1)
        self.hyper_b2.weight.data.normal_(0, 0.1)
        
    def forward(self, agent_qs, states):
        # agent_qs: [batch_size, num_agents]
        # states: [batch_size, state_dim]
        
        batch_size = agent_qs.size(0)
        num_agents = agent_qs.size(1)
        
        # 第一层权重和偏置
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        
        w1 = w1.view(batch_size, num_agents, -1)
        
        # 第一层混合
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1).squeeze(1) + b1)
        
        # 第二层权重和偏置
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        
        w2 = w2.view(batch_size, -1, 1)
        
        # 第二层混合
        q_tot = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        
        return q_tot

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, state, action, reward, next_obs, next_state, done):
        self.buffer.append((obs, state, action, reward, next_obs, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, state, action, reward, next_obs, next_state, done = zip(*batch)
        
        return obs, state, action, reward, next_obs, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class QMIXController:
    """QMIX控制器"""
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=64, buffer_size=10000, gamma=0.99, device=None):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 0.3  # 探索率
        
        # 创建每个客户端的RNN网络
        self.agents = [RNN(obs_dim, hidden_dim, action_dim).to(self.device) for _ in range(num_agents)]
        
        # 创建目标网络
        self.target_agents = [RNN(obs_dim, hidden_dim, action_dim).to(self.device) for _ in range(num_agents)]
        
        # 初始化目标网络
        for i in range(num_agents):
            self.target_agents[i].load_state_dict(self.agents[i].state_dict())
        
        # 混合网络
        self.mixer = MixingNetwork(num_agents, 2, hidden_dim).to(self.device)  # 全局状态维度为2
        self.target_mixer = MixingNetwork(num_agents, 2, hidden_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # 优化器
        self.agent_params = []
        for i in range(num_agents):
            self.agent_params += list(self.agents[i].parameters())
        
        self.mixer_params = list(self.mixer.parameters())
        self.params = self.agent_params + self.mixer_params
        
        self.optimizer = torch.optim.Adam(self.params, lr=0.001)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 隐藏状态
        self.hidden_states = [agent.init_hidden().to(self.device) for agent in self.agents]
        
    def reset_hidden_states(self):
        """重置RNN隐藏状态"""
        self.hidden_states = [agent.init_hidden().to(self.device) for agent in self.agents]
    
    def select_actions(self, observations, global_state=None, epsilon=None):
        """选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
            
        # 确定是随机探索还是按策略选择
        if random.random() < epsilon:
            # 随机动作
            actions = [random.randint(0, self.action_dim - 1) for _ in range(self.num_agents)]
        else:
            actions = []
            
            # 遍历每个客户端
            for i in range(self.num_agents):
                # 将观察转换为张量
                obs = torch.FloatTensor([observations[i]]).to(self.device)
                
                # 使用RNN计算Q值
                with torch.no_grad():
                    q_values, self.hidden_states[i] = self.agents[i](obs, self.hidden_states[i])
                
                # 选择最大Q值的动作
                action = q_values.max(dim=1)[1].item()
                actions.append(action)
        
        return actions
    
    def store_transition(self, obs, state, action, reward, next_obs, next_state, done):
        """存储经验"""
        self.replay_buffer.push(obs, state, action, reward, next_obs, next_state, done)
    
    def train(self, epochs=1, batch_size=32):
        """训练QMIX网络"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 训练多个epoch
        for _ in range(epochs):
            # 从经验池中采样
            obs_batch, state_batch, action_batch, reward_batch, next_obs_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
            
            # 转换为张量
            obs = [[torch.FloatTensor([obs_batch[b][i]]).to(self.device) for b in range(batch_size)] for i in range(self.num_agents)]
            state = [torch.FloatTensor([state_batch[b]]).to(self.device) for b in range(batch_size)]
            actions = [[action_batch[b][i] for b in range(batch_size)] for i in range(self.num_agents)]
            rewards = torch.FloatTensor([reward_batch]).t().to(self.device)
            next_obs = [[torch.FloatTensor([next_obs_batch[b][i]]).to(self.device) for b in range(batch_size)] for i in range(self.num_agents)]
            next_state = [torch.FloatTensor([next_state_batch[b]]).to(self.device) for b in range(batch_size)]
            dones = torch.FloatTensor([done_batch]).t().to(self.device)
            
            # 计算当前Q值
            agent_outs = []
            for i in range(self.num_agents):
                agent_out = []
                h = torch.zeros(batch_size, self.hidden_dim).to(self.device)
                
                for b in range(batch_size):
                    q, h[b:b+1] = self.agents[i](obs[i][b], h[b:b+1])
                    agent_out.append(q)
                
                agent_out = torch.cat(agent_out, dim=0)
                agent_outs.append(agent_out)
            
            # 获取选择的动作的Q值
            chosen_action_qvals = []
            for i in range(self.num_agents):
                chosen_action_qval = agent_outs[i].gather(1, torch.LongTensor(actions[i]).unsqueeze(1).to(self.device)).squeeze(1)
                chosen_action_qvals.append(chosen_action_qval)
            
            chosen_action_qvals = torch.stack(chosen_action_qvals, dim=1)
            
            # 混合得到联合Q值
            state_tensor = torch.cat(state, dim=0)
            q_tot = self.mixer(chosen_action_qvals, state_tensor)
            
            # 计算目标Q值
            target_agent_outs = []
            for i in range(self.num_agents):
                target_agent_out = []
                h = torch.zeros(batch_size, self.hidden_dim).to(self.device)
                
                for b in range(batch_size):
                    q, h[b:b+1] = self.target_agents[i](next_obs[i][b], h[b:b+1])
                    target_agent_out.append(q)
                
                target_agent_out = torch.cat(target_agent_out, dim=0)
                target_agent_outs.append(target_agent_out)
            
            # 获取最大Q值的动作
            target_max_qvals = []
            for i in range(self.num_agents):
                target_max_qvals.append(target_agent_outs[i].max(dim=1)[0])
            
            target_max_qvals = torch.stack(target_max_qvals, dim=1)
            
            # 混合得到目标联合Q值
            next_state_tensor = torch.cat(next_state, dim=0)
            target_max_qval = self.target_mixer(target_max_qvals, next_state_tensor)
            
            # 计算目标值
            targets = rewards + self.gamma * (1 - dones) * target_max_qval
            
            # 计算TD误差
            td_error = (q_tot - targets.detach())
            loss = (td_error ** 2).mean()
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.params, 10)
            
            self.optimizer.step()
            
            # 更新目标网络
            if random.random() < 0.01:  # 以1%的概率更新目标网络
                for i in range(self.num_agents):
                    self.target_agents[i].load_state_dict(self.agents[i].state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def save(self, path):
        """保存模型"""
        model_dict = {
            'agents': [agent.state_dict() for agent in self.agents],
            'mixer': self.mixer.state_dict()
        }
        torch.save(model_dict, path)
    
    def load(self, path):
        """加载模型"""
        model_dict = torch.load(path)
        
        for i, agent_dict in enumerate(model_dict['agents']):
            self.agents[i].load_state_dict(agent_dict)
            self.target_agents[i].load_state_dict(agent_dict)
        
        self.mixer.load_state_dict(model_dict['mixer'])
        self.target_mixer.load_state_dict(model_dict['mixer'])