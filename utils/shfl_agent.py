import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from loguru import logger

# ==========================================
# 1. 网络定义 (直接复用学长的代码)
# ==========================================

class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states): 
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents) 
        states = states.reshape(-1, self.args.state_shape) 

        w1 = torch.abs(self.hyper_w1(states)) 
        b1 = self.hyper_b1(states) 

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim) 
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim) 

        hidden = F.elu(torch.bmm(q_values, w1) + b1) 

        w2 = torch.abs(self.hyper_w2(states)) 
        b2 = self.hyper_b2(states) 

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1) 
        b2 = b2.view(-1, 1, 1) 

        q_total = torch.bmm(hidden, w2) + b2 
        q_total = q_total.view(episode_num, -1, 1) 
        return q_total

# ==========================================
# 2. 参数配置类 (模拟 args)
# ==========================================
class SHFL_Args:
    def __init__(self, n_agents):
        # [⚠️ 重要] 这些参数必须与学长训练模型时的设置完全一致！
        # 如果加载 .pkl 报错尺寸不匹配，请调整这里
        
        self.n_agents = n_agents
        
        # 动作空间: Model 1-4 + 不参与 = 5个动作? 还是只有4个模型?
        # 通常 SHFL 包含一个 "不选" 的动作。
        # 假设: 0=不选, 1=Model1, 2=Model2, 3=Model3, 4=Model4
        self.n_actions = 4
        
        # 神经网络维度 (常见默认值，报错需修改)
        self.rnn_hidden_dim = 64
        self.qmix_hidden_dim = 32
        self.two_hyper_layers = False 
        self.hyper_hidden_dim = 64
        
        # 状态维度
        # state_shape 通常是所有 agent 的 obs 拼接，或者全局信息
        # SHFL 论文 Eq.12: state = [L, C, E, r] (4维) * n_agents ?
        # 这里暂时给一个估计值，如果报错我们需要调整
        self.state_shape = 3 * n_agents 
        
        # RNN 输入维度
        # Input = Observation + Last_Action(OneHot) + Agent_ID(OneHot)
        # Obs = [L, C, E] (3维)
        self.obs_shape = 3
        self.input_shape = self.obs_shape + self.n_actions + self.n_agents

# ==========================================
# 3. 智能体管理器 (核心逻辑)
# ==========================================
class SHFLAgentManager:
    def __init__(self, n_agents, model_dir, device='cpu'):
        self.args = SHFL_Args(n_agents)
        self.device = device
        self.n_agents = n_agents
        
        # 初始化 RNN 智能体
        self.eval_rnn = RNN(self.args.input_shape, self.args).to(device)
        
        # 加载权重
        rnn_path = os.path.join(model_dir, 'latest_rnn_net_params.pkl')
        if os.path.exists(rnn_path):
            try:
                # map_location='cpu' 防止 GPU 显存不够或不匹配
                self.eval_rnn.load_state_dict(torch.load(rnn_path, map_location=device))
                logger.success(f"✅ Loaded SHFL Agent weights from {rnn_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load Agent weights: {e}")
                logger.error("请检查 SHFL_Args 中的 hidden_dim 或 input_shape 是否与 .pkl 文件匹配！")
                raise e
        else:
            logger.warning(f"⚠️ Weights not found at {rnn_path}, using random initialization!")

        self.eval_rnn.eval()
        
        # 初始化隐藏状态 (每个 Agent 一个)
        # Shape: [n_agents, rnn_hidden_dim]
        self.hidden_states = torch.zeros(self.n_agents, self.args.rnn_hidden_dim).to(device)
        
        # 记录上一次的动作 (One-hot), 初始全 0
        self.last_actions = torch.zeros(self.n_agents, self.args.n_actions).to(device)
    
    def select_models(self, observations,device_types, round_num=None):
        """
        根据当前观测值，返回每个 Client 的最佳动作及其 Q 值。
        Returns:
            client_decisions: list of dict, 每个元素包含:
                              {'client_idx': int, 'action': int, 'q_value': float}
        """
        if len(observations) != self.n_agents:
            logger.warning(f"Observation length ({len(observations)}) != Real Agents ({self.n_agents})")

        client_decisions = []
        
        with torch.no_grad():
            for i in range(self.n_agents):
                # Obs: [L, C, E]
                obs_data = observations[i]
                obs_vec = torch.tensor([
                    obs_data['L'], obs_data['C'], obs_data['E']
                ], dtype=torch.float32).to(self.device)
                
                last_act_vec = self.last_actions[i]
                
                id_vec = torch.zeros(self.n_agents).to(self.device)
                id_vec[i] = 1.0 
                
                input_tensor = torch.cat([obs_vec, last_act_vec, id_vec], dim=0).unsqueeze(0) 
                
                # ... (前向传播) ...
                h_in = self.hidden_states[i].unsqueeze(0)
                q_values, h_out = self.eval_rnn(input_tensor, h_in)
                
                self.hidden_states[i] = h_out.squeeze(0)
                
                # [🔥 关键修改: 获取 Q 值]
                # q_values 是一个 [1, 4] 的向量，对应 Model 1-4 的分数
                # 我们选分最高的那个
                max_q, action_idx = torch.max(q_values, dim=1)
                
                action = action_idx.item()      # 0-3
                max_q_val = max_q.item()        # 对应的最高分数
                
                # 记录决策信息
                client_decisions.append({
                    'client_idx': i,
                    'action': action + 1,  # 映射为 Model 1-4
                    'q_value': max_q_val
                })
                
                # 更新 Last Action
                self.last_actions[i].zero_()
                self.last_actions[i][action] = 1.0
                
        return client_decisions
    
    # ==========================================
    # [新增] 断点续训支持
    # ==========================================
    def get_state(self):
        """保存RL Agent的完整状态（用于断点续训）"""
        return {
            'hidden_states': self.hidden_states.clone(),
            'last_actions': self.last_actions.clone(),
            'rnn_state_dict': self.eval_rnn.state_dict()
        }
    
    def load_state(self, state):
        """加载RL Agent的状态（断点恢复）"""
        self.hidden_states = state['hidden_states'].to(self.device)
        self.last_actions = state['last_actions'].to(self.device)
        self.eval_rnn.load_state_dict(state['rnn_state_dict'])
        logger.info("✅ RL Agent state restored from checkpoint")
