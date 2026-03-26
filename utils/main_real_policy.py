import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.main_real_profiles import (
    dvfs_mode_for,
    get_device_type,
    max_model_idx_for_device,
    model_depth_ratio_from_idx,
)
from utils.power_manager_real import LOW_BATTERY_THRESHOLD_J, get_device_capacity


DEVICE_TYPE_TO_BOARD_ID = {
    "nano": 0,
    "agx_xavier": 1,
    "orinnanosuper": 2,
}

RUN_ENERGY_BY_DVFS = torch.tensor(
    [
        [[3.6601, 4.0002, 4.3666, 5.0962], [4.4470, 4.8115, 5.3074, 6.3109]],
        [[1.7716, 2.2170, 2.3527, 2.8420], [1.8327, 2.3969, 2.6204, 3.0826]],
        [[0.952895, 1.245961, 1.3711, 1.502065], [1.082965, 1.090585, 1.497655, 1.535977]],
    ],
    dtype=torch.float32,
)

SWITCH_ENERGY_BY_DVFS = torch.tensor(
    [
        [[0.1212478469, 0.7274870813, 3.1221320574, 12.6704], [0.1096352919, 0.6578117512, 2.8231087656, 11.456888]],
        [[0.2826412344, 1.6958474067, 7.2780117871, 29.536009], [0.3667688421, 2.2006130526, 9.4442976842, 38.327344]],
        [[0.3433993301, 2.0603959809, 8.8425327512, 35.88523], [0.3148180861, 1.8889085167, 8.1065657177, 32.89849]],
    ],
    dtype=torch.float32,
)

WAITING_POWER_BY_DVFS = torch.tensor(
    [
        [3.038518, 2.838381],
        [7.504458, 9.642757],
        [4.4, 4.4],
    ],
    dtype=torch.float32,
)


class ActionRNN(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super().__init__()
        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class RoleController(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_roles):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, n_roles)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h = self.rnn(x, hidden_state)
        q_roles = self.fc2(h)
        return q_roles, h


def estimate_idle_drain_joules(device_type, dvfs_label, duration_seconds):
    board_id = DEVICE_TYPE_TO_BOARD_ID[device_type]
    dvfs_level = 0 if str(dvfs_label) == "low" else 1
    power_w = float(WAITING_POWER_BY_DVFS[board_id][dvfs_level].item())
    return power_w * max(0.0, float(duration_seconds))


class MainRealPolicy:
    def __init__(self, bundle_dir, manifest_path=None, device="cpu", num_clients=None):
        self.bundle_dir = Path(bundle_dir).resolve()
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"policy bundle not found: {self.bundle_dir}")

        self.manifest_path = Path(manifest_path).resolve() if manifest_path else self.bundle_dir / "policy_manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"policy manifest not found: {self.manifest_path}")

        self.device = torch.device(device)
        self.manifest = self._load_manifest(self.manifest_path)

        self.n_actions = int(self.manifest["n_actions"])
        self.n_roles = int(self.manifest["n_roles"])
        self.n_agents = int(self.manifest["n_agents"])
        self.obs_shape = int(self.manifest["obs_shape"])
        self.n_models = int(self.manifest["n_models"])
        self.n_dvfs_levels = int(self.manifest["n_dvfs_levels"])
        self.last_action = bool(self.manifest["last_action"])
        self.reuse_network = bool(self.manifest["reuse_network"])
        self.use_hrl = bool(self.manifest["use_hrl"])
        self.rnn_hidden_dim = int(self.manifest["rnn_hidden_dim"])
        self.role_hidden_dim = int(self.manifest["role_hidden_dim"])
        self.role_horizon = int(self.manifest.get("role_horizon", 3))

        if num_clients is not None and int(num_clients) != self.n_agents:
            raise ValueError(
                f"policy bundle n_agents={self.n_agents} does not match num_clients={num_clients}"
            )
        if self.n_actions != self.n_models * self.n_dvfs_levels + 1:
            raise ValueError("policy manifest action mapping is inconsistent with n_models/n_dvfs_levels")

        self.input_dim = self.obs_shape
        if self.last_action:
            self.input_dim += self.n_actions
        if self.reuse_network:
            self.input_dim += self.n_agents

        self.eval_rnn = ActionRNN(self.input_dim, self.rnn_hidden_dim, self.n_actions).to(self.device)
        rnn_path = self.bundle_dir / "low_level_eval_rnn.pt"
        self.eval_rnn.load_state_dict(torch.load(rnn_path, map_location=self.device))
        self.eval_rnn.eval()

        self.role_eval_controller = None
        if self.use_hrl:
            role_path = self.bundle_dir / "high_level_role_eval.pt"
            if not role_path.exists():
                raise FileNotFoundError(f"HRL bundle missing role controller: {role_path}")
            self.role_eval_controller = RoleController(self.input_dim, self.role_hidden_dim, self.n_roles).to(self.device)
            self.role_eval_controller.load_state_dict(torch.load(role_path, map_location=self.device))
            self.role_eval_controller.eval()

        self.reset_runtime_state()

    def _load_manifest(self, manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "n_actions",
            "n_roles",
            "n_agents",
            "obs_shape",
            "n_models",
            "n_dvfs_levels",
            "last_action",
            "reuse_network",
            "rnn_hidden_dim",
            "role_hidden_dim",
            "use_hrl",
        }
        missing = sorted(required.difference(manifest.keys()))
        if missing:
            raise ValueError(f"policy manifest missing fields: {missing}")
        return manifest

    def reset_runtime_state(self):
        self.last_actions = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        self.current_roles = np.full((self.n_agents,), -1, dtype=np.int64)
        self.role_age = np.zeros((self.n_agents,), dtype=np.int64)
        self.eval_hidden = torch.zeros((1, self.n_agents, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        if self.use_hrl:
            self.role_hidden = torch.zeros((1, self.n_agents, self.role_hidden_dim), dtype=torch.float32, device=self.device)
        else:
            self.role_hidden = None

    def get_state(self):
        return {
            "last_actions": self.last_actions.tolist(),
            "current_roles": self.current_roles.tolist(),
            "role_age": self.role_age.tolist(),
            "eval_hidden": self.eval_hidden.detach().cpu(),
            "role_hidden": None if self.role_hidden is None else self.role_hidden.detach().cpu(),
        }

    def load_state(self, state):
        self.last_actions = np.asarray(state["last_actions"], dtype=np.float32)
        self.current_roles = np.asarray(state["current_roles"], dtype=np.int64)
        self.role_age = np.asarray(state["role_age"], dtype=np.int64)
        self.eval_hidden = torch.as_tensor(state["eval_hidden"], dtype=torch.float32, device=self.device).clone()
        role_hidden = state.get("role_hidden")
        if self.use_hrl:
            if role_hidden is None:
                raise ValueError("policy checkpoint missing role_hidden")
            self.role_hidden = torch.as_tensor(role_hidden, dtype=torch.float32, device=self.device).clone()

    def _build_agent_input(self, obs, last_action, agent_num):
        parts = [np.asarray(obs, dtype=np.float32)]
        if self.last_action:
            parts.append(np.asarray(last_action, dtype=np.float32))
        if self.reuse_network:
            agent_id = np.zeros((self.n_agents,), dtype=np.float32)
            agent_id[int(agent_num)] = 1.0
            parts.append(agent_id)
        inputs = np.concatenate(parts, axis=0)
        return torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _build_role_action_ids(self, avail_actions):
        idle_id = self.n_models * self.n_dvfs_levels
        all_ids = np.arange(self.n_actions, dtype=np.int64)
        avail_ids = all_ids[np.asarray(avail_actions, dtype=np.float32) > 0.5]
        return {
            0: avail_ids[avail_ids == idle_id],
            1: avail_ids[(avail_ids >= 0) & (avail_ids < self.n_models)],
            2: avail_ids[(avail_ids >= self.n_models) & (avail_ids < idle_id)],
            3: avail_ids[(avail_ids >= 0) & (avail_ids <= idle_id)],
        }

    def _choose_role(self, obs, last_action, agent_num, avail_actions, epsilon):
        if not self.use_hrl:
            return 3
        inputs = self._build_agent_input(obs, last_action, agent_num)
        hidden_state = self.role_hidden[:, agent_num, :]
        with torch.no_grad():
            q_roles, next_hidden = self.role_eval_controller(inputs, hidden_state)
        self.role_hidden[:, agent_num, :] = next_hidden
        role_to_ids = self._build_role_action_ids(avail_actions)
        valid_roles = [rid for rid, ids in role_to_ids.items() if len(ids) > 0]
        if not valid_roles:
            return 0
        q_np = q_roles.detach().cpu().numpy().reshape(-1)
        if float(epsilon) > 0.0 and np.random.uniform() < float(epsilon):
            return int(np.random.choice(valid_roles))
        return int(max(valid_roles, key=lambda rid: q_np[rid]))

    def _choose_action_with_role(self, obs, last_action, agent_num, avail_actions, epsilon, role_id):
        inputs = self._build_agent_input(obs, last_action, agent_num)
        hidden_state = self.eval_hidden[:, agent_num, :]
        with torch.no_grad():
            q_values, next_hidden = self.eval_rnn(inputs, hidden_state)
        self.eval_hidden[:, agent_num, :] = next_hidden
        q_np = q_values.detach().cpu().numpy().reshape(-1)
        role_to_ids = self._build_role_action_ids(avail_actions)
        actual_role = int(role_id)
        candidate_ids = role_to_ids.get(actual_role, np.array([], dtype=np.int64))
        if len(candidate_ids) == 0:
            actual_role = 3
            candidate_ids = role_to_ids.get(actual_role, np.array([], dtype=np.int64))
        if len(candidate_ids) == 0:
            actual_role = 0
            candidate_ids = role_to_ids.get(actual_role, np.array([], dtype=np.int64))
        if len(candidate_ids) == 0:
            action_id = int(self.n_actions - 1)
            return action_id, float(q_np[action_id]), actual_role
        if float(epsilon) > 0.0 and np.random.uniform() < float(epsilon):
            action_id = int(np.random.choice(candidate_ids))
        else:
            action_id = int(candidate_ids[np.argmax(q_np[candidate_ids])])
        return action_id, float(q_np[action_id]), actual_role

    def _build_model_mask(self, device_type, battery_joules):
        available = np.zeros((self.n_models,), dtype=np.float32)
        if float(battery_joules) <= float(LOW_BATTERY_THRESHOLD_J):
            return available
        board_id = DEVICE_TYPE_TO_BOARD_ID[device_type]
        device_max_model_idx = max_model_idx_for_device(device_type)
        for model_id in range(device_max_model_idx - 1, -1, -1):
            if float(battery_joules) - 2.0 * float(SWITCH_ENERGY_BY_DVFS[board_id][0][model_id].item()) + float(
                RUN_ENERGY_BY_DVFS[board_id][0][model_id].item()
            ) > 0.0:
                available[: model_id + 1] = 1.0
                break
        return available

    def _build_avail_actions(self, device_type, battery_joules):
        avail_actions = np.zeros((self.n_actions,), dtype=np.float32)
        board_id = DEVICE_TYPE_TO_BOARD_ID[device_type]
        device_max_model_idx = max_model_idx_for_device(device_type)
        for dvfs_level in range(self.n_dvfs_levels):
            for model_id in range(device_max_model_idx):
                action_id = dvfs_level * self.n_models + model_id
                energy_cost = (
                    2.0 * float(SWITCH_ENERGY_BY_DVFS[board_id][dvfs_level][model_id].item())
                    + float(RUN_ENERGY_BY_DVFS[board_id][dvfs_level][model_id].item())
                )
                if float(battery_joules) - energy_cost > 0.0:
                    avail_actions[action_id] = 1.0
        avail_actions[-1] = 1.0
        return avail_actions

    def _build_obs(self, battery_ratio, model_mask):
        obs = np.zeros((self.obs_shape,), dtype=np.float32)
        obs[0] = float(battery_ratio)
        usable = min(self.n_models, self.obs_shape - 1)
        for model_id in range(usable):
            obs[1 + model_id] = float(model_mask[model_id])
        return obs

    def decode_action(self, action_id, device_type):
        idle_action = self.n_models * self.n_dvfs_levels
        if int(action_id) >= idle_action:
            return {
                "selected": False,
                "model_idx": None,
                "model_depth_ratio": None,
                "dvfs_label": "idle",
                "dvfs_mode": None,
            }
        model_id = int(action_id) % self.n_models
        dvfs_level = int(action_id) // self.n_models
        dvfs_label = "low" if dvfs_level == 0 else "high"
        return {
            "selected": True,
            "model_idx": int(model_id + 1),
            "model_depth_ratio": float(model_depth_ratio_from_idx(model_id + 1)),
            "dvfs_label": dvfs_label,
            "dvfs_mode": int(dvfs_mode_for(device_type, dvfs_label)),
        }

    def plan_round(self, active_clients, battery_state_joules, epsilon=0.0):
        plans = {}
        for cid in active_clients:
            cid = int(cid)
            device_type = get_device_type(cid)
            battery_joules = float(battery_state_joules[cid])
            model_mask = self._build_model_mask(device_type, battery_joules)
            avail_actions = self._build_avail_actions(device_type, battery_joules)
            obs = self._build_obs(max(0.0, battery_joules) / 150000.0, model_mask)

            if float(battery_joules) <= float(LOW_BATTERY_THRESHOLD_J):
                action_id = int(self.n_actions - 1)
                role_id = 0
            else:
                need_new_role = int(self.current_roles[cid]) < 0 or int(self.role_age[cid]) >= int(self.role_horizon)
                if self.use_hrl and need_new_role:
                    self.current_roles[cid] = self._choose_role(
                        obs=obs,
                        last_action=self.last_actions[cid],
                        agent_num=cid,
                        avail_actions=avail_actions,
                        epsilon=epsilon,
                    )
                    self.role_age[cid] = 0
                elif not self.use_hrl and int(self.current_roles[cid]) < 0:
                    self.current_roles[cid] = 3

                action_id, _, role_id = self._choose_action_with_role(
                    obs=obs,
                    last_action=self.last_actions[cid],
                    agent_num=cid,
                    avail_actions=avail_actions,
                    epsilon=epsilon,
                    role_id=int(self.current_roles[cid]),
                )
                self.current_roles[cid] = int(role_id)
                self.role_age[cid] += 1

            self.last_actions[cid].fill(0.0)
            self.last_actions[cid][int(action_id)] = 1.0
            decoded = self.decode_action(action_id, device_type)
            plans[cid] = {
                "cid": cid,
                "device_type": device_type,
                "battery_joules": battery_joules,
                "battery_ratio": max(0.0, battery_joules) / float(get_device_capacity(device_type)),
                "obs": obs.tolist(),
                "model_mask": model_mask.tolist(),
                "avail_actions": avail_actions.tolist(),
                "role_id": int(self.current_roles[cid]) if decoded["selected"] else 0,
                "action_id": int(action_id),
                **decoded,
            }
        return plans
