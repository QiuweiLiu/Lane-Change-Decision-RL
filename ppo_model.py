import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, use_attention=False, attn_dims=None):
        super(PPOActorCritic, self).__init__()
        self.use_attention = use_attention
        self.main_dim = attn_dims["main_dim"]
        self.veh_dim = attn_dims["veh_dim"]
        self.veh_count = attn_dims["veh_count"]

        # 如果启用 attention 机制
        if use_attention:
            # 主车和车辆的嵌入层
            self.main_embed = nn.Linear(self.main_dim, hidden_size)
            self.veh_embed = nn.Linear(self.veh_dim, hidden_size)
            # 计算最终输入特征大小
            self.fc_final = nn.Linear(hidden_size * 2 + 3, hidden_size)  # 拼接：主车 + 概率 + 对抗车辆
        else:
            self.shared_fc1 = nn.Linear(state_dim, hidden_size)
            self.shared_fc2 = nn.Linear(hidden_size, hidden_size)

        # 策略头和价值头
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, action=None, prob=None):
        """
        :param x: 状态输入 [B, main_dim + veh_count * veh_dim]
        :param action: 当前选择的动作（用于计算 attention 机制）
        :param prob: 网络输出的换道概率 [B, 3]（用于计算 Attn-1）
        """
        if self.use_attention:
            B = x.shape[0]
            main_feat = x[:, :self.main_dim]  # [B, main_dim]
            veh_feat = x[:, self.main_dim:].view(B, self.veh_count, self.veh_dim)  # [B, veh_count, veh_dim]

            # 主车嵌入
            main_embed = self.main_embed(main_feat)  # [B, H]

            # Attn-1: 基于动作与概率之间的偏离进行加权
            if action is not None and prob is not None:
                action_onehot = F.one_hot(action, num_classes=3).float()  # [B, 3]
                inverse = 1.0 - action_onehot  # 取反动作
                diff = (inverse - prob).abs().sum(dim=1, keepdim=True)  # [B, 1] 偏离计算
                gate = torch.sigmoid(2.0 * diff)  # 加权因子
                weighted_prob = gate * prob  # [B, 3]
            else:
                weighted_prob = torch.zeros(B, 3).to(x.device)  # 默认概率为0

            # Attn-2: 基于与标准危险车辆的相似度进行加权
            danger_std = torch.tensor([-5.0, 0.0, -5.0, 0.0, 0, 0, 0], dtype=torch.float32).to(x.device)
            dist = torch.norm(veh_feat - danger_std, dim=-1)  # [B, veh_count]
            weights = F.softmax(-dist, dim=1)  # 距离越近，权重越大
            veh_embeds = self.veh_embed(veh_feat)  # [B, veh_count, H]
            weighted_veh = torch.sum(weights.unsqueeze(-1) * veh_embeds, dim=1)  # [B, H]

            # 拼接：主车嵌入 + 换道概率 + 对抗车辆加权
            final_input = torch.cat([main_embed, weighted_prob, weighted_veh], dim=1)  # [B, 2H + 3]
            x = F.relu(self.fc_final(final_input))  # [B, H]
        else:
            x = F.relu(self.shared_fc1(x))  # [B, H]
            x = F.relu(self.shared_fc2(x))  # [B, H]

        return self.policy_head(x), self.value_head(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.98, eps_clip=0.2,
                 entropy_coef=0.01, use_attention=False, attn_dims=None):
        self.use_attention = use_attention
        self.model = PPOActorCritic(state_dim, action_dim, use_attention=use_attention, attn_dims=attn_dims)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.best_reward_state = None  # 用于存储历史最佳状态
        self.best_reward = -np.inf  # 初始化最好的奖励为负无穷

    def update_best_reward_state(self, episode_reward, state):
        """更新最好的奖励状态"""
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_reward_state = state

    def select_action(self, state, prob=None):
        """根据状态选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        if self.use_attention and prob is not None:
            logits, _ = self.model(state, action=None, prob=prob)
        else:
            logits, _ = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, states, actions, probs=None):
        """评估当前状态和动作的对数概率与价值"""
        if self.use_attention and probs is not None:
            logits, values = self.model(states, action=actions, prob=probs)
        else:
            logits, values = self.model(states)
        probs_dist = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs_dist)
        return dist.log_prob(actions), torch.squeeze(values), dist.entropy()

    def update(self, memory, batch_size=64, epochs=4):
        """更新策略网络"""
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.LongTensor(np.array(memory['actions']))
        old_logprobs = torch.FloatTensor(np.array(memory['logprobs']))
        returns = torch.FloatTensor(np.array(memory['returns']))
        advantages = returns - self.model(states)[1].squeeze().detach()
        probs = torch.FloatTensor(np.array(memory['probs'])) if self.use_attention else None

        for _ in range(epochs):
            for i in range(0, len(states), batch_size):
                idx = slice(i, i + batch_size)
                s_batch, a_batch = states[idx], actions[idx]
                lp_old, adv, ret = old_logprobs[idx], advantages[idx], returns[idx]
                p_batch = probs[idx] if self.use_attention else None

                logprobs, values, entropy = self.evaluate(s_batch, a_batch, p_batch)
                ratios = torch.exp(logprobs - lp_old)
                surr1, surr2 = ratios * adv, torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(values,
                                                                          ret) - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()


def compute_gae(rewards, masks, values, gamma=0.98, lam=0.90):
    """
    计算Generalized Advantage Estimation (GAE)

    :param rewards: 每一步的奖励
    :param masks: 每一步是否结束的标记（done），0表示结束，1表示未结束
    :param values: 每一步的状态价值
    :param gamma: 折扣因子，默认为0.99
    :param lam: GAE的lambda参数，默认为0.95
    :return: 每一步的优势值
    """
    values = values + [0]  # 将values的末尾添加一个0（处理最后一步）
    gae = 0
    returns = []

    # 从最后一步往前计算GAE
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]  # 计算delta
        gae = delta + gamma * lam * masks[step] * gae  # 更新GAE
        returns.insert(0, gae + values[step])  # 插入返回值，保留顺序

    return returns


def apply_her(memory, trajectory, env, alpha=0.3):
    """使用HER机制的奖励调整"""
    if env.best_reward_state is None:
        return memory

    goal_state = env.best_reward_state[:2]  # 目标位置

    new_rewards = []
    for (state, action, old_reward, next_state, done) in trajectory:
        distance = np.linalg.norm(state[:2] - goal_state)
        her_reward = 1.0 if distance < 5.0 else 0.0
        fused_reward = old_reward + alpha * her_reward
        new_rewards.append(fused_reward)

    memory['rewards'] = new_rewards
    return memory


def train_ppo(env, agent, max_episodes=300, max_steps=5000, use_her=False, use_dlc_input=False):
    reward_curve = []
    collision_count = 0
    total_speed_all = 0.0
    speed_count_all = 0.0
    lane_attempt_count = 0
    lane_success_count = 0

    for episode in range(max_episodes):
        decay_lr =  max(2e-4, 5e-4 * (1 - episode / max_episodes))
        agent.optimizer.param_groups[0]['lr'] = decay_lr
        agent.entropy_coef = max(0.005, 0.02 * (1 - episode / max_episodes))  # 更高初始熵权重
        agent.eps_clip = 0.2

        state = env.reset()
        memory = {
            'states': [], 'actions': [], 'logprobs': [],
            'rewards': [], 'masks': [], 'values': [], 'returns': [],
            'probs': []
        }

        episode_reward = 0
        trajectory = []
        attempted_lane_change = False
        lane_id_before = None

        for step in range(max_steps):
            if step % 100 == 0:
                attempted_lane_change = False
                lane_id_before = env.world.get_map().get_waypoint(env.cars['maincar'].get_location()).lane_id

            prob = env._get_lane_change_probs() if use_dlc_input else np.array([0.3, 0.4, 0.3], dtype=np.float32)
            action, logprob, entropy = agent.select_action(state, prob=prob)
            next_state, reward, done, _ = env.step(action)

            with torch.no_grad():
                value = agent.model(torch.FloatTensor(state).unsqueeze(0),
                                    action=torch.tensor([action]),
                                    prob=torch.tensor(prob).unsqueeze(0))[1]

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob.item())
            memory['rewards'].append(reward)
            memory['masks'].append(1 - done)
            memory['values'].append(value.item())
            memory['probs'].append(prob)

            if use_her:
                trajectory.append((state, action, reward, next_state, done))

            if action in [1, 2]:
                attempted_lane_change = True

            if (step + 1) % 100 == 0 and attempted_lane_change:
                lane_id_after = env.world.get_map().get_waypoint(env.cars['maincar'].get_location()).lane_id
                lane_attempt_count += 1
                if lane_id_after != lane_id_before:
                    lane_success_count += 1

            speed = np.linalg.norm([env.cars['maincar'].get_velocity().x,
                                    env.cars['maincar'].get_velocity().y])
            total_speed_all += speed
            speed_count_all += 1

            state = next_state
            episode_reward += reward
            if done:
                break

        if env.collision_event:
            collision_count += 1

        if use_her:
            memory = apply_her(memory, trajectory, env, alpha=0.3)

        memory['returns'] = compute_gae(memory['rewards'], memory['masks'], memory['values'])
        agent.update(memory)
        reward_curve.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    print(f"Total Collisions: {collision_count}")
    print(f"Collision Rate: {collision_count / max_episodes:.2%}")
    if lane_attempt_count > 0:
        print(f"Lane Change Success Rate: {lane_success_count / lane_attempt_count:.2%}")
    else:
        print("No lane change attempts")
    print(f"Average Speed: {total_speed_all / speed_count_all:.2f} m/s")

    # Plot reward curve
    plt.plot(reward_curve)
    plt.title("Reward Convergence Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig("reward_curve.png")
    plt.close()

    # ==== ⏬保存统计结果、奖励CSV、热图（可选） ====

    # 1. 保存 summary.txt
    # 在 f-string 之前先计算换道成功率字符串
    if lane_attempt_count > 0:
        lane_success_rate_str = f"{lane_success_count / lane_attempt_count:.2%}"
    else:
        lane_success_rate_str = "0.00%"

    summary = f"""🚗 训练总结（PPO模型）
    -------------------------
    总回合数         : {max_episodes}
    总碰撞次数       : {collision_count}
    碰撞率           : {collision_count / max_episodes:.2%}

    换道尝试次数     : {lane_attempt_count}
    换道成功次数     : {lane_success_count}
    换道成功率       : {lane_success_rate_str}

    平均速度         : {total_speed_all / speed_count_all:.2f} m/s
    平均每回合奖励   : {np.mean(reward_curve):.2f}"""

    with open("training_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary.strip())

    # 2. 保存每回合奖励为 CSV
    reward_df = pd.DataFrame({
        "episode": list(range(1, len(reward_curve) + 1)),
        "reward": reward_curve
    })
    reward_df.to_csv("episode_rewards.csv", index=False)

    # 3. 若启用注意力机制，则保存注意力热图（模拟值）
    if agent.use_attention:
        attn_matrix = np.random.rand(env.attn_dims["veh_count"], env.attn_dims["veh_dim"])
        plt.figure(figsize=(10, 4))
        sns.heatmap(attn_matrix, cmap="YlOrBr", annot=True)
        plt.title("Attention Heatmap (示意)")
        plt.xlabel("Feature Index")
        plt.ylabel("Opponent Vehicle")
        plt.tight_layout()
        plt.savefig("attention_heatmap.png")
        plt.close()

    return agent
