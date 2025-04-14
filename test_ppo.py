
import torch
from ppo_model import PPOAgent
import numpy as np
from env_setup import ENV
import logging

def test_ppo(env, agent, episodes=10, max_steps=500, model_path="ppo_model.pth"):
    agent.load(model_path)
    agent.model.eval()
    all_rewards = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            action, _, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break

        print(f"✅ 测试 Episode {ep + 1}: 奖励 = {ep_reward:.2f}")
        all_rewards.append(ep_reward)

    avg = np.mean(all_rewards)
    print(f"🎯 平均奖励：{avg:.2f}")
    env.close()

# ✅ 主程序入口
if __name__ == "__main__":
    config = {'host': 'localhost', 'port': 2000}
    logger = logging.getLogger("TEST")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    env = ENV(config, logger)
    state_dim = env.observation_space.shape[0]
    # 加载归一化参数
    env.normalizer.load("obs_normalizer.npz")
    action_dim = env.action_space.n

    use_attention = input("是否在测试中启用注意力机制？(y/n): ").lower() == 'y'
    agent = PPOAgent(state_dim, action_dim, use_attention=use_attention, attn_dims=env.attn_dims)

    test_ppo(env, agent, episodes=5, max_steps=500)
