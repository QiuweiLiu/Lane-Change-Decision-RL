import logging
import numpy as np
from env_setup import ENV


def test_dlc_policy(env, episodes=5, max_steps=500):
    all_rewards = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            # 获取神经网络输出的换道概率
            lane_probs = env._get_lane_change_probs()  # [keep, left, right]

            # 决策逻辑：若左或右概率 > 0.5，则换道
            if lane_probs[1] > 0.5:
                action = 1  # 左换道
            elif lane_probs[2] > 0.5:
                action = 2  # 右换道
            else:
                action = 0  # 保持

            _, reward, done, _ = env.step(action)
            ep_reward += reward

            if done:
                break

        print(f"✅ Episode {ep + 1}: 总奖励 = {ep_reward:.2f}")
        all_rewards.append(ep_reward)

    avg_reward = np.mean(all_rewards)
    print(f"🎯 平均奖励 = {avg_reward:.2f}")
    env.close()


if __name__ == "__main__":
    config = {'host': 'localhost', 'port': 2000}
    logger = logging.getLogger("DLC_TEST")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 初始化环境，启用 dlc_input 以激活神经网络
    env = ENV(config, logger, use_dlc_input=True)

    test_dlc_policy(env, episodes=5, max_steps=500)
