# 从 lane_prob_module 导入 get_lane_prob 和 TrajectoryBuffer
from lane_prob_module import get_lane_prob, TrajectoryBuffer
from env_setup import ENV
import logging
import numpy as np






def test_dlc_policy(episodes=5, max_steps=500, model_path="ppo_model.pth"):
    all_rewards = []
    total_lane_attempts = 0
    total_lane_successes = 0

    # 初始化环境
    config = {'host': 'localhost', 'port': 2000}
    logger = logging.getLogger("TEST")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 创建环境对象
    env = ENV(config, logger, use_dlc_input=True)  # 环境初始化


    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        lane_changed = False
        lane_id_before = None

        # 获取主车和周围车辆的对象
        maincar = env.cars['maincar']
        surrounding_cars = [actor for actor in env.actor_list if actor != maincar]  # 获取周围的车辆

        # 创建 TrajectoryBuffer 实例，用于存储轨迹数据
        buffer = TrajectoryBuffer(max_len=48)

        for step in range(max_steps):
            # 获取神经网络输出的换道概率
            lane_probs = get_lane_prob(maincar, surrounding_cars, buffer, model_path)  # 使用 lane_prob_model 中的 get_lane_prob

            # ==================== 检查 1: 输入维度检验 ====================
            if state.shape[0] != env.observation_space.shape[0]:

                break  # 或者跳过当前步，具体取决于你想如何处理这个错误

            # ==================== 每100步打印换道概率 ====================
            if step % 100 == 0:
                print(f"🔍 Step {step + 1} 换道概率: {lane_probs}")

            # 决策逻辑：若左或右概率 > 0.5，则换道
            if lane_probs[1] > 0.5:
                action = 1  # 左换道
            elif lane_probs[2] > 0.5:
                action = 2  # 右换道
            else:
                action = 0  # 保持

            # 记录换道尝试
            if step % 100 == 0:
                lane_changed = False
                lane_id_before = env.world.get_map().get_waypoint(env.cars['maincar'].get_location()).lane_id

            _, reward, done, _ = env.step(action)
            ep_reward += reward

            # 换道尝试统计
            if (step + 1) % 100 == 0 and action in [1, 2]:
                lane_id_after = env.world.get_map().get_waypoint(env.cars['maincar'].get_location()).lane_id
                total_lane_attempts += 1
                if lane_id_after != lane_id_before:
                    total_lane_successes += 1

            if done:
                break

        print(f"✅ Episode {ep + 1}: 总奖励 = {ep_reward:.2f}")
        all_rewards.append(ep_reward)

    avg_reward = np.mean(all_rewards)
    print(f"\n🎯 平均奖励 = {avg_reward:.2f}")

    if total_lane_attempts > 0:
        success_rate = total_lane_successes / total_lane_attempts
        print(f"🚗 换道成功率: {success_rate:.2%} ({total_lane_successes}/{total_lane_attempts})")
    else:
        print("🚗 无换道尝试")

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

    # 显式传入换道概率模型路径
    test_dlc_policy(episodes=5, max_steps=2000, model_path="checkpoint.pth")



