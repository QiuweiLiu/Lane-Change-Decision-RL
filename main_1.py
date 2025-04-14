from env_setup import ENV
from ppo_model import PPOAgent, train_ppo
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    # ======= 用户交互：是否启用 HER 和 Attention ==========
    use_her = input("是否启用 HER 机制？(y/n): ").lower() == 'y'
    use_attention = input("是否启用 注意力机制？(y/n): ").lower() == 'y'
    use_dlc_input = input("是否使用神经网络输出作为状态输入？(y/n): ").lower() == 'y'

    config = {'host': 'localhost', 'port': 2000}
    logger = logging.getLogger("TRAIN")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    env = ENV(config, logger, use_dlc_input=use_dlc_input)

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        use_attention=use_attention,
        attn_dims=env.attn_dims
    )

    train_ppo(env, agent, max_episodes=300, max_steps=2000, use_her=use_her, use_dlc_input=use_dlc_input)
    # 保存归一化参数
    env.normalizer.save("obs_normalizer.npz")

    agent.save("ppo_model.pth")
    logger.info("✅ 模型已保存为 ppo_model.pth")
