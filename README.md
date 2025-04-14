Lane-Change-Decision-RL

项目简介：
本项目利用强化学习（Reinforcement Learning, RL）方法，在CARLA仿真环境中实现自动驾驶车辆的换道决策。通过多智能体系统和自主训练机制，本项目旨在优化换道时机的决策，以提高自动驾驶系统的效率与安全性。

项目目标：

  强化学习训练：使用**PPO（Proximal Policy Optimization）**算法训练强化学习智能体，使其能够在动态交通环境中做出合适的换道决策。

  CARLA仿真环境：基于CARLA仿真平台，构建适用于自动驾驶研究的高保真虚拟环境，模拟复杂的道路场景。

  换道决策：训练智能体识别换道时机并决策是否换道，同时考虑周围车辆、交通规则等因素。

  多层次决策机制：结合注意力机制（Attention Mechanism）和模仿学习，提高决策的准确性和稳定性。

主要特点：

  CARLA仿真环境：通过CARLA提供的高质量仿真数据进行模型训练和测试，确保结果具有高可用性和实际应用价值。

  强化学习算法：使用PPO算法进行策略优化，提升换道决策的智能性和执行效率。

  环境设置与奖励机制：设计了基于车辆状态、换道时机和交通规则的定制化奖励函数，确保智能体能够学习到正确的换道时机。

  数据可视化：通过注意力热图和奖励曲线等数据可视化手段，帮助评估模型的训练效果与决策过程。

项目结构：

  env_setup.py：环境设置和车辆管理脚本，负责CARLA仿真环境的初始化与配置。

  ppo_model.py：PPO强化学习模型的实现，包括策略网络和价值网络。

  test_ppo.py：强化学习模型的测试与评估脚本，验证智能体在训练过程中的表现。

  lane_prob_module.py：处理车辆轨迹数据并计算换道概率的模块，支持基于历史数据的决策分析。

  tool_funcion.py：提供用于计算DOP（Dilution of Precision）矩阵、统计数据等的工具函数。

  network_output.py：实现神经网络的前向计算，生成换道决策概率。

如何运行：

  确保已安装并配置好CARLA仿真环境。

  安装项目依赖：

    pip install -r requirements.txt

配置环境并运行训练脚本：

    python main_1.py

  选择是否启用HER机制（Hindsight Experience Replay）和注意力机制，并开始强化学习训练。

未来发展方向：

  引入更多的环境变量和复杂场景，提升训练模型的鲁棒性。

  结合实际驾驶数据，进一步改进模型的实际适用性。

  实现更复杂的多智能体决策，模拟交通流中的复杂交互行为。

贡献：
欢迎任何对自动驾驶、强化学习、或者CARLA仿真感兴趣的开发者贡献代码、提出问题或进行改进。如果你有任何建议或想法，请提交Pull Request或Issues。








Lane-Change-Decision-RL

Project Overview:
This project utilizes Reinforcement Learning (RL) methods to implement lane change decision-making for autonomous vehicles in the CARLA simulation environment. By leveraging multi-agent systems and autonomous training mechanisms, the project aims to optimize lane change timing decisions, improving both efficiency and safety in autonomous driving systems.

Project Goals:

  Reinforcement Learning Training: Train reinforcement learning agents using the Proximal Policy Optimization (PPO) algorithm to make appropriate lane change decisions in dynamic traffic environments.

  CARLA Simulation Environment: Set up a high-fidelity virtual environment using the CARLA simulator for autonomous driving research and testing.

  Lane Change Decision-Making: Train agents to identify the right moments to change lanes while considering surrounding vehicles, traffic rules, and other environmental factors.

  Multi-Layered Decision Mechanism: Integrate Attention Mechanism and Imitation Learning to improve decision accuracy and stability.

Key Features:

   CARLA Simulation: Utilizes the CARLA simulator to train and test models with high-fidelity simulation data, ensuring the results are both practical and applicable.

   Reinforcement Learning Algorithms: Implements the PPO algorithm to optimize policies and enhance the efficiency of lane change decision-making.

   Customized Reward Mechanisms: Designs a custom reward function based on vehicle states, lane change timing, and traffic rules to guide the agent's learning process.

   Data Visualization: Uses tools like attention heatmaps and reward curves to visualize and evaluate the agent's training effectiveness and decision-making.

Project Structure:

  env_setup.py: Environment setup and vehicle management script responsible for initializing and configuring the CARLA simulation environment.
  
  ppo_model.py: Implementation of the PPO reinforcement learning model, including both the policy and value networks.

  test_ppo.py: Testing and evaluation script for the reinforcement learning model to validate agent performance during training.

  lane_prob_module.py: Handles vehicle trajectory data and calculates lane change probabilities, supporting decision analysis based on historical data.

  tool_funcion.py: Provides utility functions for computing DOP (Dilution of Precision) matrices, statistical data, etc.

  network_output.py: Implements the forward pass of the neural network to generate lane change decision probabilities.

How to Run:

  Make sure you have the CARLA simulation environment installed and properly configured.

  Install project dependencies:

    pip install -r requirements.txt

  Set up the environment and run the training script:

    python main_1.py

  Choose whether to enable the HER mechanism (Hindsight Experience Replay) and Attention Mechanism, and start the reinforcement learning training.

Future Directions:

  Introduce more environment variables and complex scenarios to improve model robustness.

  Incorporate real-world driving data to enhance the model’s applicability.

  Implement more complex multi-agent decision-making to simulate intricate interactions in traffic flow.

Contributions:
Welcome contributions from developers interested in autonomous driving, reinforcement learning, or the CARLA simulation platform. Feel free to submit Pull Requests or Issues if you have suggestions or improvements.
