# CommRL

# CommRL: A Deep Reinforcement Learning Library for Wireless Communication Systems 📡🤖

**CommRL** is a modular, extensible Deep Reinforcement Learning (DRL) library tailored for wireless communication scenarios. It provides a clean interface to implement, train, and evaluate intelligent agents in complex radio environments such as:

- 📶 Dynamic Spectrum Access (DSA)
- 🚁 UAV Communication Scheduling
- 📡 Beamforming & MIMO resource allocation
- 🧠 Mobile Edge Computing (MEC) task offloading

## ✨ Features

- Modular agent design (DQN, PPO, SAC, etc.)
- Built-in support for communication-related environments
- Easy configuration with YAML
- Logging and visualization via TensorBoard
- Support for discrete and continuous action spaces


CommRL/
├── agents/                 # 各类RL算法，如DQN, PPO等
│   ├── base_agent.py
│   ├── dqn.py
│   ├── ppo.py
│   └── sac.py
├── networks/               # 神经网络模块
│   ├── mlp.py
│   ├── cnn.py
│   └── custom_comm_net.py
├── buffers/                # Replay Buffer & Prioritized Buffer
│   └── replay_buffer.py
├── envs/                   # 通信场景封装，如MAC、UAV、MEC等
│   ├── gym_wrapper.py
│   ├── uav_channel_env.py
│   └── dynamic_spectrum_env.py
├── trainers/               # 通用训练器，用于调度训练/测试
│   └── trainer.py
├── configs/                # YAML 格式的超参数配置
│   └── dqn_uav.yaml
├── utils/                  # 工具函数，如logger, seeding, tensorboard接口
│   ├── logger.py
│   └── schedulers.py
├── tests/                  # 单元测试
│   └── test_buffer.py
├── scripts/                # 启动脚本或实验脚本
│   ├── train_dqn_uav.py
│   └── evaluate_policy.py
├── notebooks/              # 可选：Jupyter实验记录
│   └── analysis.ipynb
├── requirements.txt        # 项目依赖包
├── README.md               # 项目说明文档
└── setup.py                # pip 安装脚本（可选）
