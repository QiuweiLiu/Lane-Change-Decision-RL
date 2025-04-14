# lane_prob_module.py

import torch
import numpy as np
from tool_funcion import CAR48step, CarEnvInfo
from network_output import dlc_model_output
import carla


# 存储每辆车历史轨迹缓存
class TrajectoryBuffer:
    def __init__(self, max_len=48):  # 每辆车最多保留48步数据
        self.max_len = max_len
        self.history = {}

    def update(self, actor: carla.Actor):
        actor_id = actor.id
        tf = actor.get_transform()
        vel = actor.get_velocity()
        acc = actor.get_acceleration()
        loc = tf.location
        yaw = tf.rotation.yaw * np.pi / 180  # 航向角度（转换为弧度）

        # 存储车辆绝对坐标信息：x, y, vx, vy, ax, ay, dhw, thw
        entry = {
            "x": loc.x, "y": loc.y,  # 车辆绝对位置
            "vx": vel.x, "vy": vel.y,  # 车辆速度
            "ax": acc.x, "ay": acc.y,  # 车辆加速度
            "dhw": 0.0, "thw": 0.0,  # 距离前车和时间头距，可以根据需要进行调整
        }

        # 初始化或更新车辆历史轨迹（最多48步）
        if actor_id not in self.history:
            self.history[actor_id] = {
                "x0": loc.x, "y0": loc.y,  # 初始位置
                "data": [entry]
            }
        else:
            self.history[actor_id]["data"].append(entry)
            if len(self.history[actor_id]["data"]) > self.max_len:
                self.history[actor_id]["data"].pop(0)

    def to_car48(self, actor_id):
        # 获取指定车辆的48步历史数据并转换为CAR48step格式
        if actor_id not in self.history or len(self.history[actor_id]["data"]) < 2:
            return CAR48step([0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2)

        traj = self.history[actor_id]["data"]
        return CAR48step(
            [d["x"] for d in traj],
            [d["y"] for d in traj],
            [d["vx"] for d in traj],
            [d["vy"] for d in traj],
            [d["ax"] for d in traj],
            [d["ay"] for d in traj],
            [d["dhw"] for d in traj],
            [d["thw"] for d in traj],
        )

    def clear(self):
        # 清空历史数据
        self.history = {}


def get_lane_prob(maincar, surrounding_cars, buffer: TrajectoryBuffer, model_path: 'checkpoint.pth'):
    buffer.update(maincar)  # 更新主车数据

    # 找出论文中要求的 7 辆对抗车辆
    sorted_cars = {
        "p_car": None, "pl_car": None, "pr_car": None,
        "fl_car": None, "fr_car": None, "als_l_car": None, "als_r_car": None
    }

    # 获取主车位置与朝向
    main_loc = maincar.get_location()
    main_yaw = maincar.get_transform().rotation.yaw * np.pi / 180  # 主车的航向角度（弧度）

    # 遍历所有周围车辆，并根据相对位置确定角色
    for car in surrounding_cars:
        car_loc = car.get_location()
        car_yaw = car.get_transform().rotation.yaw * np.pi / 180  # 周围车的航向角度（弧度）

        # 计算主车与周围车辆的相对位置
        dx = car_loc.x - main_loc.x
        dy = car_loc.y - main_loc.y

        # 将相对位置转换到主车坐标系（将主车朝向作为x轴）
        rel_x = dx * np.cos(-main_yaw) - dy * np.sin(-main_yaw)
        rel_y = dx * np.sin(-main_yaw) + dy * np.cos(-main_yaw)

        # 根据位置分类
        if rel_y > 0:  # 左侧车辆
            if rel_x > 0:
                sorted_cars["pl_car"] = car  # 左前车
            elif rel_x < 0:
                sorted_cars["fl_car"] = car  # 左后车
            else:
                sorted_cars["als_l_car"] = car  # 左侧相邻车
        elif rel_y < 0:  # 右侧车辆
            if rel_x > 0:
                sorted_cars["pr_car"] = car  # 右前车
            elif rel_x < 0:
                sorted_cars["fr_car"] = car  # 右后车
            else:
                sorted_cars["als_r_car"] = car  # 右侧相邻车
        else:  # 主车前后车辆
            if rel_x > 0:
                sorted_cars["p_car"] = car  # 前车

    # 将每辆车的数据按照顺序传入 CarEnvInfo
    car_info = CarEnvInfo(
        buffer.to_car48(maincar.id),  # 主车数据
        buffer.to_car48(sorted_cars["p_car"].id if sorted_cars["p_car"] else None),
        buffer.to_car48(sorted_cars["pl_car"].id if sorted_cars["pl_car"] else None),
        buffer.to_car48(sorted_cars["pr_car"].id if sorted_cars["pr_car"] else None),
        buffer.to_car48(sorted_cars["fl_car"].id if sorted_cars["fl_car"] else None),
        buffer.to_car48(sorted_cars["fr_car"].id if sorted_cars["fr_car"] else None),
        buffer.to_car48(sorted_cars["als_l_car"].id if sorted_cars["als_l_car"] else None),
        buffer.to_car48(sorted_cars["als_r_car"].id if sorted_cars["als_r_car"] else None)
    )

    prob = dlc_model_output(car_info, model_path).squeeze().cpu().numpy()  # 调用神经网络进行决策
    return prob
