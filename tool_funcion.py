"""
该文件存放了一些用于计算的函数
"""
import torch
import math
import pandas as pd
import numpy as np


class CAR48step:
    def __init__(self, x: list, y: list, v_x: list, v_y: list, a_x: list, a_y: list, dhw: list, thw: list):
        self.x = pd.Series(x)
        self.y = pd.Series(y)
        self.v_x = pd.Series(v_x)
        self.v_y = pd.Series(v_y)
        self.a_x = pd.Series(a_x)
        self.a_y = pd.Series(a_y)
        self.dhw = pd.Series(dhw)
        self.thw = pd.Series(thw)
        self.delta_x = self.x - x[0]
        self.delta_y = self.y - y[0]

    def dop_maker(self):
        """
        :return:  8*7的tensor类型dop矩阵
        """
        dop_matrix = DOP_maker(self.delta_y, self.delta_x, self.v_y, self.v_x, self.a_y, self.a_x, self.dhw, self.thw)
        return dop_matrix


class CarEnvInfo:
    def __init__(self, ego_car: CAR48step,
                 p_car: CAR48step,
                 pl_car: CAR48step,
                 pr_car: CAR48step,
                 fl_car: CAR48step,
                 fr_car: CAR48step,
                 als_l_car: CAR48step,
                 als_r_car: CAR48step):
        self.ego_car = ego_car
        self.p_car = p_car
        self.pl_car = pl_car
        self.pr_car = pr_car
        self.fl_car = fl_car
        self.fr_car = fr_car
        self.als_l_car = als_l_car
        self.als_r_car = als_r_car

    def dlc_maker(self):
        dlc = DLC_maker(self.ego_car, self.p_car, self.pl_car, self.pr_car, self.fl_car, self.fr_car)
        return dlc

    def ego_car_dop_maker(self):
        e_dop = self.ego_car.dop_maker()
        e_dop = e_dop.unsqueeze(0)
        return e_dop

    def surround_car_dop_maker(self):
        p_dop = self.p_car.dop_maker()
        pl_dop = self.pl_car.dop_maker()
        pr_dop = self.pr_car.dop_maker()
        fl_dop = self.fl_car.dop_maker()
        fr_dop = self.fr_car.dop_maker()
        als_l_dop = self.als_l_car.dop_maker()
        als_r_dop = self.als_r_car.dop_maker()
        dop_surround = torch.stack([
            p_dop,
            pl_dop,
            pr_dop,
            fl_dop,
            fr_dop,
            als_l_dop,
            als_r_dop,
        ], dim=0)
        return dop_surround


def status_list_maker(data: pd.Series):
    """
    name : status_list_maker(统计数据列生成器)
    function : 根据输入的数据生成对应的统计数据列，用于生成DOP
    """
    status_list = [
        data.mean(),                # 均值
        data.std(),                 # 标准差
        data.median(),              # 中位数
        data.min(),                 # 最小值
        data.max(),                 # 最大值
        data.quantile(0.25),        # 25%分位数
        data.quantile(0.75)         # 75%分位数
    ]
    return torch.tensor(status_list, dtype=torch.float32)


def DOP_maker(
        y_data: pd.Series,
        x_data: pd.Series,
        v_y: pd.Series,
        v_x: pd.Series,
        a_y: pd.Series,
        a_x: pd.Series,
        dhw: pd.Series,
        thw: pd.Series
):
    """
    dop是一个8*7的矩阵，八行分别表示
    :param y_data:
    :param x_data:
    :param v_y:
    :param v_x:
    :param a_y:
    :param a_x:
    :param dhw:
    :param thw:
    :return: 8*7tensor
    """
    dop = [
        status_list_maker(y_data),
        status_list_maker(x_data),
        status_list_maker(v_y),
        status_list_maker(v_x),
        status_list_maker(a_y),
        status_list_maker(a_x),
        status_list_maker(dhw),
        status_list_maker(thw),
    ]
    # dop = [[dop[j][i] for j in range(len(dop))] for i in range(len(dop[0]))]  # 矩阵转置
    dop = torch.stack(dop)

    return dop


def DLC_maker(ego_car: CAR48step,
              p_car: CAR48step,
              pl_car: CAR48step,
              pr_car: CAR48step,
              fl_car: CAR48step,
              fr_car: CAR48step):
    dlc_matrix = torch.tensor([
        ego_car.v_x.iloc[-1] - p_car.v_x.iloc[-1],
        pl_car.v_x.iloc[-1] - p_car.v_x.iloc[-1],
        pr_car.v_x.iloc[-1] - p_car.v_x.iloc[-1],
        pl_car.x.iloc[-1] - p_car.x.iloc[-1],
        pr_car.x.iloc[-1] - p_car.x.iloc[-1],
        fl_car.x.iloc[-1] - ego_car.x.iloc[-1],
        fr_car.x.iloc[-1] - ego_car.x.iloc[-1],
        ego_car.v_x.iloc[-1] - fl_car.v_x.iloc[-1],
        ego_car.v_x.iloc[-1] - fr_car.v_x.iloc[-1],
        (p_car.x.iloc[-1] - ego_car.x.iloc[-1]) - ego_car.v_x.iloc[-1] * 2
    ])
    return dlc_matrix


def surround_dop_maker(car_information: dict):
    return 0
