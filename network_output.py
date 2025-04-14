import tool_func as tf
import torch
import torch.nn as nn
import network_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dlc_model_output(car_env_info: tf.CarEnvInfo, model_parameter_path):
    """
    该函数接受主车的dop矩阵输入，周围车的dop矩阵输入和dlc_vector因素矩阵
    :param car_env_info:
    :param model_parameter_path: 存放模型参数的.pth文件的绝对路径
    :return: 返回一个三分类结果，[不换道概率，左换道概率，右换道概率]
    """

    checkpoint = torch.load(model_parameter_path, weights_only=True)
    """
    :ego_car_dop: torch.tensor，形状为[input_channel, dop_row, dop_col]
    :surround_car_dop: torch.tensor，形状为[input_channel, dop_row, dop_col]
    :dlc_vector: torch.tensor，形状为[dlc(一般为10维)]
    """
    ego_car_dop = car_env_info.ego_car_dop_maker()
    surround_car_dop = car_env_info.surround_car_dop_maker()
    dlc_vector = car_env_info.dlc_maker()
    # 扩展维度使得与网络输入匹配
    ego_car_dop = ego_car_dop.unsqueeze(0)
    surround_car_dop = surround_car_dop.unsqueeze(0)
    dlc_vector = dlc_vector.unsqueeze(0)
    with torch.no_grad():
        # 载入数据
        ego_car_dop = ego_car_dop.to(device)
        surround_car_dop = surround_car_dop.to(device)
        dlc_vector = dlc_vector.to(device)
        # 初始化函数并进入测试模式
        cnn_sur_model = network_output.CNN1().to(device).eval()
        cnn_ego_model = network_output.CNN2().to(device).eval()
        fc_layer_model = network_output.FcLayer().to(device).eval()
        # 加载参数
        cnn_sur_model.load_state_dict(checkpoint['sur_cnn_model_state_dict'])
        cnn_ego_model.load_state_dict(checkpoint['ego_cnn_model_state_dict'])
        fc_layer_model.load_state_dict(checkpoint['fc_layer_model_state_dict'])
        # cnn层
        output_sur = cnn_sur_model(surround_car_dop)
        output_ego = cnn_ego_model(ego_car_dop)
        # fc层
        fc_input = torch.cat([output_sur.float(), output_ego.float(), dlc_vector.float()], dim=1)
        outputs = fc_layer_model(fc_input)
        result = nn.functional.softmax(outputs, dim=1)
        return result
