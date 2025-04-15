import tool_funcion as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ✅ CNN1: 周围车辆 DOP（输入：7×8×7）
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(7, 16, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.flatten(x)


# ✅ CNN2: 主车 DOP（输入：1×8×7）
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.flatten(x)


# ✅ 全连接层：输入为 CNN1 + CNN2 + DLC（490 维）
class FcLayer(nn.Module):
    def __init__(self):
        super(FcLayer, self).__init__()
        self.fc1 = nn.Linear(490, 50)
        self.fc2 = nn.Linear(50, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


# ✅ 主调用接口：返回三类 softmax 概率
def dlc_model_output(car_env_info: tf.CarEnvInfo, model_parameter_path):

    checkpoint = torch.load(model_parameter_path, map_location=torch.device('cpu'))


    # 准备输入
    ego_car_dop = car_env_info.ego_car_dop_maker().unsqueeze(0)            # [1, 1, 8, 7]
    surround_car_dop = car_env_info.surround_car_dop_maker().unsqueeze(0)  # [1, 7, 8, 7]
    dlc_vector = car_env_info.dlc_maker().unsqueeze(0)                      # [1, 10]



    with torch.no_grad():
        ego_car_dop = ego_car_dop.to(device)
        surround_car_dop = surround_car_dop.to(device)
        dlc_vector = dlc_vector.to(device)

        # 初始化模型并加载权重
        cnn_sur_model = CNN1().to(device).eval()
        cnn_ego_model = CNN2().to(device).eval()
        fc_layer_model = FcLayer().to(device).eval()

        cnn_sur_model.load_state_dict(checkpoint['sur_cnn_model_state_dict'])
        cnn_ego_model.load_state_dict(checkpoint['ego_cnn_model_state_dict'])
        fc_layer_model.load_state_dict(checkpoint['fc_layer_model_state_dict'])



        # 推理阶段
        out_sur = cnn_sur_model(surround_car_dop)  # e.g., [1, 464]
        out_ego = cnn_ego_model(ego_car_dop)       # e.g., [1, 16]
        fc_input = torch.cat([out_sur, out_ego, dlc_vector.float()], dim=1)  # [1, 490]



        logits = fc_layer_model(fc_input)  # [1, 3]


        probs = nn.functional.softmax(logits, dim=1)

        return probs

