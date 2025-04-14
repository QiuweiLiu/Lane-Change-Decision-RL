
class RunningNormalizer:
    def __init__(self, shape, clip_range=5.0):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4  # 防止除零
        self.clip_range = clip_range

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        normed = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -self.clip_range, self.clip_range)

    def save(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.var = data['var']
        self.count = data['count']


import carla
import random
import time
import numpy as np
import gym
import logging
import cv2
import time
from scipy.spatial import KDTree
from threading import Lock

CENTER_LANE = 1
VEHICLE_VEL = 45


# 自定义日志格式
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    red = "\x1b[31;20m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: grey + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)




class ENV:
    def _get_lane_change_probs(self):
        # 🚧 占位神经网络输出，未来替换为模型预测
        return np.array([0.3, 0.4, 0.3], dtype=np.float32)
    def __init__(self, config, logger,use_dlc_input=False):
        self.use_dlc_input = use_dlc_input
        try:
            self.client = carla.Client(config['host'], config['port'])
            self.client.get_server_version()  # 验证连接
        except RuntimeError as e:
            logger.error(f"无法连接Carla服务器: {e}")
            raise

        self.client.set_timeout(20.0)
        self.lane_changed_once = False
        # 加载地图
        self.world = self.client.load_world("Town05")

        # 保存环境设置
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        self.logger = logger
        # 设置自定义日志格式
        ch = logging.StreamHandler()
        ch.setFormatter(CustomFormatter())
        self.logger.addHandler(ch)

        self.actor_list = []
        self.collision_sensor = None
        self.sensorlist = []
        self.collision_event = False
        self.cars = {}
        self.opponent_speed_settings = {}
        self.time_sleep = 10000
        self.step_count = 0
        self.camera = None
        self.camera_image = None
        self.camera_destroyed = False
        self.actor_destroyed = {}
        self.sensor_destroyed = {}
        self.camera_lock = Lock()

        # 获取蓝图库
        self.blp_lib = self.world.get_blueprint_library()
        # 选择主车模型
        self.model_3 = self.blp_lib.filter("model3")[0]

        # 初始化 best_reward_state
        self.best_reward_state = None  # 记录最佳奖励状态
        self.best_reward = -np.inf  # 初始化最好的奖励为负无穷

        # 设置主车的初始位置和姿态

        # 初始化观测空间

        # ✅ 集中式状态结构定义
        self.state_info = {
            "main_keys": [
                "x", "y", "z", "vx", "vy", "vz",
                "ax", "ay", "az", "yaw",
                "lane_id", "left_lane_id", "right_lane_id", "prob_keep", "prob_left", "prob_right"
            ],
            "veh_keys": ["dx", "dy", "dvx", "dvy", "dax", "day", "dyaw"],
            "veh_count": 5
        }
        self.attn_dims = {
            "main_dim": len(self.state_info["main_keys"]),
            "veh_dim": len(self.state_info["veh_keys"]),
            "veh_count": self.state_info["veh_count"]
        }
        self.obs_dim = self.attn_dims["main_dim"] + self.attn_dims["veh_dim"] * self.attn_dims["veh_count"]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(3)  # 0=保持, 1=左变道, 2=右变道

        # 生成主车
        # 生成主车和交通车
        self._setup_vehicles()

        self.lane_index = CENTER_LANE
        self.vel_ref = VEHICLE_VEL

        self.stuck_counter = 0
        # 平均速度奖励相关变量
        self.total_speed = 0.0
        self.speed_count = 0
        self.normalizer = RunningNormalizer(self.obs_dim)

        self.speed_history = []
        self.lane_change_indices = []  # 记录换道成功步数
        self.lane_speed_rewards_given = set()  # 防止重复给奖励
        self.speed_history = np.zeros(5000, dtype=np.float32)  # 用于记录每步速度
        self.speed_history_ptr = 0
        self.speed_compare_window = 30  # 可以根据需要调整比较窗口大小


    def _get_closest_spawn_point(self, ref_transform):
        spawn_points = self.world.get_map().get_spawn_points()
        ref_loc = ref_transform.location

        def distance(loc):
            return np.sqrt((loc.x - ref_loc.x) ** 2 + (loc.y - ref_loc.y) ** 2)

        sorted_points = sorted(spawn_points, key=lambda sp: distance(sp.location))
        closest = sorted_points[0]
        self.logger.info(f"使用最近的合法 spawn 点生成主车，位置: {closest.location}, yaw: {closest.rotation.yaw}")
        return closest

    def update_best_reward_state(self, episode_reward, state):
        """更新最好的奖励状态"""
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_reward_state = state

    def reset(self):
        # ✅ 重置卡计数器
        self.stuck_counter = 0

        # ✅ 销毁已有车辆
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    self.logger.warning(f"Failed to destroy actor: {e}")
        self.actor_list.clear()

        # ✅ 强制清除 Carla 世界中所有车辆（包括残留）
        all_vehicles = self.world.get_actors().filter("vehicle.*")
        for vehicle in all_vehicles:
            if vehicle.is_alive:
                try:
                    vehicle.destroy()
                except Exception as e:
                    self.logger.warning(f"❌ 无法销毁车辆 {vehicle.id}: {e}")

        # ✅ 销毁摄像头
        if hasattr(self, 'camera') and not self.camera_destroyed:
            try:
                self.camera.stop()
                self.camera.destroy()
            except Exception as e:
                self.logger.warning(f"Failed to destroy camera: {e}")
            finally:
                self.camera_destroyed = True

        # ✅ 销毁所有传感器
        for sensor in self.sensorlist:
            if sensor.is_alive:
                try:
                    sensor.destroy()
                except Exception as e:
                    self.logger.warning(f"Failed to destroy sensor: {e}")
        self.sensorlist.clear()

        # ✅ 销毁碰撞传感器
        if hasattr(self, 'collision_sensor') and self.collision_sensor.is_alive:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except Exception as e:
                self.logger.warning(f"Failed to destroy collision sensor: {e}")

        # ✅ 销毁主车
        if 'maincar' in self.cars:
            maincar = self.cars['maincar']
            if maincar.is_alive:
                try:
                    maincar.destroy()
                except Exception as e:
                    self.logger.warning(f"Failed to destroy maincar: {e}")
        self.cars.clear()

        # ✅ 状态变量重置
        self.step_count = 0
        self.collision_event = False
        self.lane_changed_once = False
        self.initial_lane_id = None
        self.past_pos = None
        self.camera_image = None

        # ✅ 重新生成主车和交通车辆
        self._setup_vehicles()

        # ✅ 重置速度统计与归一化器
        self.total_speed = 0.0
        self.speed_count = 0
        self.normalizer = RunningNormalizer(self.obs_dim)

        # ✅ 过滤掉已死亡车辆（冗余，但安全）
        self.actor_list = [actor for actor in self.actor_list if actor.is_alive]

        self.lane_change_indices = []
        self.lane_speed_rewards_given = set()
        self.previous_lane_id = None
        self.lane_stable_steps = 0

        self.speed_history.fill(0.0)
        self.speed_history_ptr = 0

        self.lane_change_step = -1  # 初始化换道步数记录
        self.opponent_speed_settings.clear()

        # ✅ 返回初始观测
        return self._GetObs()

    def step(self, action):
        self.step_count += 1

        maincar = self.cars['maincar']
        traffic_manager = self.client.get_trafficmanager()

        # === 控制主车行为 ===
        if self.step_count <= 250:
            # 前10步静止等待
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
            maincar.apply_control(control)

            #if self.step_count == 1:
                #self.logger.info("🛑 主车正在原地等待 10 步...")
        elif self.step_count == 251:
            # 第11步开始：移交给 autopilot
            maincar.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.vehicle_percentage_speed_difference(maincar, -10)
            traffic_manager.auto_lane_change(maincar, False)
            #self.logger.info("🚗 主车已移交给 Carla 自动驾驶")

            # 为避免此步空转，先手动 tick 一下
            self.world.tick()
        else:
            # 已交给 Carla 控制，不再 apply_control()
            pass

        # === 处理换道动作（即使是 Carla 控制，也可以强制换道） ===
        if self.step_count > 250:  # Carla 自动驾驶阶段
            if action == 1:
                traffic_manager.force_lane_change(maincar, True)
            elif action == 2:
                traffic_manager.force_lane_change(maincar, False)

        # === 推进世界时间 ===
        if self.step_count != 251:  # 第11步已 tick 过
            self.world.tick()

        obs = self._GetObs()
        reward = self._GetReward()
        done = self._GetDone()

        if self.camera_image is not None:
            cv2.imshow("Monitor", self.camera_image)
            cv2.waitKey(1)

        velocity = self.cars['maincar'].get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        if self.speed_history_ptr < len(self.speed_history):
            self.speed_history[self.speed_history_ptr] = speed
            self.speed_history_ptr += 1

        # 为防止无限增长（清理缓存），保留最近 1000 步速度


        # 记录每步速度
        main_vel = self.cars['maincar'].get_velocity()
        speed = np.linalg.norm([main_vel.x, main_vel.y])
        if self.speed_history_ptr < len(self.speed_history):
            self.speed_history[self.speed_history_ptr] = speed
            self.speed_history_ptr += 1

        return obs, reward, done, {}

    def _setup_vehicles(self):
        model3 = self.blp_lib.filter("model3")[0]
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        spawn_points = self.world.get_map().get_spawn_points()

        # === 固定生成目标点（带鲁棒性） ===
        target_location = carla.Location(x=7.056836, y=-207.482941, z=0.300000)
        sorted_points = sorted(
            spawn_points,
            key=lambda sp: sp.location.distance(target_location)
        )

        main_car = None
        for idx, sp in enumerate(sorted_points[:20]):  # 最多尝试前10个最近点
            main_car = self.world.try_spawn_actor(model3, sp)
            if main_car:
            # self.logger.info(f"✅ 主车成功生成于第 {idx + 1} 个最近点，位置: {sp.location}, yaw: {sp.rotation.yaw}")
                break




        if not main_car:
            raise Exception("❌ 无法在目标点附近生成主车，尝试了前50个最近 spawn 点")

        self.cars['maincar'] = main_car


        # === 摄像头 ===
        camera_bp = self.blp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(
            carla.Location(x=-5.0, z=8.0),
            carla.Rotation(pitch=-45, yaw=0, roll=0)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=main_car)
        self.camera.listen(self._camera_callback)
        self.sensorlist.append(self.camera)
        self.camera_destroyed = False

        # === 碰撞传感器 ===
        collision_bp = self.blp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=main_car)
        self.collision_sensor.listen(self._collision_callback)
        self.sensorlist.append(self.collision_sensor)

        # === 对抗车辆生成（含多样化行为） ===
        MAX_OPPONENTS = 600
        opponent_count = 0
        main_loc = main_car.get_location()
        for sp in spawn_points:
            if sp.location.distance(main_loc) <= 1050 and sp.location.distance(main_loc) > 160.0:
                if opponent_count >= MAX_OPPONENTS:
                    break
                opponent = self.world.try_spawn_actor(model3, sp)
                if opponent:
                    opponent.set_autopilot(True, traffic_manager.get_port())

                    r = random.random()
                    if r < 0.2:
                        speed_diff = 30
                    elif r < 0.7:
                        speed_diff = 0
                    else:
                        speed_diff = -5

                    traffic_manager.vehicle_percentage_speed_difference(opponent, speed_diff)
                    self.opponent_speed_settings[opponent.id] = speed_diff  # ✅记录速度设定

                    self.actor_list.append(opponent)
                    opponent_count += 1

        # === 状态初始化 ===
        self.lane_index = CENTER_LANE
        self.vel_ref = VEHICLE_VEL
        self.past_pos = main_car.get_transform().location

    def _collision_callback(self, event):
        self.collision_event = True

    def _camera_callback(self, image):
        with self.camera_lock:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_image = array

    def _GetObs(self):
        observations = {}
        main_car = self.cars['maincar']
        main_tf = main_car.get_transform()
        main_loc = main_tf.location
        main_vel = main_car.get_velocity()
        main_acc = main_car.get_acceleration()
        yaw = main_tf.rotation.yaw

        # 当前主车 lane_id 和左右 lane id
        carla_map = self.world.get_map()
        main_wp = carla_map.get_waypoint(main_loc, project_to_road=True)
        lane_id = float(main_wp.lane_id)
        left_lane_id = float(main_wp.get_left_lane().lane_id) if main_wp.get_left_lane() else -999
        right_lane_id = float(main_wp.get_right_lane().lane_id) if main_wp.get_right_lane() else -999

        # 主车状态（10维）：位置3 + 速度3 + 加速度3 + yaw
        maincar_obs = np.array([
            main_loc.x, main_loc.y, main_loc.z,
            main_vel.x, main_vel.y, main_vel.z,
            main_acc.x, main_acc.y, main_acc.z,
            yaw
        ], dtype=np.float32)

        # 拼接换道神经网络概率（3维）
        lane_change_probs = self._get_lane_change_probs()

        # 拼接车道 ID（3维）
        lane_info = np.array([lane_id, left_lane_id, right_lane_id], dtype=np.float32)

        # 最终主车状态为 16 维
        observations['maincar'] = np.concatenate([maincar_obs, lane_change_probs, lane_info])

        # 获取最近 5 辆对抗车，7维每辆：dx, dy, dvx, dvy, dax, day, dyaw
        locations = [actor.get_location() for actor in self.actor_list if actor != main_car and actor.is_alive]
        vehicles = [actor for actor in self.actor_list if actor != main_car and actor.is_alive]

        nearest_vehicles = []
        if locations:
            from scipy.spatial import KDTree
            kd_tree = KDTree([(loc.x, loc.y) for loc in locations])
            k = min(5, len(locations))
            distances, indices = kd_tree.query([main_loc.x, main_loc.y], k=k)

            if isinstance(indices, (np.integer, int)):  # 单个索引转为列表
                indices = [indices]

            for idx in indices:
                veh = vehicles[idx]
                rel = veh.get_location()
                dx = rel.x - main_loc.x
                dy = rel.y - main_loc.y
                tf = veh.get_transform()
                vel = veh.get_velocity()
                acc = veh.get_acceleration()
                yaw = tf.rotation.yaw
                nearest_vehicles.append(np.array([
                    dx, dy,
                    veh.get_velocity().x - main_vel.x, veh.get_velocity().y - main_vel.y,
                    veh.get_acceleration().x - main_acc.x, veh.get_acceleration().y - main_acc.y,
                    veh.get_transform().rotation.yaw - yaw
                ], dtype=np.float32))

        # 补足空位
        while len(nearest_vehicles) < 5:
            nearest_vehicles.append(np.zeros(7, dtype=np.float32))

        # 拼接观测向量：主车16维 + 5×7 = 51维
        observation_array = np.concatenate([observations['maincar']] + nearest_vehicles)

        # 维度检查
        assert observation_array.shape[0] == 51, f"观测维度错误，应为51，当前为{observation_array.shape[0]}"
        return observation_array

    def _calculate_distance(self, car1, car2=None, loc2=None):
        loc1 = car1.get_transform().location
        if car2 is not None:
            loc2 = car2.get_transform().location
        elif loc2 is None:
            raise ValueError("Both car2 and loc2 cannot be None")
        distance = np.sqrt((loc2.x - loc1.x) ** 2 + (loc2.y - loc1.y) ** 2)
        return distance

    def _clear_area_around(self, location, radius=10.0):
        actors = self.world.get_actors().filter("vehicle.*")
        for actor in actors:
            actor_loc = actor.get_location()
            distance = np.sqrt((actor_loc.x - location.x) ** 2 + (actor_loc.y - location.y) ** 2)
            if distance <= radius:
                self.logger.info(f"Removing vehicle {actor.id} near spawn area (distance = {distance:.2f})")
                try:
                    actor.destroy()
                except RuntimeError as e:
                    return
                    # self.logger.warning(f"Failed to destroy vehicle {actor.id}: {e}")

    @staticmethod
    def get_lane_direction_angle(vehicle, world):
        vehicle_location = vehicle.get_location()
        carla_map = world.get_map()
        waypoint = carla_map.get_waypoint(vehicle_location)
        lane_yaw = waypoint.transform.rotation.yaw
        return lane_yaw

    def _GetReward(self):
        reward = 0
        main_car = self.cars['maincar']
        transform = main_car.get_transform()
        velocity = main_car.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        map = self.world.get_map()
        main_wp = map.get_waypoint(transform.location, project_to_road=True)

        # 🚫 惩罚项
        if self.collision_event:
            return -30
        if main_wp.lane_type != carla.LaneType.Driving:
            self.logger.warning("主车离开机动车道！")
            return -30

        speed = np.linalg.norm([velocity.x, velocity.y])
        reward += 0.001 * speed  # 每秒1m/s，奖励0.05

        # ✅ 稳定换道检测（持续5步保持在新车道）
        current_lane_id = main_wp.lane_id
        if self.step_count < 5:
            self.previous_lane_id = current_lane_id
            self.lane_stable_steps = 0
        else:
            if current_lane_id != self.previous_lane_id:
                self.previous_lane_id = current_lane_id
                self.lane_stable_steps = 1
            else:
                self.lane_stable_steps += 1
                if self.lane_stable_steps == 5:
                    lane_change_step = self.step_count - 5
                    if lane_change_step not in self.lane_change_indices:
                        self.lane_change_indices.append(lane_change_step)
                        reward += 5


                        # ✅ 新增：换道时是否成功避开原车道慢车
                        original_lane_id = self.previous_lane_id
                        main_tf = main_car.get_transform()
                        main_loc = main_tf.location
                        main_yaw = main_tf.rotation.yaw

                        for veh in self.actor_list:
                            if not veh.is_alive:
                                continue
                            veh_wp = self.world.get_map().get_waypoint(veh.get_location(), project_to_road=True)
                            if veh_wp.lane_id != original_lane_id:
                                continue
                            dx = veh.get_location().x - main_loc.x
                            dy = veh.get_location().y - main_loc.y
                            rel_angle = np.degrees(np.arctan2(dy, dx)) - main_yaw
                            rel_angle = (rel_angle + 360) % 360
                            if 45 < rel_angle < 135:  # 粗略判断是否在前方
                                speed_diff = self.opponent_speed_settings.get(veh.id, 0)
                                if speed_diff >= 30:  # 是慢车
                                    reward += 10
                                    self.logger.info(f"🏁 成功换道避开慢速车（ID: {veh.id}），奖励 +30")
                                    break

        return reward

    def _GetDone(self):
            # ✅ 终止1：碰撞
            if self.collision_event:
                return True

            # ✅ 终止2：步数超限
            if self.step_count > self.time_sleep:
                return True

            # ✅ 终止3：主车离开机动车道
            main_car = self.cars['maincar']
            map = self.world.get_map()
            main_wp = map.get_waypoint(main_car.get_location(), project_to_road=True)

            if main_wp.lane_type != carla.LaneType.Driving:
                self.logger.warning("🚫 主车离开机动车道，结束训练！")
                return True

            # ✅ 终止4：主车逆行
            car_yaw = main_car.get_transform().rotation.yaw
            lane_yaw = main_wp.transform.rotation.yaw
            yaw_diff = abs(car_yaw - lane_yaw) % 360
            yaw_diff = min(yaw_diff, 360 - yaw_diff)

            if yaw_diff > 90:
                self.logger.warning("🚫 主车逆行，结束训练！")
                return True

            return False

    def close(self):
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)

        for actor in self.actor_list:
            actor_id = actor.id
            if not self.actor_destroyed.get(actor_id, False):
                try:
                    actor.destroy()
                    self.actor_destroyed[actor_id] = True
                    time.sleep(0.1)
                except RuntimeError as e:
                    self.logger.error(f"Failed to destroy actor {actor_id}: {e}")
        for sensor in self.sensorlist:
            sensor_id = sensor.id
            if not self.sensor_destroyed.get(sensor_id, False):
                try:
                    sensor.destroy()
                    self.sensor_destroyed[sensor_id] = True
                    time.sleep(0.1)
                except RuntimeError as e:
                    self.logger.error(f"Failed to destroy sensor {sensor_id}: {e}")
        if self.camera is not None and not self.camera_destroyed:
            try:
                self.logger.info("Destroying camera actor")
                self.camera.destroy()
                self.camera_destroyed = True
                time.sleep(0.1)
            except RuntimeError as e:
                self.logger.error(f"Failed to destroy camera: {e}")
                self.camera_destroyed = True
        cv2.destroyAllWindows()

        self.logger.info("Environment resources cleaned up.")
