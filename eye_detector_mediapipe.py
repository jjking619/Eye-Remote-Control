import cv2
import numpy as np
from collections import deque
import time
import mediapipe as mp

class MediaPipeEyeDetector:
    def __init__(self):
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )        
        
        # MediaPipe 绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 眼部关键点索引
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        
        # 嘴巴关键点索引（用于头部姿态估计）
        self.MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        
        # 配置参数
        self.GAZING_STABILITY_THRESHOLD = 25  # 注视稳定性阈值（放宽要求）
        self.EAR_THRESHOLD = 0.21  # 眼睛纵横比阈值
        self.VERTICAL_MOVEMENT_THRESHOLD = 3  # 垂直移动阈值（降低敏感度要求）
        self.VERTICAL_MOVEMENT_RESET_TIME = 0.8  # 垂直动作重置时间（秒）
        
        # 数据缓存
        self.face_position_history = deque(maxlen=15)  # 增加历史记录长度
        self.left_ear_history = deque(maxlen=30)  # 增加历史记录长度
        self.right_ear_history = deque(maxlen=30)  # 增加历史记录长度
        self.eyes_state_history = deque(maxlen=5)  # 眼睛状态历史记录，用于稳定判断
        self.last_vertical_action_time = 0  # 上次垂直动作时间
        
        # FPS计算相关
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
        print("使用 MediaPipe 眼睛检测器")
    
    def calculate_ear(self, eye_landmarks):
        """计算眼睛纵横比 (Eye Aspect Ratio)"""
        # 计算眼部关键点之间的距离
        # 垂直距离
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        # 水平距离
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # 计算 EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_eyes_state(self, frame):
        """使用 MediaPipe 检测眼睛状态"""
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 计算FPS
        self.frame_count += 1
        if self.frame_count % 10 == 0:  # 每10帧更新一次FPS
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.start_time = time.time()
            self.frame_count = 0
        
        detection_result = {
            'face_detected': False,
            'eyes_closed': False,
            'is_gazing': False,
            'vertical_movement': None,
            'left_ear': 0,
            'right_ear': 0,
            'eye_center': None,
            'fps': self.fps
        }
        
        # 处理帧
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # 如果没有检测到人脸，重置垂直动作跟踪
            current_time = time.time()
            if current_time - self.last_vertical_action_time > self.VERTICAL_MOVEMENT_RESET_TIME:
                self.last_vertical_action_time = 0
            
            # 当没有人脸时，将眼睛状态标记为开放（避免误判为闭眼）
            # 修复：当face_detected为False时，eyes_closed应该为False
            self.eyes_state_history.append(True)  # False 表示眼睛睁开
            return detection_result
        
        detection_result['face_detected'] = True
        
        # 获取第一个人脸的关键点
        face_landmarks = results.multi_face_landmarks[0]
        
        # 提取眼部关键点坐标
        h, w = frame.shape[:2]
        left_eye_points = []
        right_eye_points = []
        
        # 左眼关键点
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            left_eye_points.append(np.array([x, y]))
        
        # 右眼关键点
        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            right_eye_points.append(np.array([x, y]))
        
        # 计算眼睛纵横比
        left_ear = self.calculate_ear(left_eye_points)
        right_ear = self.calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        detection_result['left_ear'] = left_ear
        detection_result['right_ear'] = right_ear
        
        # 眼睛状态检测 - 使用历史记录进行平滑处理
        current_eyes_closed = False
        
        # 使用动态阈值检测闭眼
        if len(self.left_ear_history) >= 10 and len(self.right_ear_history) >= 10:
            # 计算历史平均值和标准差
            left_avg = np.mean(self.left_ear_history)
            right_avg = np.mean(self.right_ear_history)
            left_std = np.std(self.left_ear_history)
            right_std = np.std(self.right_ear_history)
            
            # 使用动态阈值检测闭眼
            left_threshold = left_avg - 1.5 * left_std
            right_threshold = right_avg - 1.5 * right_std
            
            # 如果任意一只眼睛闭合，则认为是闭眼状态
            if left_ear < max(left_threshold, self.EAR_THRESHOLD) or right_ear < max(right_threshold, self.EAR_THRESHOLD):
                current_eyes_closed = True
        else:
            # 使用固定阈值
            if avg_ear < self.EAR_THRESHOLD:
                current_eyes_closed = True
        
        # 记录历史值
        self.left_ear_history.append(left_ear)
        self.right_ear_history.append(right_ear)
        
        # 将当前眼睛状态添加到历史记录中
        self.eyes_state_history.append(current_eyes_closed)
        
        # 使用多数投票法决定最终的眼睛状态，增加稳定性
        if len(self.eyes_state_history) >= 3:
            # 统计最近几帧中闭眼的次数
            closed_count = sum(self.eyes_state_history)
            total_count = len(self.eyes_state_history)
            
            # 如果超过一半的帧是闭眼状态，则判定为闭眼
            if closed_count >= total_count * 0.6:
                detection_result['eyes_closed'] = True
            else:
                detection_result['eyes_closed'] = False
        else:
            # 历史记录不足时，使用当前状态
            detection_result['eyes_closed'] = current_eyes_closed
        
        # 计算眼睛中心位置（使用两个眼睛的中心点）
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        eye_center = ((left_eye_center + right_eye_center) / 2).astype(int)
        detection_result['eye_center'] = (int(eye_center[0]), int(eye_center[1]))
        
        # 记录人脸中心位置和时间
        self.face_position_history.append((tuple(eye_center), time.time()))
        
        # 检测注视状态（基于位置稳定性）
        if len(self.face_position_history) >= 5:
            recent_positions = [pos for pos, _ in list(self.face_position_history)[-5:]]
            x_variance = np.var([p[0] for p in recent_positions])
            y_variance = np.var([p[1] for p in recent_positions])
            total_variance = x_variance + y_variance
            
            detection_result['is_gazing'] = total_variance < self.GAZING_STABILITY_THRESHOLD
        
        # 检测垂直移动
        detection_result['vertical_movement'] = self._detect_vertical_movement()
        
        return detection_result
    
    def _detect_vertical_movement(self):
        """检测垂直方向的移动"""
        current_time = time.time()
        
        # 如果历史数据不足，返回None
        if len(self.face_position_history) < 8:
            return None
            
        # 检查时间窗口内的数据
        recent_data = list(self.face_position_history)
        recent_times = [t for _, t in recent_data]
        
        # 只考虑最近1秒内的数据
        valid_indices = [i for i, t in enumerate(recent_times) if current_time - t <= 1.0]
        if len(valid_indices) < 8:
            return None
            
        # 获取有效数据
        valid_data = [recent_data[i] for i in valid_indices]
        if len(valid_data) < 8:
            return None
            
        # 分析最近8帧的垂直位置变化
        recent_positions = [pos for pos, _ in valid_data[-8:]]
        first_half = recent_positions[:4]
        second_half = recent_positions[-4:]
        
        first_avg_y = np.mean([p[1] for p in first_half])
        second_avg_y = np.mean([p[1] for p in second_half])
        
        vertical_change = second_avg_y - first_avg_y
        
        # 调试信息
        if len(recent_positions) >= 8:
            print(f"垂直移动检测: 变化量={vertical_change:.2f}, 阈值={self.VERTICAL_MOVEMENT_THRESHOLD}")
        
        # 检查是否需要重置动作状态
        if current_time - self.last_vertical_action_time > self.VERTICAL_MOVEMENT_RESET_TIME:
            self.last_vertical_action_time = 0
        
        # 判断垂直移动方向
        if vertical_change > self.VERTICAL_MOVEMENT_THRESHOLD:
            # 眼睛快速由上到下移动（向前翻页）
            if self.last_vertical_action_time == 0:
                self.last_vertical_action_time = current_time
                print(f"检测到向下眼球移动: {vertical_change:.2f}")
                return "down"
        elif vertical_change < -self.VERTICAL_MOVEMENT_THRESHOLD:
            # 眼睛快速由下到上移动（向后翻页）
            if self.last_vertical_action_time == 0:
                self.last_vertical_action_time = current_time
                print(f"检测到向上眼球移动: {vertical_change:.2f}")
                return "up"
        
        return None
    
    def draw_landmarks(self, frame, detection_result):
        """在帧上绘制关键点（用于调试）"""
        if detection_result['eye_center']:
            center_x, center_y = detection_result['eye_center']
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 绘制注视稳定性的可视化
            if detection_result['is_gazing']:
                cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), 2)
        
        # 显示 EAR 值
        if detection_result['left_ear'] > 0 and detection_result['right_ear'] > 0:
            cv2.putText(frame, f"Left EAR: {detection_result['left_ear']:.2f}", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Right EAR: {detection_result['right_ear']:.2f}", (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # # 显示FPS
        # cv2.putText(frame, f"FPS: {detection_result.get('fps', 0):.1f}", (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)