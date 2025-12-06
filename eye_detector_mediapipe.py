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
        
        # 标准EAR计算使用的6个关键点索引
        # 左眼：上眼皮(159, 145)，下眼皮(158, 153)，眼角(33, 133)
        self.LEFT_EYE_INDICES = [33, 159, 158, 133, 153, 145]  # 顺序：p1, p2, p3, p4, p5, p6
        # 右眼：上眼皮(386, 374)，下眼皮(385, 380)，眼角(362, 263)
        self.RIGHT_EYE_INDICES = [362, 386, 385, 263, 380, 374]  # 顺序：p1, p2, p3, p4, p5, p6
        
        # 配置参数
        self.GAZING_STABILITY_THRESHOLD = 25  # 注视稳定性阈值
        self.EAR_THRESHOLD = 0.21  # 眼睛纵横比闭眼阈值
        self.EAR_BLINK_THRESHOLD = 0.18  # 眨眼阈值
        self.EAR_OPEN_THRESHOLD = 0.25  # 眼睛完全睁开阈值
        self.VERTICAL_MOVEMENT_THRESHOLD = 3  # 垂直移动阈值
        self.VERTICAL_MOVEMENT_RESET_TIME = 0.8  # 垂直动作重置时间（秒）
        
        # 眨眼检测参数
        self.BLINK_FRAME_THRESHOLD = 3  # 眨眼持续时间阈值（帧数）
        self.BLINK_COOLDOWN = 2  # 眨眼冷却时间（帧数）
        
        # 数据缓存
        self.face_position_history = deque(maxlen=15)
        self.left_ear_history = deque(maxlen=30)
        self.right_ear_history = deque(maxlen=30)
        self.eyes_state_history = deque(maxlen=30)  # 增加历史记录长度
        
        # 眼睛状态跟踪
        self.eye_state = "open"  # open, closing, closed, opening
        self.blink_counter = 0
        self.closed_counter = 0
        self.blink_cooldown_counter = 0
        self.last_vertical_action_time = 0
        
        # FPS计算相关
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
        print("使用 MediaPipe 眼睛检测器（改进版）")
    
    def calculate_ear(self, eye_landmarks):
        """使用标准6点法计算眼睛纵横比 (Eye Aspect Ratio)"""
        # 标准EAR计算公式: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        
        # 计算垂直距离1（上眼皮到下眼皮）
        A = np.linalg.norm(p2 - p6)
        # 计算垂直距离2（上眼皮到下眼皮）
        B = np.linalg.norm(p3 - p5)
        # 计算水平距离（眼角到眼角）
        C = np.linalg.norm(p1 - p4)
        
        # 避免除以0
        if C == 0:
            return 0.0
        
        # 计算EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def update_eye_state(self, avg_ear):
        """更新眼睛状态机"""
        # 减少眨眼冷却计数器
        if self.blink_cooldown_counter > 0:
            self.blink_cooldown_counter -= 1
        
        # 状态转移逻辑
        if self.eye_state == "open":
            # 睁眼状态 -> 如果EAR低于眨眼阈值，开始闭眼
            if avg_ear < self.EAR_BLINK_THRESHOLD:
                self.eye_state = "closing"
                self.blink_counter = 1
                self.closed_counter = 0
                
        elif self.eye_state == "closing":
            # 闭眼过程中 -> 继续闭眼
            if avg_ear < self.EAR_BLINK_THRESHOLD:
                self.blink_counter += 1
                # 如果闭眼时间超过阈值，进入闭眼状态
                if self.blink_counter > self.BLINK_FRAME_THRESHOLD:
                    self.eye_state = "closed"
                    self.closed_counter = self.blink_counter
            else:
                # EAR恢复，回到睁眼状态（可能是短暂抖动）
                self.eye_state = "open"
                self.blink_counter = 0
                
        elif self.eye_state == "closed":
            # 闭眼状态 -> 如果EAR高于睁眼阈值，开始睁开
            if avg_ear > self.EAR_OPEN_THRESHOLD:
                self.eye_state = "opening"
                self.blink_counter = 0
            else:
                # 保持闭眼
                self.closed_counter += 1
                
        elif self.eye_state == "opening":
            # 睁眼过程中 -> 如果EAR稳定在睁眼阈值以上，回到睁眼状态
            if avg_ear > self.EAR_OPEN_THRESHOLD:
                self.blink_counter -= 1
                if self.blink_counter <= 0:
                    self.eye_state = "open"
                    self.blink_counter = 0
                    self.closed_counter = 0
            else:
                # EAR又下降，回到闭眼状态
                self.eye_state = "closed"
                
        return self.eye_state
    
    def detect_eyes_state(self, frame):
        """使用 MediaPipe 检测眼睛状态"""
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 计算FPS
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.start_time = time.time()
            self.frame_count = 0
        
        detection_result = {
            'face_detected': False,
            'eyes_closed': False,
            'is_blinking': False,  # 眨眼（快速闭眼-睁开）
            'eye_state': 'unknown',  # 眼睛状态
            'is_gazing': False,
            'vertical_movement': None,
            'left_ear': 0,
            'right_ear': 0,
            'avg_ear': 0,
            'eye_center': None,
            'fps': self.fps
        }
        
        # 处理帧
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # 如果没有检测到人脸，重置状态
            current_time = time.time()
            if current_time - self.last_vertical_action_time > self.VERTICAL_MOVEMENT_RESET_TIME:
                self.last_vertical_action_time = 0
            
            # 重置眼睛状态
            self.eye_state = "open"
            self.blink_counter = 0
            self.closed_counter = 0
            
            detection_result['eyes_closed'] = True
            detection_result['eye_state'] = 'no_face'
            return detection_result
        
        detection_result['face_detected'] = True
        
        # 获取第一个人脸的关键点
        face_landmarks = results.multi_face_landmarks[0]
        
        # 提取眼部关键点坐标
        h, w = frame.shape[:2]
        left_eye_points = []
        right_eye_points = []
        
        # 左眼关键点（按标准6点顺序）
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            left_eye_points.append(np.array([x, y]))
        
        # 右眼关键点（按标准6点顺序）
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
        detection_result['avg_ear'] = avg_ear
        
        # 记录历史EAR值
        self.left_ear_history.append(left_ear)
        self.right_ear_history.append(right_ear)
        
        # 更新眼睛状态机
        eye_state = self.update_eye_state(avg_ear)
        detection_result['eye_state'] = eye_state
        
        # 根据状态确定眼睛是否闭合
        if eye_state == "closed":
            detection_result['eyes_closed'] = True
            # 长时间闭眼（超过眨眼帧数阈值）不是眨眼
            detection_result['is_blinking'] = False
        elif eye_state == "closing":
            # 正在闭眼过程中
            detection_result['eyes_closed'] = avg_ear < self.EAR_THRESHOLD
            # 如果闭眼时间短且不在冷却期，可能是眨眼
            if (self.blink_counter <= self.BLINK_FRAME_THRESHOLD and 
                self.blink_cooldown_counter == 0 and
                len(self.eyes_state_history) >= 3):
                # 检查历史记录，确认是从睁眼状态开始的
                recent_states = list(self.eyes_state_history)[-3:]
                open_states = [state for state in recent_states if state in ["open", "opening"]]
                if len(open_states) >= 2:  # 前几帧大多是睁眼
                    detection_result['is_blinking'] = True
                    self.blink_cooldown_counter = self.BLINK_COOLDOWN
        else:
            detection_result['eyes_closed'] = False
            detection_result['is_blinking'] = False
        
        # 记录眼睛状态历史
        self.eyes_state_history.append(eye_state)
        
        # 计算眼睛中心位置
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
        
        # 检查是否需要重置动作状态
        if current_time - self.last_vertical_action_time > self.VERTICAL_MOVEMENT_RESET_TIME:
            self.last_vertical_action_time = 0
        
        # 判断垂直移动方向
        if vertical_change > self.VERTICAL_MOVEMENT_THRESHOLD:
            # 眼睛快速由上到下移动
            if self.last_vertical_action_time == 0:
                self.last_vertical_action_time = current_time
                return "down"
        elif vertical_change < -self.VERTICAL_MOVEMENT_THRESHOLD:
            # 眼睛快速由下到上移动
            if self.last_vertical_action_time == 0:
                self.last_vertical_action_time = current_time
                return "up"
        
        return None
    
    def draw_landmarks(self, frame, detection_result):
        """在帧上绘制关键点和信息"""
        if detection_result['eye_center']:
            center_x, center_y = detection_result['eye_center']
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 绘制注视稳定性的可视化
            if detection_result['is_gazing']:
                cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), 2)
        
        # 显示 EAR 值和眼睛状态
        if detection_result['left_ear'] > 0 and detection_result['right_ear'] > 0:
            # 显示平均EAR
            cv2.putText(frame, f"EAR: {detection_result['avg_ear']:.3f}", (10, frame.shape[0] - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示眼睛状态
            state = detection_result['eye_state']
            if state == "open":
                status_color = (0, 255, 0)  # 绿色
                status_text = "Open"
            elif state == "closing":
                status_color = (0, 165, 255)  # 橙色
                status_text = "Closing"
            elif state == "closed":
                status_color = (0, 0, 255)  # 红色
                status_text = "Closed"
            elif state == "opening":
                status_color = (255, 255, 0)  # 青色
                status_text = "Opening"
            else:
                status_color = (255, 255, 255)  # 白色
                status_text = state
            
            cv2.putText(frame, f"State: {status_text}", (10, frame.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # 显示闭眼状态
            if detection_result['eyes_closed']:
                cv2.putText(frame, "Eyes: CLOSED", (10, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Eyes: OPEN", (10, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示眨眼状态
            if detection_result['is_blinking']:
                cv2.putText(frame, "BLINKING!", (10, frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # 显示注视状态
            gaze_color = (0, 255, 0) if detection_result['is_gazing'] else (0, 0, 255)
            gaze_text = "Gazing" if detection_result['is_gazing'] else "Not Gazing"
            cv2.putText(frame, f"Gaze: {gaze_text}", (frame.shape[1] - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_color, 2)
            
            # 显示FPS
            cv2.putText(frame, f"FPS: {detection_result['fps']:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)