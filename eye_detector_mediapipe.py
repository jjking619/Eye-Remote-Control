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
        self.LEFT_EYE_INDICES = [33, 159, 158, 133, 153, 145]  # 顺序：p1, p2, p3, p4, p5, p6
        self.RIGHT_EYE_INDICES = [362, 386, 385, 263, 380, 374]  # 顺序：p1, p2, p3, p4, p5, p6
        
        # 鼻子关键点索引（用于头部稳定性检测）
        self.NOSE_INDICES = [1, 4, 6, 168, 197, 195, 5]
        
        # 配置参数
        self.GAZING_STABILITY_THRESHOLD = 35  # 凝视稳定性阈值
        self.GAZING_CONFIRMATION_FRAMES = 12  # 凝视确认需要的连续帧数（降低要求）
        self.GAZING_BREAK_FRAMES = 15  # 打破凝视需要的连续不稳定帧数（增加要求）
        self.GAZING_BLINK_TOLERANCE = 8  # 凝视中允许的眨眼最大帧数
        
        self.EAR_THRESHOLD = 0.21  # 眼睛纵横比闭眼阈值
        self.EAR_BLINK_THRESHOLD = 0.18  # 眨眼阈值
        self.EAR_OPEN_THRESHOLD = 0.25  # 眼睛完全睁开阈值
        
        # 眨眼检测参数
        self.BLINK_FRAME_THRESHOLD = 4  # 眨眼持续时间阈值（帧数）
        self.BLINK_COOLDOWN = 10  # 眨眼冷却时间（帧数）
        
        # 数据缓存
        self.face_position_history = deque(maxlen=25)
        self.left_ear_history = deque(maxlen=40)
        self.right_ear_history = deque(maxlen=40)
        self.eyes_state_history = deque(maxlen=30)
        self.nose_position_history = deque(maxlen=25)
        
        # 眼睛状态跟踪
        self.eye_state = "open"
        self.blink_counter = 0
        self.closed_counter = 0
        self.blink_cooldown_counter = 0
        self.in_blink_phase = False  # 是否在眨眼阶段
        self.blink_start_frame = 0  # 眨眼开始帧数
        
        # 凝视状态跟踪
        self.gazing_state = "not_gazing"  # not_gazing, transitioning, gazing
        self.gazing_confirm_counter = 0
        self.gazing_break_counter = 0
        
        # FPS计算相关
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
        print("使用 MediaPipe 眼睛检测器（凝视时眨眼不影响视频播放）")
    
    def calculate_ear(self, eye_landmarks):
        """使用标准6点法计算眼睛纵横比 (Eye Aspect Ratio)"""
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_position_variance(self, position_history):
        """计算位置历史记录的方差"""
        if len(position_history) < 5:
            return 1000  # 返回一个大值表示不稳定
        
        positions = [pos for pos, _ in position_history]
        x_variance = np.var([p[0] for p in positions])
        y_variance = np.var([p[1] for p in positions])
        return x_variance + y_variance
    
    def update_eye_state(self, avg_ear):
        """更新眼睛状态机"""
        if self.blink_cooldown_counter > 0:
            self.blink_cooldown_counter -= 1
        
        # 状态转移逻辑
        if self.eye_state == "open":
            if avg_ear < self.EAR_BLINK_THRESHOLD:
                self.eye_state = "closing"
                self.blink_counter = 1
                self.closed_counter = 0
                self.in_blink_phase = True
                self.blink_start_frame = self.frame_count
                
        elif self.eye_state == "closing":
            if avg_ear < self.EAR_BLINK_THRESHOLD:
                self.blink_counter += 1
                if self.blink_counter > self.BLINK_FRAME_THRESHOLD:
                    self.eye_state = "closed"
                    self.closed_counter = self.blink_counter
            else:
                self.eye_state = "open"
                self.blink_counter = 0
                self.in_blink_phase = False
                
        elif self.eye_state == "closed":
            if avg_ear > self.EAR_OPEN_THRESHOLD:
                self.eye_state = "opening"
                self.blink_counter = 0
            else:
                self.closed_counter += 1
                
        elif self.eye_state == "opening":
            if avg_ear > self.EAR_OPEN_THRESHOLD:
                self.blink_counter -= 1
                if self.blink_counter <= 0:
                    self.eye_state = "open"
                    self.blink_counter = 0
                    self.closed_counter = 0
                    self.in_blink_phase = False
            else:
                self.eye_state = "closed"
                
        return self.eye_state
    
    def update_gazing_state(self, position_variance):
        """更新凝视状态机 - 简化版本，只根据头部稳定性判断"""
        is_stable = position_variance < self.GAZING_STABILITY_THRESHOLD
        
        if self.gazing_state == "not_gazing":
            if is_stable:
                self.gazing_confirm_counter += 1
                self.gazing_break_counter = 0
                
                if self.gazing_confirm_counter >= self.GAZING_CONFIRMATION_FRAMES:
                    self.gazing_state = "gazing"
                    self.gazing_confirm_counter = 0
            else:
                self.gazing_confirm_counter = 0
                self.gazing_state = "not_gazing"
                
        elif self.gazing_state == "gazing":
            if not is_stable:
                self.gazing_break_counter += 1
                self.gazing_confirm_counter = 0
                
                if self.gazing_break_counter >= self.GAZING_BREAK_FRAMES:
                    self.gazing_state = "not_gazing"
                    self.gazing_break_counter = 0
            else:
                self.gazing_break_counter = 0
                self.gazing_state = "gazing"
        
        return self.gazing_state
    
    def detect_eyes_state(self, frame):
        """使用 MediaPipe 检测眼睛状态"""
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
            'is_blinking': False,
            'is_short_blink': False,  # 是否为短时眨眼（在凝视中应忽略）
            'eye_state': 'unknown',
            'is_gazing': False,
            'gazing_state': 'not_gazing',
            'left_ear': 0,
            'right_ear': 0,
            'avg_ear': 0,
            'eye_center': None,
            'position_variance': 1000,
            'fps': self.fps
        }
        
        # 处理帧
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            self.eye_state = "open"
            self.blink_counter = 0
            self.closed_counter = 0
            self.in_blink_phase = False
            self.gazing_state = "not_gazing"
            self.gazing_confirm_counter = 0
            self.gazing_break_counter = 0
            
            detection_result['eyes_closed'] = True
            detection_result['eye_state'] = 'no_face'
            detection_result['gazing_state'] = 'not_gazing'
            return detection_result
        
        detection_result['face_detected'] = True
        
        # 获取第一个人脸的关键点
        face_landmarks = results.multi_face_landmarks[0]
        
        # 提取眼部关键点坐标
        h, w = frame.shape[:2]
        left_eye_points = []
        right_eye_points = []
        nose_points = []
        
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
        
        # 鼻子关键点
        for idx in self.NOSE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            nose_points.append(np.array([x, y]))
        
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
        
        # 确定是否为眨眼（短时闭眼）
        is_blinking = False
        is_short_blink = False
        
        if eye_state == "closed":
            detection_result['eyes_closed'] = True
            # 判断是否为短时眨眼（持续时间在阈值内）
            if self.closed_counter <= self.BLINK_FRAME_THRESHOLD:
                is_short_blink = True
                is_blinking = True
        elif eye_state == "closing":
            detection_result['eyes_closed'] = avg_ear < self.EAR_THRESHOLD
            if (self.blink_counter <= self.BLINK_FRAME_THRESHOLD and 
                self.blink_cooldown_counter == 0 and
                self.in_blink_phase):
                is_blinking = True
                if self.blink_counter <= 3:  # 非常短的闭眼
                    is_short_blink = True
        else:
            detection_result['eyes_closed'] = False
        
        detection_result['is_blinking'] = is_blinking
        detection_result['is_short_blink'] = is_short_blink
        
        # 记录眼睛状态历史
        self.eyes_state_history.append(eye_state)
        
        # 计算眼睛中心位置
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        eye_center = ((left_eye_center + right_eye_center) / 2).astype(int)
        detection_result['eye_center'] = (int(eye_center[0]), int(eye_center[1]))
        
        # 计算鼻子中心位置
        nose_center = np.mean(nose_points, axis=0).astype(int)
        
        # 记录位置历史
        current_time = time.time()
        self.face_position_history.append((tuple(eye_center), current_time))
        self.nose_position_history.append((tuple(nose_center), current_time))
        
        # 计算位置方差
        if len(self.face_position_history) >= 5 and len(self.nose_position_history) >= 5:
            eye_variance = self.calculate_position_variance(self.face_position_history)
            nose_variance = self.calculate_position_variance(self.nose_position_history)
            position_variance = (eye_variance + nose_variance) / 2.0
            detection_result['position_variance'] = position_variance
            
            # 更新凝视状态（简化版，不考虑眨眼）
            gazing_state = self.update_gazing_state(position_variance)
            detection_result['gazing_state'] = gazing_state
            detection_result['is_gazing'] = (gazing_state == "gazing")
        else:
            detection_result['is_gazing'] = False
            detection_result['gazing_state'] = "not_gazing"
        
        return detection_result
    
    def draw_landmarks(self, frame, detection_result):
        """在帧上绘制关键点和信息"""
        if detection_result['eye_center']:
            center_x, center_y = detection_result['eye_center']
            
            # 绘制凝视状态可视化
            if detection_result['is_gazing']:
                # 绿色圆圈表示凝视状态
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 3)
                cv2.putText(frame, "GAZING", (center_x - 40, center_y - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # 红色圆圈表示未凝视
                cv2.circle(frame, (center_x, center_y), 30, (0, 0, 255), 2)
        
        # 显示 EAR 值和眼睛状态
        if detection_result['left_ear'] > 0 and detection_result['right_ear'] > 0:
            # 显示平均EAR
            cv2.putText(frame, f"EAR: {detection_result['avg_ear']:.3f}", (10, frame.shape[0] - 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示眼睛状态
            state = detection_result['eye_state']
            if state == "open":
                status_color = (0, 255, 0)
                status_text = "Open"
            elif state == "closing":
                status_color = (0, 165, 255)
                status_text = "Closing"
            elif state == "closed":
                status_color = (0, 0, 255)
                status_text = "Closed"
            elif state == "opening":
                status_color = (255, 255, 0)
                status_text = "Opening"
            else:
                status_color = (255, 255, 255)
                status_text = state
            
            cv2.putText(frame, f"Eyes: {status_text}", (10, frame.shape[0] - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # 显示凝视状态
            gazing_state = detection_result['gazing_state']
            if gazing_state == "gazing":
                gaze_color = (0, 255, 0)
                gaze_text = "GAZING"
                # 在凝视状态下，显示视频播放状态
                cv2.putText(frame, "VIDEO: PLAYING", (frame.shape[1] - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # 在凝视状态下，如果眨眼也不暂停视频
                if detection_result['is_blinking']:
                    cv2.putText(frame, "BLINK (GAZING)", (frame.shape[1] - 200, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                gaze_color = (0, 0, 255)
                gaze_text = "NOT GAZING"
                # 在非凝视状态下，显示视频暂停状态
                cv2.putText(frame, "VIDEO: PAUSED", (frame.shape[1] - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 在非凝视状态下，眨眼会导致视频暂停
                if detection_result['is_blinking']:
                    cv2.rectangle(frame, (frame.shape[1] - 200, 60), (frame.shape[1] - 10, 100), (0, 0, 255), -1)
                    cv2.putText(frame, "BLINK (PAUSED)", (frame.shape[1] - 190, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Gaze: {gaze_text}", (10, frame.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_color, 2)
            
            # 显示凝视计数器（调试用）
            cv2.putText(frame, f"Gaze Confirm: {self.gazing_confirm_counter}/{self.GAZING_CONFIRMATION_FRAMES}", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 显示FPS
            cv2.putText(frame, f"FPS: {detection_result['fps']:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)