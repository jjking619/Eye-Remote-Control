#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#识别眼眶的代码
import cv2
import numpy as np
import time
import mediapipe as mp
from eye_detector_mediapipe import MediaPipeEyeDetector

class EyeLandmarksVisualizer:
    def __init__(self):
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 使用与MediaPipeEyeDetector相同的索引
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        
        # EAR计算中使用的点对
        self.LEFT_EAR_POINTS = [1, 5, 2, 4, 0, 3]  # 对应 indices 中的索引
        self.RIGHT_EAR_POINTS = [1, 5, 2, 4, 0, 3]  # 对应 indices 中的索引
        
        print("眼部关键点可视化工具初始化完成")

    def visualize_ear_calculation(self, frame):
        """可视化EAR计算过程"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # 处理左眼
        left_eye_points = []
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            left_eye_points.append(np.array([x, y]))
            
        # 处理右眼
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            right_eye_points.append(np.array([x, y]))
            
        # 绘制眼部轮廓
        self._draw_eye_outline(frame, left_eye_points, (0, 255, 0))
        self._draw_eye_outline(frame, right_eye_points, (0, 255, 0))
        
        # 可视化EAR计算点
        self._visualize_ear_points(frame, left_eye_points, (0, 0, 255), "Left")
        self._visualize_ear_points(frame, right_eye_points, (255, 0, 0), "Right")
        
        # 计算并显示EAR值
        left_ear = self._calculate_ear(left_eye_points)
        right_ear = self._calculate_ear(right_eye_points)
        
        # 显示EAR值
        cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 显示当前检测器使用的阈值
        cv2.putText(frame, f"Threshold: 0.21", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 显示闭眼状态判断
        eyes_closed = left_ear < 0.21 or right_ear < 0.21
        status_text = "Eyes: CLOSED" if eyes_closed else "Eyes: OPEN"
        status_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame

    def _draw_eye_outline(self, frame, eye_points, color):
        """绘制眼部轮廓"""
        points = np.array(eye_points, np.int32)
        cv2.polylines(frame, [points], True, color, 1)
        
        # 绘制关键点
        for point in eye_points:
            cv2.circle(frame, tuple(point), 2, color, -1)

    def _visualize_ear_points(self, frame, eye_points, color, eye_name):
        """可视化用于EAR计算的关键点"""
        # EAR计算中的点:
        # A = 垂直距离1 (点1到点5)
        # B = 垂直距离2 (点2到点4)  
        # C = 水平距离 (点0到点3)
        
        pt1 = tuple(eye_points[1])  # A的第一点
        pt5 = tuple(eye_points[5])  # A的第二点
        pt2 = tuple(eye_points[2])  # B的第一点
        pt4 = tuple(eye_points[4])  # B的第二点
        pt0 = tuple(eye_points[0])  # C的第一点
        pt3 = tuple(eye_points[3])  # C的第二点
        
        # 绘制垂直距离线
        cv2.line(frame, pt1, pt5, color, 2)
        cv2.line(frame, pt2, pt4, color, 2)
        
        # 绘制水平距离线
        cv2.line(frame, pt0, pt3, (255, 255, 0), 2)  # 用黄色表示水平距离
        
        # 标记点索引
        for i, idx in enumerate([0, 1, 2, 3, 4, 5]):
            point = tuple(eye_points[idx])
            cv2.circle(frame, point, 4, (255, 255, 255), -1)
            cv2.putText(frame, str(idx), (point[0]+5, point[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _calculate_ear(self, eye_landmarks):
        """计算眼睛纵横比 (Eye Aspect Ratio)"""
        # 根据图片中的坐标点定义
        # P1: 左上角，P2: 右上角，P3: 右下角，P4: 左下角，P5: 左侧中间，P6: 右侧中间
        # 使用图片中的点索引：P1=0, P2=1, P3=2, P4=3, P5=4, P6=5
        
        # 计算眼部关键点之间的距离
        # 垂直距离
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # P2到P6
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # P3到P5
        # 水平距离
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # P1到P4
        
        # 计算 EAR
        if C == 0:
            return 0
        ear = (A + B) / (2.0 * C)
        return ear

    def visualize_eye_model(self, frame):
        """可视化完整的眼部模型，包括所有关键点"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # 处理左眼
        left_eye_points = []
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            left_eye_points.append(np.array([x, y]))
            
        # 处理右眼
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            right_eye_points.append(np.array([x, y]))
        
        # 绘制完整眼部轮廓
        self._draw_eye_outline(frame, left_eye_points, (0, 255, 0))
        self._draw_eye_outline(frame, right_eye_points, (0, 255, 0))
        
        # 标注所有关键点索引
        for i, point in enumerate(left_eye_points):
            cv2.circle(frame, tuple(point), 3, (0, 255, 255), -1)
            cv2.putText(frame, str(self.LEFT_EYE_INDICES[i]), (point[0]+5, point[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
        for i, point in enumerate(right_eye_points):
            cv2.circle(frame, tuple(point), 3, (0, 255, 255), -1)
            cv2.putText(frame, str(self.RIGHT_EYE_INDICES[i]), (point[0]+5, point[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
        return frame

def main():
    # 创建可视化工具
    visualizer = EyeLandmarksVisualizer()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 使用摄像头ID 2，与主程序一致
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    visualization_mode = "ear"  # 默认为EAR计算可视化
    
    print("眼部关键点可视化工具启动")
    print("按 'q' 键退出程序")
    print("按 'm' 键切换可视化模式 (EAR计算/完整模型)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 根据模式选择不同的可视化
        if visualization_mode == "ear":
            processed_frame = visualizer.visualize_ear_calculation(frame)
            cv2.putText(processed_frame, "Mode: EAR Calculation", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            processed_frame = visualizer.visualize_eye_model(frame)
            cv2.putText(processed_frame, "Mode: Full Eye Model", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示结果
        cv2.imshow('Eye Landmarks Visualization', processed_frame)
        
        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            visualization_mode = "model" if visualization_mode == "ear" else "ear"
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()