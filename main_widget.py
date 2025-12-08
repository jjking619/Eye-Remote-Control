#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import time
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QFileDialog, QMessageBox, QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

# 导入现有的模块
sys.path.append(os.path.dirname(__file__))
from eye_detector_mediapipe import MediaPipeEyeDetector
from action_controller_simple import SimpleActionController, ControlMode
from media_controller_simple_fallback import SimpleMediaController

class VideoCaptureThread(QThread):
    frame_ready = Signal(object)
    finished = Signal()
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.detecting = True  # 默认启用检测
        self.show_landmarks = True  # 默认显示关键点
        
        # 组件初始化
        self.eye_detector = MediaPipeEyeDetector()
        self.action_controller = SimpleActionController()
        self.action_controller.switch_mode(ControlMode.VIDEO)  # 固定为视频模式
        
    def find_available_camera(self):
        """自动检测可用的摄像头"""
        for i in range(10):  # 检查前10个摄像头ID
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    print(f"找到可用摄像头，ID: {i}")
                    return i
            cap.release()
        print("未找到可用摄像头")
        return None
        
    def start_capture(self, camera_id=None):
        """启动摄像头捕获，如果未指定camera_id则自动检测"""
        # 如果没有指定摄像头ID，则自动检测
        if camera_id is None:
            camera_id = self.find_available_camera()
            if camera_id is None:
                raise Exception("未找到可用的摄像头设备")
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        self.running = True
        self.start()
        
    def stop_capture(self):
        self.running = False
        self.wait()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def toggle_detection(self, detecting):
        self.detecting = detecting
        
    def toggle_landmarks(self, show):
        self.show_landmarks = show
            
    def run(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                processed_frame = frame.copy()
                
                # 如果启用检测，则处理帧
                if self.detecting:
                    # 检测眼睛状态
                    detection_result = self.eye_detector.detect_eyes_state(processed_frame)
                    
                    # 处理动作
                    command = self.action_controller.process_detection(detection_result)
                    
                    # 绘制关键点
                    if self.show_landmarks:
                        self.eye_detector.draw_landmarks(processed_frame, detection_result)
                    
                    # 发出命令信号（只处理视频命令）
                    if command:
                        self.command_detected.emit(command)
                
                # 发出帧准备好的信号
                self.frame_ready.emit(processed_frame)
                
            time.sleep(0.03)  # ~30 FPS
            
        self.finished.emit()
    
    command_detected = Signal(str)  # command

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.media_controller = SimpleMediaController()
        self.video_thread = VideoCaptureThread()
        self.current_video_file = ""
        self.video_loaded = False
        
        # 连接视频线程信号
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.command_detected.connect(self.handle_command)
        self.video_thread.finished.connect(self.on_video_stopped)
        
        self.init_ui()
        self.auto_start_camera()  # 自动启动摄像头
        
    def init_ui(self):
        self.setWindowTitle('AI Eye Remote Control - 简化版')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 视频显示区域
        self.video_label = QLabel("正在启动摄像头...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        main_layout.addWidget(self.video_label)
        
        # 控制面板
        controls_layout = QHBoxLayout()
        
        # 摄像头控制组
        camera_group = QGroupBox("摄像头控制")
        camera_layout = QVBoxLayout()
        
        self.stop_camera_btn = QPushButton("关闭摄像头")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        camera_layout.addWidget(self.stop_camera_btn)
        
        camera_group.setLayout(camera_layout)
        controls_layout.addWidget(camera_group)
        
        # 视频控制组
        video_group = QGroupBox("视频控制")
        video_layout = QVBoxLayout()
        
        self.select_video_btn = QPushButton("选择视频文件")
        self.select_video_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.select_video_btn)
        
        self.status_label = QLabel("状态: 未加载视频")
        video_layout.addWidget(self.status_label)
        
        video_group.setLayout(video_layout)
        controls_layout.addWidget(video_group)
        
        main_layout.addLayout(controls_layout)
        
    def auto_start_camera(self):
        """自动启动摄像头"""
        try:
            self.video_thread.start_capture()
            # QMessageBox.information(self, "提示", 
            #     "摄像头已自动启动\n眼部检测已启用\n\n控制方式：\n- 眨眼：播放/暂停视频\n\n请先选择要播放的视频文件")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法自动启动摄像头: {str(e)}")
            
    def stop_camera(self):
        self.video_thread.stop_capture()
        self.video_label.setText("摄像头已关闭")
        QMessageBox.information(self, "提示", "摄像头已关闭")
        
    def on_video_stopped(self):
        self.video_label.setText("摄像头已关闭")
        
    def toggle_detection(self, state):
        self.video_thread.toggle_detection(state == Qt.CheckState.Checked.value)
        
    def toggle_landmarks(self, state):
        self.video_thread.toggle_landmarks(state == Qt.CheckState.Checked.value)
        
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        
        if file_path:
            self.current_video_file = file_path
            if self.media_controller.load_video(file_path):
                self.video_loaded = True
                self.status_label.setText(f"状态: 已加载 {os.path.basename(file_path)}")
                QMessageBox.information(self, "成功", 
                    f"已加载视频: {os.path.basename(file_path)}\n\n现在可以通过眨眼来控制视频的播放/暂停了！")
            else:
                self.video_loaded = False
                self.status_label.setText("状态: 加载失败")
                QMessageBox.warning(self, "失败", f"无法加载视频: {os.path.basename(file_path)}")
                
    def handle_command(self, command):
        """处理从视频线程发出的命令"""
        if command and self.video_loaded:
            if command == "play":
                self.media_controller.play_video()
                self.status_label.setText("状态: 播放中")
            elif command == "pause":
                self.media_controller.pause_video()
                self.status_label.setText("状态: 已暂停")
            
    def update_frame(self, frame):
        """更新视频帧显示"""
        # 将OpenCV的BGR格式转换为RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # 缩放图片以适应标签大小
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.video_thread.stop_capture()
        self.media_controller.stop_video()
        event.accept()
        

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()