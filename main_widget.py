#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import time
import os
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QFileDialog, QMessageBox, QGroupBox, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# 导入现有的模块
sys.path.append(os.path.dirname(__file__))
from eye_detector_mediapipe import MediaPipeEyeDetector
from action_controller_simple import SimpleActionController, ControlMode
from media_controller_simple_fallback import SimpleMediaController

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(object)
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.detecting = False
        self.show_landmarks = True
        
        # 组件初始化
        self.eye_detector = MediaPipeEyeDetector()
        self.action_controller = SimpleActionController()
        
    def start_capture(self, camera_id=0):
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
        
    def switch_mode(self, mode):
        if mode == "video":
            self.action_controller.switch_mode(ControlMode.VIDEO)
        elif mode == "document":
            self.action_controller.switch_mode(ControlMode.DOCUMENT)
            
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
                        
                    # 在画面上绘制调试信息
                    self.draw_debug_info(processed_frame, detection_result, command)
                    
                    # 发出命令信号
                    if command:
                        self.command_detected.emit(self.action_controller.mode.value, command)
                
                # 发出帧准备好的信号
                self.frame_ready.emit(processed_frame)
                
            time.sleep(0.03)  # ~30 FPS
            
        self.finished.emit()
        
    def draw_debug_info(self, frame, detection_result, last_command):
        """在画面上绘制调试信息"""
        h, w = frame.shape[:2]
        
        # 绘制状态信息
        status_lines = [
            f"Mode: {self.action_controller.mode.value}",
            f"Face: {'Yes' if detection_result['face_detected'] else 'No'}",
            f"Eyes: {'Closed' if detection_result['eyes_closed'] else 'Open'}",
            f"Gazing: {'Yes' if detection_result['is_gazing'] else 'No'}",
            f"FPS: {detection_result.get('fps', 0):.1f}"
        ]
        
        for i, line in enumerate(status_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                       
    command_detected = pyqtSignal(str, str)  # mode, command

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.media_controller = SimpleMediaController()
        self.video_thread = VideoCaptureThread()
        self.current_video_file = ""
        
        # 连接视频线程信号
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.command_detected.connect(self.handle_command)
        self.video_thread.finished.connect(self.on_video_stopped)
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('AI Eye Remote Control - PyQt6 Version')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 视频显示区域
        self.video_label = QLabel("点击'启动摄像头'开始")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label)
        
        # 控制面板
        controls_layout = QHBoxLayout()
        
        # 摄像头控制组
        camera_group = QGroupBox("摄像头控制")
        camera_layout = QVBoxLayout()
        
        self.start_camera_btn = QPushButton("启动摄像头")
        self.start_camera_btn.clicked.connect(self.start_camera)
        camera_layout.addWidget(self.start_camera_btn)
        
        self.stop_camera_btn = QPushButton("停止摄像头")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        camera_layout.addWidget(self.stop_camera_btn)
        
        self.detect_checkbox = QCheckBox("启用眼部检测")
        self.detect_checkbox.stateChanged.connect(self.toggle_detection)
        camera_layout.addWidget(self.detect_checkbox)
        
        self.landmarks_checkbox = QCheckBox("显示关键点")
        self.landmarks_checkbox.setChecked(True)
        self.landmarks_checkbox.stateChanged.connect(self.toggle_landmarks)
        camera_layout.addWidget(self.landmarks_checkbox)
        
        # 添加启动眼部关键点可视化工具的按钮
        self.visualize_btn = QPushButton("启动眼部关键点可视化工具")
        self.visualize_btn.clicked.connect(self.start_visualization)
        camera_layout.addWidget(self.visualize_btn)
        
        camera_group.setLayout(camera_layout)
        controls_layout.addWidget(camera_group)
        
        # 模式控制组
        mode_group = QGroupBox("模式控制")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("视频模式", "video")
        self.mode_combo.addItem("文档模式", "document")
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        mode_layout.addWidget(self.mode_combo)
        
        mode_group.setLayout(mode_layout)
        controls_layout.addWidget(mode_group)
        
        # 视频控制组
        video_group = QGroupBox("视频控制")
        video_layout = QVBoxLayout()
        
        self.select_video_btn = QPushButton("选择视频文件")
        self.select_video_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.select_video_btn)
        
        self.play_video_btn = QPushButton("播放视频")
        self.play_video_btn.clicked.connect(self.play_video)
        self.play_video_btn.setEnabled(False)
        video_layout.addWidget(self.play_video_btn)
        
        self.pause_video_btn = QPushButton("暂停视频")
        self.pause_video_btn.clicked.connect(self.pause_video)
        self.pause_video_btn.setEnabled(False)
        video_layout.addWidget(self.pause_video_btn)
        
        self.stop_video_btn = QPushButton("停止视频")
        self.stop_video_btn.clicked.connect(self.stop_video)
        self.stop_video_btn.setEnabled(False)
        video_layout.addWidget(self.stop_video_btn)
        
        video_group.setLayout(video_layout)
        controls_layout.addWidget(video_group)
        
        # 文档控制组
        document_group = QGroupBox("文档控制")
        document_layout = QVBoxLayout()
        
        self.prev_page_btn = QPushButton("上一页")
        self.prev_page_btn.clicked.connect(lambda: self.control_document("page_up"))
        document_layout.addWidget(self.prev_page_btn)
        
        self.next_page_btn = QPushButton("下一页")
        self.next_page_btn.clicked.connect(lambda: self.control_document("page_down"))
        document_layout.addWidget(self.next_page_btn)
        
        document_group.setLayout(document_layout)
        controls_layout.addWidget(document_group)
        
        main_layout.addLayout(controls_layout)
        
    def start_camera(self):
        try:
            self.video_thread.start_capture()
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法启动摄像头: {str(e)}")
            
    def stop_camera(self):
        self.video_thread.stop_capture()
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.detect_checkbox.setChecked(False)
        self.video_label.setText("点击'启动摄像头'开始")
        
    def on_video_stopped(self):
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.video_label.setText("点击'启动摄像头'开始")
        
    def toggle_detection(self, state):
        self.video_thread.toggle_detection(state == Qt.CheckState.Checked.value)
        
    def toggle_landmarks(self, state):
        self.video_thread.toggle_landmarks(state == Qt.CheckState.Checked.value)
        
    def change_mode(self, text):
        mode = self.mode_combo.currentData()
        self.video_thread.switch_mode(mode)
        
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        
        if file_path:
            self.current_video_file = file_path
            if self.media_controller.load_video(file_path):
                self.play_video_btn.setEnabled(True)
                QMessageBox.information(self, "成功", f"已加载视频: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "失败", f"无法加载视频: {os.path.basename(file_path)}")
                
    def play_video(self):
        if self.current_video_file:
            if self.media_controller.play_video():
                self.play_video_btn.setEnabled(False)
                self.pause_video_btn.setEnabled(True)
                self.stop_video_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "失败", "无法播放视频")
                
    def pause_video(self):
        if self.media_controller.pause_video():
            self.play_video_btn.setEnabled(True)
            self.pause_video_btn.setEnabled(False)
            
    def stop_video(self):
        if self.media_controller.stop_video():
            self.play_video_btn.setEnabled(True)
            self.pause_video_btn.setEnabled(False)
            self.stop_video_btn.setEnabled(False)
            
    def control_document(self, command):
        self.media_controller.control_document(command)
        
    def handle_command(self, mode, command):
        """处理从视频线程发出的命令"""
        if mode == "video":
            if command == "play":
                self.play_video()
            elif command == "pause":
                self.pause_video()
        elif mode == "document":
            self.control_document(command)
            
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
        
    def start_visualization(self):
        """启动眼部关键点可视化工具"""
        try:
            # 先停止当前摄像头
            self.video_thread.stop_capture()
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            
            # 启动可视化工具
            import subprocess
            import os
            visualization_script = os.path.join(os.path.dirname(__file__), 'eye_points_visualizer.py')
            subprocess.Popen(['python3', visualization_script])
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法启动可视化工具: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()