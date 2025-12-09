#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import time
import os
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QFileDialog, QMessageBox, QGroupBox, QCheckBox, QFrame,
    QSplitter, QGridLayout, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation
from PySide6.QtGui import QImage, QPixmap

# 导入现有的模块
sys.path.append(os.path.dirname(__file__))
from eye_detector_mediapipe import MediaPipeEyeDetector
from action_controller_simple import SimpleActionController

class VideoCaptureThread(QThread):
    frame_ready = Signal(object)
    detection_status = Signal(dict)  # 发送检测状态
    fps_updated = Signal(float)  # 发送FPS更新
    finished = Signal()
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.detecting = True
        self.show_landmarks = True
        
         # 添加退出标志
        self.exiting = False
        
        # 组件初始化
        self.eye_detector = MediaPipeEyeDetector()
        self.action_controller = SimpleActionController()
        
        # FPS计算
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_command = None
        self.last_face_detected_time = time.time()
        
    def find_available_camera(self):
        """自动检测可用的摄像头"""
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    return i
            cap.release()
        return None
        
    def start_capture(self, camera_id=None):
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
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.start()
        
    def stop_capture(self):
        """改进的停止方法"""
        self.running = False
        self.exiting = True
        
        # 等待线程结束，但设置超时
        if self.isRunning():
            self.wait(2000)  # 最多等待2秒
            
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
                # 计算FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:  # 每秒更新一次
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                    self.fps_updated.emit(self.fps)
                
                processed_frame = frame.copy()
                detection_result = {}
                
                # 如果启用检测，则处理帧
                if self.detecting:
                    try:
                        # 检测眼睛状态
                        detection_result = self.eye_detector.detect_eyes_state(processed_frame)
                        
                        # 发送检测状态
                        self.detection_status.emit(detection_result)
                        
                        # 根据新的控制逻辑处理动作
                        # 新逻辑：播放视频时，眼睛注视屏幕则继续播放，眼睛闭上或离开屏幕，则暂停播放
                        command = None
                        face_detected = detection_result.get('face_detected', False)
                        
                        if face_detected:
                            # 更新最后检测到脸部的时间
                            self.last_face_detected_time = current_time
                            
                            # 检查眼睛是否闭合
                            eyes_closed = detection_result.get('eyes_closed', False)
                            
                            # 检查是否在凝视
                            is_gazing = detection_result.get('is_gazing', False)
                            
                            # 新逻辑：如果眼睛闭合或没有凝视，则暂停
                            if eyes_closed or not is_gazing:
                                command = "pause"
                            else:
                                command = "play"
                        else:
                            # 如果超过1秒没有检测到脸部，暂停视频
                            if current_time - self.last_face_detected_time > 1.0:
                                command = "pause"
                        
                        # 绘制关键点（可选）
                        if self.show_landmarks and face_detected:
                            self.eye_detector.draw_landmarks(processed_frame, detection_result)
                        
                        # 发出命令信号
                        if command and command != self.last_command:
                            self.command_detected.emit(command)
                            self.last_command = command
                            
                    except Exception as e:
                        print(f"检测错误: {e}")
                        # 发送空状态表示检测失败
                        self.detection_status.emit({})
                else:
                    # 如果检测被禁用，发送空状态
                    self.detection_status.emit({})
                
                # 发出帧准备好的信号
                self.frame_ready.emit(processed_frame)
                
            time.sleep(0.03)  # ~30 FPS
            
        # 清理资源
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 释放 MediaPipe 资源
        try:
            self.eye_detector.close()
        except:
            pass
            
        print("摄像头线程已退出")
        self.finished.emit()

    command_detected = Signal(str)

class VideoPlayerThread(QThread):
    """视频播放线程"""
    frame_ready = Signal(object)
    playback_finished = Signal()
    video_info_ready = Signal(dict)  # 发送视频信息
    seek_requested = Signal(int)  # 新增信号：请求跳转到指定帧

    def __init__(self):
        super().__init__()
        self.cap = None
        self.playing = False
        self.paused = False
        self.stopped = True
        self.current_file = ""
        self.video_fps = 30
        self.current_frame = 0 
        self.total_frames = 0
        self.video_width = 0
        self.video_height = 0
        self.video_duration = 0
         # 添加退出标志
        self.exiting = False

    def load_video(self, file_path):
        """加载视频文件"""
        try:
            # 如果已经有视频在加载，先释放
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                print(f"无法打开视频文件: {file_path}")
                return False
                
            self.current_file = file_path
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 确保帧率有效
            if self.video_fps <= 0:
                self.video_fps = 30  # 默认值
                
            # 计算视频时长
            if self.video_fps > 0 and self.total_frames > 0:
                self.video_duration = self.total_frames / self.video_fps
            else:
                self.video_duration = 0
                
            # 准备视频信息
            video_info = {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'width': self.video_width,
                'height': self.video_height,
                'fps': self.video_fps,
                'total_frames': self.total_frames,
                'duration': self.video_duration
            }
            
            # 发送视频信息
            self.video_info_ready.emit(video_info)
            
            return True
        except Exception as e:
            print(f"加载视频失败: {e}")
            return False
    
    def play(self):
        """开始播放"""
        self.playing = True
        self.paused = False
        self.stopped = False
        
    def pause(self):
        """暂停播放"""
        self.paused = True
        
    def stop(self):
        """停止播放"""
        self.playing = False
        self.paused = False
        self.stopped = True
        self.current_frame = 0
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
    def seek(self, frame_num):
        """跳转到指定帧"""
        if self.cap and 0 <= frame_num < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.current_frame = frame_num
            
    def get_position(self):
        """获取当前播放位置"""
        if self.cap and self.total_frames > 0:
            return self.current_frame / self.total_frames
        return 0
    
    def run(self):
        """播放线程主循环"""
        while not self.exiting:
            if self.stopped:
                time.sleep(0.1)
                continue
                
            if not self.playing or self.paused:
                time.sleep(0.1)
                continue
                
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame += 1
                    self.frame_ready.emit(frame)
                    
                    # 控制播放速度
                    sleep_time = 1.0 / self.video_fps
                    time.sleep(sleep_time)
                    
                    # 检查是否播放完毕
                    if self.current_frame >= self.total_frames:
                        self.playing = False
                        self.stopped = True
                        self.current_frame = 0
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.playback_finished.emit()
                else:
                    # 播放完毕
                    self.playing = False
                    self.stopped = True
                    self.current_frame = 0
                    if self.cap:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.playback_finished.emit()
            else:
                time.sleep(0.1)
        
        # 清理资源
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("视频播放线程已退出")
    def shutdown(self):
            """安全关闭线程"""
            self.exiting = True
            self.playing = False
            self.paused = False
            self.stopped = True
 
  # ==================== 新增全屏播放窗口类 ====================
class FullScreenPlayer(QWidget):
    """全屏播放窗口"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        # 设置窗口标志，使其成为一个全屏窗口
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 视频显示区域
        self.video_label = QLabel("正在加载视频...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        
        # 调整叠加层的位置和样式
        self.detection_overlay = QLabel(self.video_label)
        self.detection_overlay.setStyleSheet("""
            QLabel {
                color: #ff5555;
                font-size: 24px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 180);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.detection_overlay.setAlignment(Qt.AlignCenter)
        self.detection_overlay.hide()
        
        # 调整播放状态标签的位置和样式
        self.playback_status_overlay = QLabel(self.video_label)
        self.playback_status_overlay.setStyleSheet("""
            QLabel {
                color: #50fa7b;
                font-size: 24px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 180);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.playback_status_overlay.setAlignment(Qt.AlignCenter)
        self.playback_status_overlay.hide()
        
        # 添加新的状态标签（用于显示时间等信息）
        self.status_overlay = QLabel(self.video_label)
        self.status_overlay.setStyleSheet("""
            QLabel {
                color: #f1fa8c;
                font-size: 20px;
                font-weight: normal;
                background-color: rgba(0, 0, 0, 180);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.status_overlay.setAlignment(Qt.AlignCenter)
        self.status_overlay.hide()
        
        # 控制条（默认隐藏，鼠标移动时显示）
        self.control_bar = QWidget()
        self.control_bar.setObjectName("control_bar")
        self.control_bar.setFixedHeight(80)
        self.control_bar.hide()
        
        control_layout = QHBoxLayout(self.control_bar)
        control_layout.setContentsMargins(20, 0, 20, 20)
        
        # 返回按钮
        self.back_btn = QPushButton("返回")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(self.exit_fullscreen)
        
        # 播放/暂停按钮
        self.play_pause_btn = QPushButton("暂停")
        self.play_pause_btn.setFixedSize(100, 40)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        
        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        
        # 时间标签
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        
        # 状态标签（显示识别状态）
        self.status_label = QLabel("正在检测...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                padding: 5px 10px;
                background-color: rgba(0, 0, 0, 150);
                border-radius: 5px;
            }
        """)
        
        control_layout.addWidget(self.back_btn)
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addWidget(self.progress_slider, 1)
        control_layout.addWidget(self.time_label)
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(self.video_label, 1)
        main_layout.addWidget(self.control_bar)
        
        # 鼠标移动检测定时器
        self.mouse_timer = QTimer()
        self.mouse_timer.timeout.connect(self.hide_controls)
        self.mouse_timer.setSingleShot(True)
        
        # 控制条显示/隐藏动画
        self.control_animation = QPropertyAnimation(self.control_bar, b"windowOpacity")
        self.control_animation.setDuration(300)
        
        # 状态标签定时器（自动隐藏）
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.hide_status)
        self.status_timer.setSingleShot(True)
        
        # 叠加层显示定时器
        self.overlay_timer = QTimer()
        self.overlay_timer.timeout.connect(self.hide_overlays)
        self.overlay_timer.setSingleShot(True)
        
    def setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #000000;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 30);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 50);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 20);
            }
            QSlider::groove:horizontal {
                border: 1px solid rgba(255, 255, 255, 50);
                height: 6px;
                background: rgba(255, 255, 255, 20);
                margin: 0px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #cccccc;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #89b4fa;
                border: 1px solid rgba(137, 180, 250, 100);
                height: 6px;
                border-radius: 3px;
            }
        """)
        
    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        self.showFullScreen()
        
    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key_Escape:
            self.exit_fullscreen()
        elif event.key() == Qt.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key_F11:
            # 切换全屏/窗口模式
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)
            
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 显示控制条"""
        super().mouseMoveEvent(event)
        self.show_controls()
        
    def show_controls(self):
        """显示控制条"""
        if not self.control_bar.isVisible():
            self.control_bar.show()
            self.control_animation.setStartValue(0)
            self.control_animation.setEndValue(1)
            self.control_animation.start()
        
        # 重置隐藏定时器
        self.mouse_timer.stop()
        self.mouse_timer.start(3000)  # 3秒后隐藏
        
    def hide_controls(self):
        """隐藏控制条"""
        self.control_animation.setStartValue(1)
        self.control_animation.setEndValue(0)
        self.control_animation.finished.connect(lambda: self.control_bar.hide())
        self.control_animation.start()
        
    def show_status(self, message, duration=2000):
        """显示状态信息"""
        self.status_label.setText(message)
        self.status_label.show()
        self.status_timer.stop()
        self.status_timer.start(duration)
        
    def hide_status(self):
        """隐藏状态信息"""
        self.status_label.hide()
        
    def show_overlays(self, detection_text="", playback_text="", status_text=""):
        """显示叠加层信息"""
        # 确保标签根据文本内容调整大小
        if detection_text:
            self.detection_overlay.setText(detection_text)
            self.detection_overlay.adjustSize()  # 根据文本调整大小
            self.detection_overlay.show()
            
        if playback_text:
            self.playback_status_overlay.setText(playback_text)
            self.playback_status_overlay.adjustSize()  # 根据文本调整大小
            self.playback_status_overlay.show()
            
        if status_text:
            self.status_overlay.setText(status_text)
            self.status_overlay.adjustSize()  # 根据文本调整大小
            self.status_overlay.show()
            
        # 确保叠加层不会重叠
        self.adjust_overlay_positions()
        
        # 重置隐藏定时器
        self.overlay_timer.stop()
        self.overlay_timer.start(2000)  # 2秒后隐藏
        
    def hide_overlays(self):
        """隐藏叠加层"""
        self.detection_overlay.hide()
        self.playback_status_overlay.hide()
        self.status_overlay.hide()
        
    def update_detection_status(self, detection_result):
        """更新检测状态显示"""
        if detection_result and detection_result.get('face_detected', False):
            eyes_closed = detection_result.get('eyes_closed', False)
            is_gazing = detection_result.get('is_gazing', False)
            
            if eyes_closed:
                self.show_status("眼睛闭合 - 视频暂停", 1000)
                self.show_overlays(
                    detection_text="眼睛闭合", 
                    playback_text="已暂停",
                    status_text="眼睛闭合"
                )
            elif not is_gazing:
                self.show_status("未注视屏幕 - 视频暂停", 1000)
                self.show_overlays(
                    detection_text="未注视屏幕", 
                    playback_text="已暂停",
                    status_text="未注视屏幕"
                )
            else:
                self.show_status("正在注视 - 视频播放", 1000)
                self.show_overlays(
                    detection_text="正在注视", 
                    playback_text="正在播放",
                    status_text="正在注视"
                )
        else:
            self.show_status("未检测到人脸 - 视频暂停", 1000)
            self.show_overlays(
                detection_text="未检测到人脸", 
                playback_text="已暂停",
                status_text="未检测到人脸"
            )
        
        
    def exit_fullscreen(self):
        """退出全屏模式"""
        self.close()
        if self.parent_window:
            self.parent_window.showNormal()
            self.parent_window.show()
            
    def toggle_play_pause(self):
        """切换播放/暂停"""
        if self.parent_window:
            if self.parent_window.video_player_thread.playing and not self.parent_window.video_player_thread.paused:
                self.parent_window.pause_video()
                self.play_pause_btn.setText("播放")
                self.show_status("已暂停")
                self.show_overlays(playback_text="已暂停")
            else:
                self.parent_window.play_video()
                self.play_pause_btn.setText("暂停")
                self.show_status("正在播放")
                self.show_overlays(playback_text="正在播放")
                
    def update_video_frame(self, frame):
        """更新视频帧"""
        if self.parent_window:
            self.parent_window.display_frame(self.video_label, frame)
            
    def update_progress(self, position, duration):
        """更新进度条和时间显示"""
        if not self.progress_slider.isSliderDown():  # 如果用户没有拖动进度条
            self.progress_slider.setValue(int(position * 1000))
            
        # 更新时间显示
        current_str = f"{int(position * duration // 60):02d}:{int(position * duration % 60):02d}"
        total_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
        self.time_label.setText(f"{current_str} / {total_str}")
        

    def adjust_overlay_positions(self):
        """调整叠加层位置以避免重叠"""
        # 获取视频标签的尺寸
        video_rect = self.video_label.rect()
        
        # 调整检测结果叠加层位置（左上角）
        if self.detection_overlay.isVisible():
            self.detection_overlay.adjustSize()
            detection_size = self.detection_overlay.sizeHint()
            self.detection_overlay.setGeometry(
                20,  # 左边距
                20,  # 上边距
                detection_size.width(),
                detection_size.height()
            )
            
        # 调整播放状态叠加层位置（右上角）
        if self.playback_status_overlay.isVisible():
            self.playback_status_overlay.adjustSize()
            playback_size = self.playback_status_overlay.sizeHint()
            self.playback_status_overlay.setGeometry(
                video_rect.width() - playback_size.width() - 20,  # 右边距20像素
                20,  # 上边距
                playback_size.width(),
                playback_size.height()
            )
            
        # 调整状态叠加层位置（底部居中）
        if self.status_overlay.isVisible():
            self.status_overlay.adjustSize()
            status_size = self.status_overlay.sizeHint()
            self.status_overlay.setGeometry(
                (video_rect.width() - status_size.width()) // 2,  # 居中
                video_rect.height() - status_size.height() - 20,  # 底部边距20像素
                status_size.width(),
                status_size.height()
            )
        
    def hide_overlays(self):
        """隐藏叠加层"""
        self.detection_overlay.hide()
        self.playback_status_overlay.hide()
        self.status_overlay.hide()
        
    def update_detection_status(self, detection_result):
        """更新检测状态显示"""
        if detection_result and detection_result.get('face_detected', False):
            eyes_closed = detection_result.get('eyes_closed', False)
            is_gazing = detection_result.get('is_gazing', False)
            
            if eyes_closed:
                self.show_status("眼睛闭合 - 视频暂停", 1000)
                self.show_overlays(
                    detection_text="眼睛闭合", 
                    playback_text="已暂停",
                    status_text="眼睛闭合"
                )
            elif not is_gazing:
                self.show_status("未注视屏幕 - 视频暂停", 1000)
                self.show_overlays(
                    detection_text="未注视屏幕", 
                    playback_text="已暂停",
                    status_text="未注视屏幕"
                )
            else:
                self.show_status("正在注视 - 视频播放", 1000)
                self.show_overlays(
                    detection_text="正在注视", 
                    playback_text="正在播放",
                    status_text="正在注视"
                )
        else:
            self.show_status("未检测到人脸 - 视频暂停", 1000)
            self.show_overlays(
                detection_text="未检测到人脸", 
                playback_text="已暂停",
                status_text="未检测到人脸"
            )        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.video_player_thread = VideoPlayerThread()
        self.video_thread = VideoCaptureThread()
        self.current_video_file = ""
        self.video_loaded = False
        self.camera_active = False
        self.is_fullscreen = False
        self.video_duration = 0
        self.video_position = 0
        self.is_slider_pressed = False
        
         # 新增：全屏播放窗口
        self.fullscreen_player = None
        self.is_in_fullscreen_mode = False
        
        # 连接信号
        self.video_thread.frame_ready.connect(self.update_camera_frame)
        self.video_thread.command_detected.connect(self.handle_command)
        self.video_thread.detection_status.connect(self.update_detection_status)
        self.video_thread.fps_updated.connect(self.update_fps_display)
        self.video_thread.finished.connect(self.on_video_stopped)
        
        self.video_player_thread.frame_ready.connect(self.update_video_frame)
        self.video_player_thread.playback_finished.connect(self.on_playback_finished)
        self.video_player_thread.video_info_ready.connect(self.update_video_info)

        # 设置样式
        self.setup_styles()
        
        self.init_ui()
        self.auto_start_camera()
        
        # 启动视频播放线程
        self.video_player_thread.start()
        
        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500)
        
        # 视频进度更新定时器
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(100)
        
    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
            }
            QGroupBox {
                color: #89b4fa;
                font-weight: bold;
                border: 2px solid #585b70;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #313244;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #585b70;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6c7086;
            }
            QPushButton:pressed {
                background-color: #45475a;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #7f849c;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #585b70;
            }
            QCheckBox::indicator:checked {
                background-color: #89b4fa;
                border-color: #89b4fa;
            }
            QFrame#status_frame {
                background-color: #313244;
                border-radius: 8px;
                border: 1px solid #585b70;
            }
            QLabel#status_value {
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 4px;
            }
            QSlider {
                min-height: 20px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #585b70;
                height: 8px;
                background: #313244;
                margin: 0px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                border: 1px solid #5c81e3;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #89b4fa;
                border: 1px solid #5c81e3;
                height: 8px;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #585b70;
                border-radius: 4px;
                text-align: center;
                background-color: #313244;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 4px;
            }
        """)
        
    def init_ui(self):
        self.setWindowTitle('👁️ AI Vision Control')
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题栏
        title_frame = QFrame()
        title_frame.setFixedHeight(50)
        title_frame.setStyleSheet("background-color: #313244; border-radius: 8px;")
        
        title_layout = QHBoxLayout(title_frame)
        
        title_label = QLabel("👁️ AI Vision Control")
        title_label.setStyleSheet("color: #89b4fa; font-size: 18px; font-weight: bold;")
        
        # 全屏按钮
        self.fullscreen_btn = QPushButton("全屏")
        self.fullscreen_btn.setFixedSize(160, 30)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        
         # 新增：全屏播放按钮
        self.fullscreen_play_btn = QPushButton("🎬 全屏播放模式")
        self.fullscreen_play_btn.setFixedSize(180, 30)
        self.fullscreen_play_btn.clicked.connect(self.enter_fullscreen_play_mode)
        self.fullscreen_play_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.fullscreen_play_btn)
        title_layout.addWidget(self.fullscreen_btn)
        main_layout.addWidget(title_frame)
        
        # 主体内容区域 - 水平分割
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧 - 视频显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # 摄像头显示区域
        camera_group = QGroupBox("📷 摄像头画面")
        camera_layout = QVBoxLayout()
        
        self.camera_display = QLabel("正在启动摄像头...")
        self.camera_display.setAlignment(Qt.AlignCenter)
        self.camera_display.setMinimumSize(640, 360)
        self.camera_display.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border-radius: 8px;
                border: 2px solid #585b70;
                color: #ffffff;
                font-size: 14px;
            }
        """)
        
        camera_layout.addWidget(self.camera_display)
        camera_group.setLayout(camera_layout)
        left_layout.addWidget(camera_group)
        
        # 视频播放区域
        video_group = QGroupBox("🎬 视频播放")
        video_layout = QVBoxLayout()
        
        self.video_display = QLabel("请选择视频文件")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 360)
        self.video_display.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border-radius: 8px;
                border: 2px solid #585b70;
                color: #ffffff;
                font-size: 14px;
            }
        """)
        
        # 视频控制条
        video_controls = QWidget()
        video_controls_layout = QVBoxLayout(video_controls)
        
        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        self.progress_slider.sliderMoved.connect(self.on_progress_slider_moved)
        self.progress_slider.sliderPressed.connect(self.on_progress_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_progress_slider_released)
        
        # 时间显示和按钮
        control_row = QWidget()
        control_layout = QHBoxLayout(control_row)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        
        self.video_play_btn = QPushButton("播放")
        self.video_play_btn.clicked.connect(self.play_video)
        self.video_play_btn.setFixedSize(60, 30)
        
        self.video_pause_btn = QPushButton("暂停")
        self.video_pause_btn.clicked.connect(self.pause_video)
        self.video_pause_btn.setFixedSize(60, 30)
        
        self.video_stop_btn = QPushButton("停止")
        self.video_stop_btn.clicked.connect(self.stop_video)
        self.video_stop_btn.setFixedSize(60, 30)
        
        control_layout.addWidget(self.time_label)
        control_layout.addStretch()
        control_layout.addWidget(self.video_play_btn)
        control_layout.addWidget(self.video_pause_btn)
        control_layout.addWidget(self.video_stop_btn)
        
        video_controls_layout.addWidget(self.progress_slider)
        video_controls_layout.addWidget(control_row)
        
        video_layout.addWidget(self.video_display)
        video_layout.addWidget(video_controls)
        video_group.setLayout(video_layout)
        left_layout.addWidget(video_group)
        
        left_layout.addStretch()
        
        # 右侧 - 控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        
        # 实时状态显示
        status_group = QGroupBox("📊 系统状态")
        status_layout = QGridLayout()
        
        # 摄像头状态
        cam_status_label = QLabel("📷 摄像头:")
        cam_status_label.setStyleSheet("color: #a6adc8;")
        
        self.cam_status = QLabel("运行中")
        self.cam_status.setObjectName("status_value")
        self.cam_status.setStyleSheet("background-color: #a6e3a1; color: #000000;")
        
        # FPS显示
        fps_label = QLabel("⚡ 摄像头FPS:")
        fps_label.setStyleSheet("color: #a6adc8;")
        
        self.fps_display = QLabel("0.0")
        self.fps_display.setObjectName("status_value")
        self.fps_display.setStyleSheet("background-color: #cba6f7; color: #000000;")
        
        # 检测状态
        detect_status_label = QLabel("🔍 检测状态:")
        detect_status_label.setStyleSheet("color: #a6adc8;")
        
        self.detect_status = QLabel("正在检测...")
        self.detect_status.setObjectName("status_value")
        self.detect_status.setStyleSheet("background-color: #f9e2af; color: #000000;")
        
        # 眼睛状态
        eye_status_label = QLabel("👁️ 眼睛状态:")
        eye_status_label.setStyleSheet("color: #a6adc8;")
        
        self.eye_status = QLabel("未检测")
        self.eye_status.setObjectName("status_value")
        self.eye_status.setStyleSheet("background-color: #f9e2af; color: #000000;")
        
        # 凝视状态
        gaze_status_label = QLabel("🎯 凝视状态:")
        gaze_status_label.setStyleSheet("color: #a6adc8;")
        
        self.gaze_status = QLabel("未检测")
        self.gaze_status.setObjectName("status_value")
        self.gaze_status.setStyleSheet("background-color: #f9e2af; color: #000000;")
        
        # 视频播放状态
        video_status_label = QLabel("▶️ 视频状态:")
        video_status_label.setStyleSheet("color: #a6adc8;")
        
        self.video_status = QLabel("未加载")
        self.video_status.setObjectName("status_value")
        self.video_status.setStyleSheet("background-color: #f38ba8; color: #000000;")
        
        # 视频文件信息
        file_info_group = QGroupBox("📁 视频信息")
        file_info_layout = QVBoxLayout()
        
        self.file_name_label = QLabel("文件名: 未选择")
        self.file_name_label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        
        self.file_size_label = QLabel("分辨率: 未加载")
        self.file_size_label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        
        self.file_duration_label = QLabel("时长: 未加载")
        self.file_duration_label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        
        self.file_fps_label = QLabel("帧率: 未加载")
        self.file_fps_label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        
        file_info_layout.addWidget(self.file_name_label)
        file_info_layout.addWidget(self.file_size_label)
        file_info_layout.addWidget(self.file_duration_label)
        file_info_layout.addWidget(self.file_fps_label)
        file_info_group.setLayout(file_info_layout)
        
        # 添加到网格布局
        status_layout.addWidget(cam_status_label, 0, 0)
        status_layout.addWidget(self.cam_status, 0, 1)
        status_layout.addWidget(fps_label, 0, 2)
        status_layout.addWidget(self.fps_display, 0, 3)
        
        status_layout.addWidget(detect_status_label, 1, 0)
        status_layout.addWidget(self.detect_status, 1, 1)
        status_layout.addWidget(eye_status_label, 1, 2)
        status_layout.addWidget(self.eye_status, 1, 3)
        
        status_layout.addWidget(gaze_status_label, 2, 0)
        status_layout.addWidget(self.gaze_status, 2, 1)
        status_layout.addWidget(video_status_label, 2, 2)
        status_layout.addWidget(self.video_status, 2, 3)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        right_layout.addWidget(file_info_group)
        
        # 控制指令说明
        instruction_group = QGroupBox("📋 控制指令说明")
        instruction_layout = QVBoxLayout()
        
        instructions = QLabel(
            "<b>自动控制指令:</b><br>"
            "• 播放视频时，眼睛注视屏幕 → 继续播放<br>"
            "• 眼睛闭上或离开屏幕 → 暂停播放<br>"
            "• 未检测到人脸 → 暂停播放<br><br>"
            "<b>注意:</b><br>"
            "• 确保脸部在摄像头范围内<br>"
            "• 保持光线充足<br>"
            "• 视频开始播放后，注视屏幕以继续播放"
        )
        instructions.setStyleSheet("color: #cdd6f4; padding: 5px;")
        instructions.setWordWrap(True)
        
        instruction_layout.addWidget(instructions)
        instruction_group.setLayout(instruction_layout)
        right_layout.addWidget(instruction_group)
        
        # 摄像头控制
        camera_control_group = QGroupBox("🎮 摄像头控制")
        camera_control_layout = QVBoxLayout()
        
        self.camera_toggle_btn = QPushButton("关闭摄像头")
        self.camera_toggle_btn.clicked.connect(self.toggle_camera)
        self.camera_toggle_btn.setFixedHeight(35)
        
        self.detect_checkbox = QCheckBox("启用眼部检测")
        self.detect_checkbox.setChecked(True)
        self.detect_checkbox.stateChanged.connect(self.toggle_detection)
        
        self.landmarks_checkbox = QCheckBox("显示关键点")
        self.landmarks_checkbox.setChecked(True)
        self.landmarks_checkbox.stateChanged.connect(self.toggle_landmarks)
        
        camera_control_layout.addWidget(self.camera_toggle_btn)
        camera_control_layout.addWidget(self.detect_checkbox)
        camera_control_layout.addWidget(self.landmarks_checkbox)
        camera_control_group.setLayout(camera_control_layout)
        right_layout.addWidget(camera_control_group)
        
        # 视频文件控制
        file_control_group = QGroupBox("📁 视频文件控制")
        file_control_layout = QVBoxLayout()
        
        self.select_video_btn = QPushButton("选择视频文件")
        self.select_video_btn.clicked.connect(self.select_video)
        self.select_video_btn.setFixedHeight(40)
        
        file_control_layout.addWidget(self.select_video_btn)
        file_control_group.setLayout(file_control_layout)
        right_layout.addWidget(file_control_group)
        
        right_layout.addStretch()
        
        # 添加到分割器
        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([900, 500])
        
        main_layout.addWidget(content_splitter)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 设置全屏快捷键
        self.fullscreen_btn.setShortcut("F11")
        
    def auto_start_camera(self):
        try:
            self.video_thread.start_capture()
            self.camera_active = True
            self.camera_toggle_btn.setText("关闭摄像头")
            self.cam_status.setText("运行中")
            self.cam_status.setStyleSheet("background-color: #a6e3a1; color: #000000;")
        except Exception as e:
            self.cam_status.setText("启动失败")
            self.cam_status.setStyleSheet("background-color: #f38ba8; color: #000000;")
            QMessageBox.critical(self, "错误", f"无法自动启动摄像头: {str(e)}")
            
    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
            
    def stop_camera(self):
        self.video_thread.stop_capture()
        self.camera_active = False
        self.camera_toggle_btn.setText("启动摄像头")
        self.cam_status.setText("已关闭")
        self.cam_status.setStyleSheet("background-color: #f38ba8; color: #000000;")
        self.camera_display.setText("摄像头已关闭")
        self.camera_display.setPixmap(QPixmap())
        
    def start_camera(self):
        try:
            self.video_thread.start_capture()
            self.camera_active = True
            self.camera_toggle_btn.setText("关闭摄像头")
            self.cam_status.setText("运行中")
            self.cam_status.setStyleSheet("background-color: #a6e3a1; color: #000000;")
        except Exception as e:
            self.cam_status.setText("启动失败")
            self.cam_status.setStyleSheet("background-color: #f38ba8; color: #000000;")
            QMessageBox.critical(self, "错误", f"无法启动摄像头: {str(e)}")
        
    def on_video_stopped(self):
        self.camera_active = False
        
    def toggle_detection(self, state):
        self.video_thread.toggle_detection(state == Qt.CheckState.Checked.value)
        if state:
            self.detect_status.setText("检测中")
            self.detect_status.setStyleSheet("background-color: #a6e3a1; color: #000000;")
        else:
            self.detect_status.setText("已禁用")
            self.detect_status.setStyleSheet("background-color: #f38ba8; color: #000000;")
        
    def toggle_landmarks(self, state):
        self.video_thread.toggle_landmarks(state == Qt.CheckState.Checked.value)
        
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)")
        
        if file_path:
            self.current_video_file = file_path
            
            if self.video_player_thread.load_video(file_path):
                self.video_loaded = True
                self.video_status.setText("已加载")
                self.video_status.setStyleSheet("background-color: #a6e3a1; color: #000000;")
                # 显示第一帧
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.display_video_frame(frame)
                    cap.release()
            else:
                self.video_loaded = False
                self.video_status.setText("加载失败")
                self.video_status.setStyleSheet("background-color: #f38ba8; color: #000000;")
                QMessageBox.warning(self, "失败", f"无法加载视频: {os.path.basename(file_path)}")
    def update_video_info(self, video_info):
        """更新视频信息显示"""
        filename = video_info['filename']
        width = video_info['width']
        height = video_info['height']
        fps = video_info['fps']
        duration = video_info['duration']
        
        # 更新标签
        self.file_name_label.setText(f"文件名: {filename}")
        self.file_size_label.setText(f"分辨率: {width} × {height}")
        self.file_duration_label.setText(f"时长: {int(duration // 60):02d}:{int(duration % 60):02d}")
        self.file_fps_label.setText(f"帧率: {fps:.1f} FPS")
        
        # 更新时间显示
        self.video_duration = duration
        self.update_time_label(0, duration)
        
    def handle_command(self, command):
        """处理检测命令"""
        if command and self.video_loaded:
            if command == "play":
                self.play_video()
                # 如果在全屏模式，更新按钮文本和状态显示
                if self.is_in_fullscreen_mode and self.fullscreen_player:
                    self.fullscreen_player.play_pause_btn.setText("暂停")
                    self.fullscreen_player.show_overlays(playback_text="正在播放")
            elif command == "pause":
                self.pause_video()
                # 如果在全屏模式，更新按钮文本和状态显示
                if self.is_in_fullscreen_mode and self.fullscreen_player:
                    self.fullscreen_player.play_pause_btn.setText("播放")
                    self.fullscreen_player.show_overlays(playback_text="已暂停")
                    self.fullscreen_player.play_pause_btn.setText("播放")
                
    def play_video(self):
        if self.video_loaded:
            self.video_player_thread.play()
            self.video_status.setText("播放中")
            self.video_status.setStyleSheet("background-color: #89b4fa; color: #000000;")
                
    def pause_video(self):
        if self.video_loaded:
            self.video_player_thread.pause()
            self.video_status.setText("已暂停")
            self.video_status.setStyleSheet("background-color: #f9e2af; color: #000000;")
            
    def stop_video(self):
        if self.video_loaded:
            self.video_player_thread.stop()
            self.video_status.setText("已停止")
            self.video_status.setStyleSheet("background-color: #f38ba8; color: #000000;")
            self.progress_slider.setValue(0)
            self.update_time_label(0, self.video_duration)
            
    def update_camera_frame(self, frame):
        self.display_frame(self.camera_display, frame)
        
    def update_video_frame(self, frame):
        self.display_video_frame(frame)
        
    def display_frame(self, label, frame):
        """显示帧到指定标签"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
        
    def display_video_frame(self, frame):
        """显示视频帧"""
        self.display_frame(self.video_display, frame)
        
    def update_detection_status(self, detection_result):
        """更新检测状态"""
        if detection_result and detection_result.get('face_detected', False):
            self.eye_status.setText("检测中")
            
            # 检查眼睛是否闭合
            eyes_closed = detection_result.get('eyes_closed', False)
            if eyes_closed:
                self.eye_status.setText("眼睛闭合")
                self.eye_status.setStyleSheet("background-color: #f38ba8; color: #000000;")
            else:
                self.eye_status.setText("眼睛睁开")
                self.eye_status.setStyleSheet("background-color: #a6e3a1; color: #000000;")
                
            # 检查凝视状态
            is_gazing = detection_result.get('is_gazing', False)
            if is_gazing:
                self.gaze_status.setText("凝视中")
                self.gaze_status.setStyleSheet("background-color: #89b4fa; color: #000000;")
            else:
                self.gaze_status.setText("未凝视")
                self.gaze_status.setStyleSheet("background-color: #a6adc8; color: #000000;")
        else:
            self.eye_status.setText("未检测")
            self.eye_status.setStyleSheet("background-color: #a6adc8; color: #000000;")
            self.gaze_status.setText("未检测")
            self.gaze_status.setStyleSheet("background-color: #a6adc8; color: #000000;")
        
    def update_fps_display(self, fps):
        """更新FPS显示"""
        self.fps_display.setText(f"{fps:.1f}")
        
    def update_progress(self):
        """更新进度条"""
        if self.video_loaded and self.video_player_thread.playing and not self.video_player_thread.paused:
            position = self.video_player_thread.get_position()
            self.progress_slider.setValue(int(position * 1000))
            
            # 更新时间显示
            current_time = position * self.video_duration
            self.update_time_label(current_time, self.video_duration)
        
    def update_time_label(self, current_time, total_time):
        """更新时间显示标签"""
        current_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
        total_str = f"{int(total_time // 60):02d}:{int(total_time % 60):02d}"
        self.time_label.setText(f"{current_str} / {total_str}")
        
    def on_progress_slider_moved(self, value):
        """进度条拖动事件"""
        if self.video_loaded and not self.is_slider_pressed:
            position = value / 1000.0
            target_frame = int(position * self.video_player_thread.total_frames)
            self.video_player_thread.seek(target_frame)  # 发送信号，由播放线程处理
            
    def on_progress_slider_pressed(self):
        """进度条按下事件"""
        self.is_slider_pressed = True
        
    def on_progress_slider_released(self):
        """进度条释放事件"""
        if self.video_loaded:
            position = self.progress_slider.value() / 1000.0
            self.video_player_thread.seek(int(position * self.video_player_thread.total_frames))
        self.is_slider_pressed = False
        
    def on_playback_finished(self):
        """视频播放完成事件"""
        self.video_status.setText("播放完成")
        self.video_status.setStyleSheet("background-color: #a6e3a1; color: #000000;")
        self.progress_slider.setValue(1000)
        
    def update_status(self):
        """更新状态信息"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.statusBar().showMessage(f"就绪 | {current_time}")
        
    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.is_fullscreen:
            self.showNormal()
            self.fullscreen_btn.setText("全屏")
            self.is_fullscreen = False
        else:
            self.showFullScreen()
            self.fullscreen_btn.setText("退出全屏")
            self.is_fullscreen = True
    def enter_fullscreen_play_mode(self):
        """进入全屏播放模式"""
        if not self.video_loaded:
            QMessageBox.warning(self, "提示", "请先选择视频文件")
            return
            
        if self.fullscreen_player is None:
            self.fullscreen_player = FullScreenPlayer(self)
            
            # 连接信号
            self.video_player_thread.frame_ready.connect(self.fullscreen_player.update_video_frame)
            self.video_thread.detection_status.connect(self.fullscreen_player.update_detection_status)
            
        # 根据当前播放状态设置全屏播放按钮文本
        if self.video_player_thread.playing and not self.video_player_thread.paused:
            self.fullscreen_player.play_pause_btn.setText("暂停")
        else:
            self.fullscreen_player.play_pause_btn.setText("播放")
            
        # 隐藏主窗口，显示全屏播放器
        self.hide()
        self.fullscreen_player.show()
        self.is_in_fullscreen_mode = True
        
        # 更新状态
        self.fullscreen_player.show_status("已进入全屏播放模式")
        
    def closeEvent(self, event):
        """窗口关闭事件 - 改进版本""" 
        print("正在关闭应用，清理资源...")
        # 如果全屏播放器存在，先关闭它
        if self.fullscreen_player:
            self.fullscreen_player.close()
            self.fullscreen_player = None
        # 停止定时器
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()
        
        # 停止摄像头线程
        if hasattr(self, 'video_thread'):
            print("停止摄像头线程...")
            self.video_thread.stop_capture()
        
        # 停止视频播放线程
        if hasattr(self, 'video_player_thread'):
            print("停止视频播放线程...")
            self.video_player_thread.shutdown()  # 使用新的关闭方法
            
            # 等待线程结束
            if self.video_player_thread.isRunning():
                self.video_player_thread.quit()
                self.video_player_thread.wait(2000)  # 最多等待2秒
        
        # 强制关闭 MediaPipe 相关资源（如果可能）
        try:
            # 如果有 MediaPipe 的清理方法，调用它
            if hasattr(self, 'video_thread') and hasattr(self.video_thread, 'eye_detector'):
                self.video_thread.eye_detector.close()
        except:
            pass
        
        print("资源清理完成")
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()