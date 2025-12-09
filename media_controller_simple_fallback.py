
import subprocess
import os
import time
import signal

class SimpleMediaController:
    def __init__(self):
        print("使用简化版媒体控制器（命令行模式）")
        self.video_playing = False
        self.current_video = None
        self.video_process = None  # 保存视频进程引用
        self.video_paused = False  # 跟踪暂停状态
    def load_video(self, video_path):
        """加载视频文件"""
        if os.path.exists(video_path):
            self.current_video = video_path
            print(f"视频加载成功: {video_path}")
            return True
        else:
            print(f"视频文件不存在: {video_path}")
            return False
    
    def play_video(self):
        """播放视频"""
        if self.current_video:
            # 如果已经有进程在运行，尝试恢复而不是重新开始
            if self.video_process and self.video_process.poll() is None:
                if self.video_paused:
                    return self.resume_video()
                else:
                    print("视频已在播放中")
                    return True
            
            try:
                # 终止可能存在的旧进程
                self.stop_video()
                
                # 使用 mpv 播放视频（推荐）
                self.video_process = subprocess.Popen([
                    'mpv', 
                    '--no-terminal', 
                    # '--fullscreen', 
                    '--keep-open=yes',
                    '--pause=no',
                    # '--hwdec=auto',  # 启用硬件加速
                    # '--vo=gpu',  # 使用GPU渲染
                    '--vf=scale=480:360',  # 强制分辨率
                    # '--fs',  # 全屏模式
                    self.current_video
                ])
                self.video_playing = True
                self.video_paused = False
                print("开始播放视频 (使用MPV)")
                return True
            except FileNotFoundError:
                print("请安装mpv: sudo apt install mpv")
                return False
        
    def pause_video(self):
        """暂停视频"""
        if self.video_playing and self.video_process:
            try:
                if self.video_process.poll() is None:  # 进程仍在运行
                    if not self.video_paused:
                        # 发送SIGSTOP信号暂停进程
                        self.video_process.send_signal(signal.SIGSTOP)
                        self.video_paused = True
                        print("视频已暂停")
                        return True
                    else:
                        print("视频已在暂停状态")
                        return True
            except Exception as e:
                print(f"暂停视频时出错: {e}")
        
        # 如果无法暂停进程，至少更新内部状态
        self.video_playing = False
        print("视频暂停（外部播放器状态可能不同步）")
        return True
    
    def resume_video(self):
        """恢复视频播放"""
        if self.video_process and (self.video_playing or self.video_paused):
            try:
                if self.video_process.poll() is None:  # 进程仍在运行
                    if self.video_paused:
                        # 发送SIGCONT信号恢复进程
                        self.video_process.send_signal(signal.SIGCONT)
                        self.video_paused = False
                        print("视频已恢复播放")
                        self.video_playing = True
                        return True
                    else:
                        print("视频已在播放状态")
                        self.video_playing = True
                        return True
            except Exception as e:
                print(f"恢复视频时出错: {e}")
        elif self.video_process is None and self.current_video:
            # 如果没有进程但有视频文件，重新播放
            return self.play_video()
        return False
    
    def stop_video(self):
        """停止视频"""
        if self.video_process:
            try:
                # 尝试优雅地终止进程
                self.video_process.terminate()
                try:
                    self.video_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # 如果进程没有响应，强制杀死
                    self.video_process.kill()
                    self.video_process.wait()
            except:
                pass
            self.video_process = None
        
        self.video_playing = False
        self.video_paused = False
        print("视频停止")
        return True
    
    
    def get_video_status(self):
        """获取视频播放状态"""
        # 检查进程是否仍在运行
        if self.video_process:
            if self.video_process.poll() is None:
                return self.video_playing and not self.video_paused
            else:
                # 进程已结束
                self.video_process = None
                self.video_playing = False
                self.video_paused = False
                return False
        return self.video_playing and not self.video_paused