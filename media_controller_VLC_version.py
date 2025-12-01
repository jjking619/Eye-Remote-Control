
#vlc_version
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
        # 确定使用的文档控制方法
        self.document_control_method = "xdotool"
    def load_video(self, video_path):
        """加载视频文件"""
        if os.path.exists(video_path):
            self.current_video = video_path
            print(f"视频加载成功: {video_path}")
            return True
        else:
            print(f"视频文件不存在: {video_path}")
            return False
    
    # 修改 play_video 方法，添加更多 VLC 参数来解决时间戳问题
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
                
                # 使用vlc播放视频（修复版本）
                self.video_process = subprocess.Popen([
                    'cvlc', 
                    # '--fullscreen', 
                    '--no-video-title-show',
                    '--play-and-exit',
                    '--demux=ffmpeg',  # 强制使用ffmpeg demuxer
                    '--video-filter=scale{width=1280,height=720}',  # 强制分辨率
                    self.current_video
                ])
                self.video_playing = True
                self.video_paused = False
                print("开始播放视频 (使用VLC)")
                return True
            except FileNotFoundError:
                try:
                    # 回退到使用mpv播放视频（推荐）
                    self.video_process = subprocess.Popen([
                        'mpv', 
                        '--no-terminal', 
                        '--fullscreen', 
                        '--keep-open=yes',
                        '--pause=no',
                        '--hwdec=auto',  # 启用硬件加速
                        '--vo=gpu',  # 使用GPU渲染
                        '--vf=scale=1280:720',  # 强制分辨率
                        '--fs',  # 全屏模式
                        self.current_video
                    ])
                    self.video_playing = True
                    self.video_paused = False
                    print("开始播放视频 (使用MPV)")
                    return True
                except FileNotFoundError:
                    print("请安装vlc或mpv: sudo apt install vlc 或 sudo apt install mpv")
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
    
    def open_pdf(self, pdf_path):
        """打开PDF文档"""
        if os.path.exists(pdf_path):
            try:
                # 设置环境变量
                env = os.environ.copy()
                env['DISPLAY'] = ':0'
                
                # 首先尝试使用 okular（如果安装了）
                result = subprocess.run(['which', 'okular'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # 使用 okular 打开 PDF
                    process = subprocess.Popen(['okular', pdf_path], env=env)
                    print(f"使用 Okular 打开PDF: {pdf_path}")
                else:
                    # 回退到 evince
                    process = subprocess.Popen(['evince', pdf_path], env=env)
                    print(f"使用 Evince 打开PDF: {pdf_path}")
                
                # 等待窗口打开
                time.sleep(2)
                
                # 尝试激活窗口
                try:
                    subprocess.run(['wmctrl', '-a', 'okular'], 
                                 capture_output=True, timeout=5, env=env)
                except Exception as e:
                    try:
                        subprocess.run(['wmctrl', '-a', 'evince'], 
                                     capture_output=True, timeout=5, env=env)
                    except Exception as e2:
                        print(f"激活PDF窗口失败: {e}")
                
                return True
            except Exception as e:
                print(f"打开PDF失败: {e}")
        else:
            print(f"PDF文件不存在: {pdf_path}")
        return False
    
    def control_document(self, command):
            """控制文档翻页"""
            print(f"执行文档控制命令: {command} (使用 xdotool 方案)")
            try:
                # 设置环境变量
                env = os.environ.copy()
                env['DISPLAY'] = ':0'
                
                # 直接使用 xdotool 方案（已知有效）
                window_activated = False
                
                # 首先尝试查找所有可能的 PDF 阅读器窗口
                pdf_apps = ['okular', 'evince', 'atril', 'xpdf']
                found_window = False
                
                for app_name in pdf_apps:
                    try:
                        # 使用 xdotool 搜索窗口
                        search_result = subprocess.run(['xdotool', 'search', '--name', app_name], 
                                                    capture_output=True, text=True, timeout=1, env=env)
                        if search_result.returncode == 0 and search_result.stdout.strip():
                            windows = search_result.stdout.strip().split('\n')
                            if windows and windows[0]:
                                window_id = windows[0]
                                # 激活窗口
                                subprocess.run(['xdotool', 'windowactivate', '--sync', window_id], 
                                            capture_output=True, timeout=1, env=env)
                                print(f"通过 xdotool 激活窗口 {window_id} ({app_name})")
                                window_activated = True
                                found_window = True
                                break
                    except Exception as e:
                        continue
                
                # 减少等待时间
                if window_activated:
                    time.sleep(0.1)
                
                # 发送按键 - 只尝试最可能有效的按键
                key = 'Page_Down' if command == "page_down" else 'Page_Up'
                print(f"发送按键: {key}")
                result = subprocess.run(['xdotool', 'key', key], 
                                    capture_output=True, text=True, timeout=1, env=env)
                if result.returncode == 0:
                    print(f"使用 {key} 按键翻页成功")
                    return
                else:
                    print(f"使用 {key} 按键翻页失败: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("文档控制超时")
            except FileNotFoundError as e:
                missing_tool = ""
                if 'xdotool' in str(e):
                    missing_tool = "xdotool"
                elif 'wmctrl' in str(e):
                    missing_tool = "wmctrl"
                elif 'qdbus' in str(e):
                    missing_tool = "qdbus"
                
                if missing_tool:
                    print(f"错误: 未找到 {missing_tool} 工具，请安装: sudo apt install {missing_tool}")
                else:
                    print(f"文档控制错误: {e}")
            except Exception as e:
                print(f"文档控制出现未知错误: {e}")
    
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