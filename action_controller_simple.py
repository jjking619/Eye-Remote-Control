import time

class SimpleActionController:
    def __init__(self):
        self.video_playing = False
        self.last_action_time = 0
        self.action_cooldown = 0.8  # 动作冷却时间
        
        # 注视计时
        self.gazing_start_time = 0
        self.gazing_required_duration = 1.5  # 需要持续注视1.5秒才播放
        
    def process_detection(self, detection_result):
        """处理检测结果并返回控制命令"""
        current_time = time.time()
        
        # 冷却期检查
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        command = None
        
        # 只保留视频模式处理
        command = self._handle_video_mode(detection_result, current_time)
        
        if command:
            self.last_action_time = current_time
            
        return command
    
    def _handle_video_mode(self, detection_result, current_time):
        """处理视频模式下的动作"""
        # 闭眼或无人脸 → 暂停
        if detection_result['eyes_closed'] or not detection_result['face_detected']:
            if self.video_playing:
                self.video_playing = False
                self.gazing_start_time = 0
                print("检测到闭眼或无人脸，发送暂停命令")
                return "pause"
            return None
        
        # 注视检测 → 播放
        if detection_result['is_gazing']:
            if not self.video_playing:
                if self.gazing_start_time == 0:
                    self.gazing_start_time = current_time
                    print("开始注视检测...")
                elif current_time - self.gazing_start_time >= self.gazing_required_duration:
                    self.video_playing = True
                    self.gazing_start_time = 0
                    print("注视时间足够，发送播放命令")
                    return "play"
        else:
            # 未注视，如果正在播放则暂停
            if self.video_playing:
                self.video_playing = False
                self.gazing_start_time = 0
                print("未注视屏幕，发送暂停命令")
                return "pause"
            # 未注视，重置计时器
            elif self.gazing_start_time != 0:
                print("注视中断，重置计时器")
                self.gazing_start_time = 0
        
        return None
    
