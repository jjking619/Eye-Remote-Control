import time
import os
import threading
import vlc
from moviepy.editor import VideoFileClip
from PySide6.QtCore import QThread, Signal
from log import debug, error 

class VideoPlayerThread(QThread):
    """Video player thread using MoviePy for video frames and VLC for audio"""
    frame_ready = Signal(object)
    playback_finished = Signal()
    video_info_ready = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.clip = None
        self.current_file = ""
        self.playing = False
        self.paused = False
        self.stopped = True
        self.video_fps = 30
        self.total_frames = 0
        self.video_width = 0
        self.video_height = 0
        self.video_duration = 0
        self.exiting = False
        self.current_frame = 0
        self.last_frame_time = 0
        self._lock = threading.RLock()
        
        # VLC player for audio
        self.vlc_instance = None
        self.vlc_player = None
        self._initialize_vlc()

    def _initialize_vlc(self):
        """Initialize VLC instance and player"""
        try:
            # Create VLC instance with options for audio only playback
            self.vlc_instance = vlc.Instance('--no-video', '--quiet')
            self.vlc_player = self.vlc_instance.media_player_new()
            debug("VLC initialized successfully")
        except Exception as e:
            error(f"Failed to initialize VLC: {e}")
            self.vlc_instance = None
            self.vlc_player = None

    def load_video(self, file_path):
        """Load video file using MoviePy for video frames and VLC for audio"""
        try:
            debug(f"Attempting to load video: {file_path}")
            
            with self._lock:
                # Release existing clip
                if self.clip:
                    self.clip.close()
                    self.clip = None
                
                # Stop any playing media in VLC
                if self.vlc_player:
                    self.vlc_player.stop()
                
                # Load new clip for frame extraction
                self.clip = VideoFileClip(file_path)
                
                self.current_file = file_path
                self.video_duration = self.clip.duration
                self.video_fps = self.clip.fps if self.clip.fps else 30
                self.total_frames = int(self.video_duration * self.video_fps)
                
                if self.clip.size:
                    self.video_width, self.video_height = self.clip.size
                else:
                    self.video_width, self.video_height = 1920, 1080
                
                self.stopped = True
                self.playing = False
                self.paused = False
                self.current_frame = 0

                # Load media into VLC player
                if self.vlc_player and self.vlc_instance:
                    media = self.vlc_instance.media_new(file_path)
                    self.vlc_player.set_media(media)

            # Prepare video information
            video_info = {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'width': self.video_width,
                'height': self.video_height,
                'fps': self.video_fps,
                'total_frames': self.total_frames,
                'duration': self.video_duration
            }

            # Emit video information
            self.video_info_ready.emit(video_info)
            debug(f"Successfully loaded video: {file_path}")
            return True
        except Exception as e:
            error(f"Failed to load video: {e}")
            return False

    def play(self):
        """Start playback"""
        with self._lock:
            self.playing = True
            self.paused = False
            self.stopped = False
            self.last_frame_time = time.time()
            
            # Start VLC audio if available
            if self.vlc_player:
                try:
                    # Calculate correct audio position based on current frame
                    position = self.current_frame / self.total_frames if self.total_frames > 0 else 0
                    # Set position and play
                    self.vlc_player.set_position(position)
                    self.vlc_player.play()
                    debug("VLC audio playback started")
                except Exception as e:
                    error(f"Failed to start VLC audio playback: {e}")

    def pause(self):
        """Pause playback"""
        with self._lock:
            self.paused = True
            # Pause VLC audio
            if self.vlc_player:
                try:
                    self.vlc_player.pause()
                    debug("VLC audio paused")
                except Exception as e:
                    error(f"Failed to pause VLC audio: {e}")

    def stop(self):
        """Stop playback"""
        with self._lock:
            self.playing = False
            self.paused = False
            self.stopped = True
            self.current_frame = 0
            
            # Stop VLC audio
            if self.vlc_player:
                try:
                    self.vlc_player.stop()
                    debug("VLC audio stopped")
                except Exception as e:
                    error(f"Failed to stop VLC audio: {e}")

    def get_position(self):
        """Get current playback position (0.0 to 1.0)"""
        with self._lock:
            if self.total_frames > 0:
                return self.current_frame / self.total_frames
        return 0.0

    def seek(self, frame_number):
        """Seek to specific frame"""
        with self._lock:
            frame_number = max(0, min(frame_number, self.total_frames - 1))
            self.current_frame = frame_number
            # When seeking, we need to update VLC position too
            if self.vlc_player:
                try:
                    # For seeking, we temporarily pause, set position, then resume if needed
                    was_playing = self.playing and not self.paused
                    if was_playing:
                        self.vlc_player.pause()
                    
                    position = frame_number / self.total_frames if self.total_frames > 0 else 0
                    self.vlc_player.set_position(position)
                    
                    if was_playing:
                        self.vlc_player.play()
                        
                except Exception as e:
                    error(f"Failed to seek VLC audio: {e}")

    def run(self):
        """Main playback loop"""
        while not self.exiting:
            with self._lock:
                playing = self.playing
                paused = self.paused
                stopped = self.stopped
                
            if stopped or not playing or paused:
                time.sleep(0.01)
                continue
                
            if not self.clip:
                time.sleep(0.01)
                continue
                
            with self._lock:
                # Calculate time-based frame advancement
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                # Advance frame based on elapsed time
                frames_to_advance = int(elapsed * self.video_fps)
                self.current_frame = min(self.current_frame + frames_to_advance, self.total_frames - 1)
                
                # Check if we've reached the end
                if self.current_frame >= self.total_frames - 1:
                    self.playing = False
                    self.stopped = True
                    self.playback_finished.emit()
                    # Stop audio
                    if self.vlc_player:
                        try:
                            self.vlc_player.stop()
                        except:
                            pass
                    continue
            
            # Get frame from clip
            try:
                timestamp = self.current_frame / self.video_fps
                frame = self.clip.get_frame(t=timestamp)
                
                # Convert BGR to RGB
                frame = frame[:, :, ::-1]
                
                # Emit frame
                self.frame_ready.emit(frame)
            except Exception as e:
                error(f"Error getting frame: {e}")
                
            # Maintain frame rate
            time.sleep(1.0 / self.video_fps if self.video_fps > 0 else 0.033)
            
        debug("Video player thread exited")

    def shutdown(self):
        """Safely shut down thread"""
        debug("Shutting down video player thread")
        with self._lock:
            self.exiting = True
            self.playing = False
            self.paused = False
            self.stopped = True
            
            # Stop VLC audio
            if self.vlc_player:
                try:
                    self.vlc_player.stop()
                except:
                    pass
            
            # Close clip
            if self.clip:
                self.clip.close()