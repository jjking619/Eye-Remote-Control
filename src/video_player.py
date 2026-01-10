import time
import os
import threading
import subprocess
import signal
import numpy as np
import av
from PySide6.QtCore import QThread, Signal, QTimer
from log import debug, error

class VideoPlayerThread(QThread):
    """Video player thread using PyAV for efficient video decoding"""
    frame_ready = Signal(object)
    playback_finished = Signal()
    video_info_ready = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.container = None
        self.video_stream = None
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
        
        # Frame decoding and conversion
        self.frame_buffer = None
        self.codec_context = None
        
        # Audio player process
        self.audio_process = None
        self.audio_process_start_time = 0
        self._pause_position = 0
        
        # PyAV specific
        self.time_base = None
        self.frame_duration = None
        
        # For seeking
        self.seek_requested = False
        self.seek_target = 0

    def load_video(self, file_path):
        """Load video file using PyAV for efficient video decoding"""
        try:
            with self._lock:
                # Close existing container
                self._close_container()
                
                # Stop any playing audio process
                self._stop_audio_process()
                
                # Open video file with PyAV
                self.container = av.open(file_path)
                
                # Find video stream
                self.video_stream = None
                for stream in self.container.streams:
                    if stream.type == 'video':
                        self.video_stream = stream
                        break
                
                if not self.video_stream:
                    error(f"No video stream found in {file_path}")
                    return False
                
                self.current_file = file_path
                
                # Get video properties
                self.video_fps = float(self.video_stream.average_rate) if self.video_stream.average_rate else 30
                self.time_base = float(self.video_stream.time_base) if self.video_stream.time_base else 1/90000
                
                # Calculate duration and total frames
                if self.video_stream.duration is not None and self.time_base > 0:
                    self.video_duration = self.video_stream.duration * self.time_base
                else:
                    # Fallback to container duration
                    self.video_duration = self.container.duration / 1000000.0 if self.container.duration else 0
                
                self.total_frames = int(self.video_duration * self.video_fps)
                
                # Get frame dimensions
                self.video_width = self.video_stream.width
                self.video_height = self.video_stream.height
                
                # Calculate frame duration
                if self.video_fps > 0:
                    self.frame_duration = 1.0 / self.video_fps
                
                self.stopped = True
                self.playing = False
                self.paused = False
                self.current_frame = 0
                self._pause_position = 0
                self.seek_requested = False
                
                # Prepare decoder context
                self.codec_context = self.video_stream.codec_context
                
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
            debug(f"Successfully loaded video with PyAV: {file_path}")
            return True
        except Exception as e:
            error(f"Failed to load video with PyAV: {e}")
            return False
    
    def _close_container(self):
        """Close PyAV container and release resources"""
        if self.container:
            try:
                self.container.close()
            except Exception as e:
                error(f"Error closing container: {e}")
            finally:
                self.container = None
                self.video_stream = None
                self.codec_context = None
    
    def _get_frame_at_position(self, frame_num):
        """Get a specific frame by frame number using seeking"""
        if not self.container or not self.video_stream or self.total_frames <= 0:
            return None
            
        try:
            # Calculate timestamp in stream's time base
            timestamp = int(frame_num / self.video_fps / self.time_base)
            
            # Seek to the timestamp
            self.container.seek(timestamp, stream=self.video_stream)
            
            # Decode frames until we get the desired one
            for packet in self.container.demux(self.video_stream):
                for frame in packet.decode():
                    # Calculate frame number from presentation time
                    frame_pts = frame.pts * self.time_base if frame.pts else 0
                    current_frame_num = int(frame_pts * self.video_fps)
                    
                    if current_frame_num >= frame_num:
                        # Convert frame to RGB numpy array
                        rgb_frame = frame.to_ndarray(format='rgb24')
                        return rgb_frame
            
        except Exception as e:
            error(f"Error getting frame {frame_num}: {e}")
        
        return None
    
    def _get_next_frame(self):
        """Get the next frame in sequence"""
        if not self.container or not self.video_stream:
            return None
            
        try:
            for packet in self.container.demux(self.video_stream):
                if packet.stream.type != 'video':
                    continue
                    
                for frame in packet.decode():
                    # Convert frame to RGB numpy array
                    rgb_frame = frame.to_ndarray(format='bgr24')
                    return rgb_frame
                    
        except Exception as e:
            error(f"Error getting next frame: {e}")
        
        return None
    
    def _stop_audio_process(self):
        """Safely stop audio process with proper resource release and device status tracking"""
        if self.audio_process:
            try:
                if self.audio_process.poll() is None:
                    os.killpg(os.getpgid(self.audio_process.pid), signal.SIGTERM)
                    
                    # Wait 0.5s for complete resource release
                    try:
                        self.audio_process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(self.audio_process.pid), signal.SIGKILL)
                    debug("PulseAudio process terminated safely")
                
            except Exception as e:
                error(f"Critical audio process termination error: {e}")
            finally:
                self.audio_process = None
                
    def _check_audio_device_status(self):
        """Check if audio devices are available"""
        try:
            result = subprocess.run(['pactl', 'list', 'sinks'], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL, 
                                    timeout=2)
            return result.returncode == 0
        except:
            try:
                result = subprocess.run(['pulseaudio', '--check'], 
                                        stdout=subprocess.DEVNULL, 
                                        stderr=subprocess.DEVNULL)
                return result.returncode == 0
            except:
                return False

    def _get_current_volume(self):
        """Get current system volume percentage (0-100) from PulseAudio"""
        try:
            result = subprocess.run(['pactl', 'get-sink-volume', '@DEFAULT_SINK@'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.DEVNULL, 
                                    text=True, 
                                    timeout=1)
            if result.returncode == 0:
                import re
                match = re.search(r'(\d+)%', result.stdout)
                if match:
                    return int(match.group(1))
        except Exception as e:
            error(f"Failed to get current volume: {e}")
        return 100

    def _start_audio(self, start_time=0):
        """Start audio playback with device status checking and retry mechanism"""
        if not self.container:
            return
            
        # Check device status before starting audio
        if not self._check_audio_device_status():
            error("Audio device not available, delaying audio start")
            for attempt in range(3):
                time.sleep(0.5 * (attempt + 1))
                if self._check_audio_device_status():
                    debug(f"Audio device available on attempt {attempt + 1}")
                    break
            else:
                error("Audio device not available after 3 attempts")
                return
                
        try:
            self._stop_audio_process()
            
            if not self._check_audio_device_status():
                error("Audio device became unavailable during process stop")
                return
                
            if os.system('which paplay > /dev/null 2>&1') == 0:
                debug("Using PulseAudio for audio playback")
                
                time.sleep(0.5)
                
                temp_audio = f'/tmp/temp_audio_{int(time.time())}_{os.getpid()}.wav'
                
                extract_cmd = [
                    'ffmpeg',
                    '-ss', str(start_time),
                    '-i', self.current_file,
                    '-acodec', 'pcm_s16le',
                    '-ac', '2',
                    '-ar', '48000',
                    '-f', 'wav',
                    '-y',
                    temp_audio
                ]
                
                subprocess.run(
                    extract_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                volume_percent = self._get_current_volume()
                volume_value = int(volume_percent * 655.36)
                debug(f"Setting audio volume to {volume_percent}% ({volume_value})")
                
                self.audio_process = subprocess.Popen(
                    ['paplay', '--volume', str(volume_value), temp_audio],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                
                def monitor_stderr():
                    try:
                        while True:
                            line = self.audio_process.stderr.readline()
                            if not line:
                                break
                            error(f"PulseAudio error: {line.decode().strip()}")
                    except Exception as e:
                        error(f"Error monitoring paplay: {e}")
                
                threading.Thread(target=monitor_stderr, daemon=True).start()
                
                self.audio_process_start_time = time.time() - start_time
                debug(f"Started PulseAudio (PID: {self.audio_process.pid})")
                
                def cleanup():
                    try:
                        time.sleep(max(0, self.video_duration - start_time + 1))
                        if os.path.exists(temp_audio):
                            os.remove(temp_audio)
                    except Exception as e:
                        error(f"Cleanup failed: {e}")
                
                threading.Thread(target=cleanup, daemon=True).start()
            
        except Exception as e:
            error(f"PulseAudio initialization failed: {e}")

    def _pause_audio(self):
        """Pause the audio process if running"""
        if self.audio_process:
            try:
                elapsed_time = time.time() - self.audio_process_start_time
                self._pause_position = elapsed_time
                self._stop_audio_process()
                return self._pause_position
            except Exception as e:
                error(f"Error pausing audio process: {e}")
        return self._pause_position

    def _resume_audio(self):
        """Resume audio from the pause position"""
        if self.container:
            try:
                self._start_audio(self._pause_position)
                debug(f"Resumed audio from position: {self._pause_position}")
            except Exception as e:
                error(f"Error resuming audio: {e}")

    def play(self):
        """Start playback"""
        with self._lock:
            was_stopped = self.stopped
            self.playing = True
            self.paused = False
            self.stopped = False
            self.last_frame_time = time.time()
            
            start_time = (self.current_frame / self.video_fps) if self.video_fps > 0 else 0
            
            if not was_stopped and self.paused:
                start_time = self._pause_position
                self.paused = False
            
            # Check if we have audio streams
            has_audio = False
            if self.container:
                for stream in self.container.streams:
                    if stream.type == 'audio':
                        has_audio = True
                        break
            
            if has_audio:
                try:
                    if was_stopped:
                        self._start_audio(start_time)
                    else:
                        self._resume_audio()
                    debug("Audio playback started")
                except Exception as e:
                    error(f"Failed to start audio playback: {e}")

    def pause(self):
        """Pause playback"""
        with self._lock:
            if self.playing and not self.stopped:
                elapsed_time = time.time() - self.last_frame_time
                frames_advanced = int(elapsed_time * self.video_fps)
                current_frame = self.current_frame + frames_advanced
                self._pause_position = current_frame / self.video_fps if self.video_fps > 0 else 0
                
                self._pause_audio()
                
            self.paused = True
            self.playing = False
            debug("Playback paused")

    def stop(self):
        """Stop playback"""
        with self._lock:
            self.playing = False
            self.paused = False
            self.stopped = True
            self.current_frame = 0
            self._pause_position = 0
            
            self._stop_audio_process()
            debug("Playback stopped")

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
            
            # Set seek flag for next frame retrieval
            self.seek_requested = True
            self.seek_target = frame_number
            
            if self.container:
                try:
                    seek_time = (frame_number / self.video_fps) if self.video_fps > 0 else 0
                    self._pause_position = seek_time
                    
                    if self.playing and not self.paused:
                        self._start_audio(seek_time)
                    elif self.paused:
                        self._pause_position = seek_time
                except Exception as e:
                    error(f"Failed to seek audio: {e}")

    def run(self):
        """Main playback loop using PyAV"""
        while not self.exiting:
            with self._lock:
                playing = self.playing
                paused = self.paused
                stopped = self.stopped
                seek_requested = self.seek_requested
                seek_target = self.seek_target
                
            if stopped or not playing or paused:
                time.sleep(0.01)
                continue
                
            if not self.container or not self.video_stream:
                time.sleep(0.01)
                continue
            
            # Handle seeking
            if seek_requested:
                with self._lock:
                    self.seek_requested = False
                    self.current_frame = seek_target
                    
                # Get frame at seek position
                frame = self._get_frame_at_position(seek_target)
                if frame is not None:
                    self.frame_ready.emit(frame)
                
                # Reset timing
                with self._lock:
                    self.last_frame_time = time.time()
                
                # Continue to normal playback
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
                    self._stop_audio_process()
                    continue
            
            # Get next frame
            frame = self._get_next_frame()
            if frame is not None:
                self.frame_ready.emit(frame)
            else:
                # If no frame was returned, we might be at the end
                with self._lock:
                    if self.current_frame >= self.total_frames - 1:
                        self.playing = False
                        self.stopped = True
                        self.playback_finished.emit()
                        self._stop_audio_process()
            
            # Maintain frame rate
            if self.frame_duration:
                sleep_time = max(0, self.frame_duration - (time.time() - current_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.033)
            
        debug("Video player thread exited")

    def shutdown(self):
        """Safely shut down thread"""
        debug("Shutting down video player thread")
        with self._lock:
            self.exiting = True
            self.playing = False
            self.paused = False
            self.stopped = True
            self._pause_position = 0
            
            self._stop_audio_process()
            self._close_container()