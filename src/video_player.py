import time
import os
import threading
import subprocess
import signal
from moviepy.editor import VideoFileClip
from PySide6.QtCore import QThread, Signal
from log import debug, error 

class VideoPlayerThread(QThread):
    """Video player thread using MoviePy for video frames and system audio for audio"""
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
        
        # Audio player process
        self.audio_process = None
        self.audio_process_start_time = 0  # Track when audio process started
        self._pause_position = 0  # Track the position when paused

    def load_video(self, file_path):
        """Load video file using MoviePy for video frames and prepare audio"""
        try:
            debug(f"Attempting to load video: {file_path}")
            
            with self._lock:
                # Release existing clip
                if self.clip:
                    self.clip.close()
                    self.clip = None
                
                # Stop any playing audio process
                self._stop_audio_process()
                
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
                self._pause_position = 0

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

    def _start_audio(self, start_time=0):
        """Start audio playback using system audio stack (PulseAudio -> ALSA -> VLC)"""
        if not self.clip or not self.clip.audio:
            return
            
        try:
            # Stop any existing audio process
            self._stop_audio_process()
            
            # Try PulseAudio first (preferred on Quectel Pi H1)
            if os.system('which paplay > /dev/null 2>&1') == 0:
                debug("Using PulseAudio for audio playback")
                # Extract audio segment from video file starting at start_time
                temp_audio = f'/tmp/temp_audio_{int(time.time())}_{os.getpid()}.wav'
                
                # Extract audio from video file starting at start_time
                extract_cmd = [
                    'ffmpeg',
                    '-ss', str(start_time),
                    '-i', self.current_file,
                    '-acodec', 'pcm_s16le',
                    '-ac', '2',  # Stereo
                    '-ar', '48000',  # 48kHz to match hardware
                    '-f', 'wav',
                    '-y',  # Overwrite existing file
                    temp_audio
                ]
                
                subprocess.run(
                    extract_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Play the extracted audio using PulseAudio
                self.audio_process = subprocess.Popen(
                    ['paplay', temp_audio],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid  # Create new process group
                )
                self.audio_process_start_time = time.time() - start_time
                debug(f"Started PulseAudio process with PID: {self.audio_process.pid}")
                
                # Clean up temp file after playback
                def cleanup():
                    time.sleep(self.video_duration - start_time + 1)  # Wait for playback to finish
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                
                cleanup_thread = threading.Thread(target=cleanup, daemon=True)
                cleanup_thread.start()
                
            # If PulseAudio is not available, try ALSA directly
            elif os.system('which aplay > /dev/null 2>&1') == 0:
                debug("Using ALSA for audio playback")
                # Extract audio segment from video file
                temp_audio = f'/tmp/temp_audio_{int(time.time())}_{os.getpid()}.wav'
                
                extract_cmd = [
                    'ffmpeg',
                    '-ss', str(start_time),
                    '-i', self.current_file,
                    '-acodec', 'pcm_s16le',
                    '-ac', '2',  # Stereo
                    '-ar', '48000',  # 48kHz to match hardware
                    '-f', 'wav',
                    '-y',  # Overwrite existing file
                    temp_audio
                ]
                
                subprocess.run(
                    extract_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Play the extracted audio using ALSA
                self.audio_process = subprocess.Popen(
                    ['aplay', temp_audio],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid  # Create new process group
                )
                self.audio_process_start_time = time.time() - start_time
                debug(f"Started ALSA process with PID: {self.audio_process.pid}")
                
                # Clean up temp file after playback
                def cleanup():
                    time.sleep(self.video_duration - start_time + 1)  # Wait for playback to finish
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                
                cleanup_thread = threading.Thread(target=cleanup, daemon=True)
                cleanup_thread.start()
                
            # Fallback to VLC if PulseAudio and ALSA are not available
            else:
                debug("Using VLC for audio playback")
                cmd = [
                    'cvlc',
                    '--intf', 'dummy',  # No interface
                    '--no-video',       # Audio only
                    '--start-time', str(start_time),  # Start at specific time
                    '--rate', '1',      # Normal playback rate
                    '--quiet',          # Less verbose output
                    self.current_file
                ]
                
                # Start the audio process
                self.audio_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid  # Create new process group
                )
                self.audio_process_start_time = time.time() - start_time
                debug(f"Started VLC audio process with PID: {self.audio_process.pid}")
                
        except Exception as e:
            error(f"Failed to start audio process: {e}")
            # Fallback to a simple approach
            self._start_audio_simple(start_time)

    def _start_audio_simple(self, start_time=0):
        """Simple audio playback as final fallback"""
        try:
            # Try using cvlc if available
            if os.system('which cvlc > /dev/null 2>&1') == 0:
                cmd = [
                    'cvlc',
                    '--intf', 'dummy',
                    '--no-video',
                    '--play-and-exit',
                    '--start-time', str(start_time),
                    '--quiet',
                    self.current_file
                ]
                
                self.audio_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid
                )
                self.audio_process_start_time = time.time() - start_time
                debug(f"Started simple VLC audio process with PID: {self.audio_process.pid}")
            else:
                error("No audio player found (paplay, aplay, cvlc)")
        except Exception as e:
            error(f"Failed to start simple audio process: {e}")

    def _pause_audio(self):
        """Pause the audio process if running - for this implementation we stop and restart at correct position"""
        if self.audio_process:
            try:
                # Calculate the elapsed time and store it as the pause position
                elapsed_time = time.time() - self.audio_process_start_time
                self._pause_position = elapsed_time
                # For this implementation, we'll stop the current process and restart at the correct position
                self._stop_audio_process()
                return self._pause_position
            except Exception as e:
                error(f"Error pausing audio process: {e}")
        return self._pause_position

    def _stop_audio_process(self):
        """Stop the audio process if running"""
        if self.audio_process:
            try:
                # Check if process is still alive before attempting to stop it
                if self.audio_process.poll() is None:  # Process is still running
                    # Terminate the process group
                    os.killpg(os.getpgid(self.audio_process.pid), signal.SIGTERM)
                    # Wait a short time for graceful termination
                    try:
                        self.audio_process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate gracefully
                        os.killpg(os.getpgid(self.audio_process.pid), signal.SIGKILL)
                    debug("Audio process stopped")
                else:
                    # Process has already terminated, just clean up
                    debug("Audio process already terminated")
            except Exception as e:
                error(f"Error stopping audio process: {e}")
            finally:
                # Always reset the audio process reference
                self.audio_process = None

    def _resume_audio(self):
        """Resume audio from the pause position"""
        if self.clip and self.clip.audio:
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
            
            # Calculate start time based on current frame position
            start_time = (self.current_frame / self.video_fps) if self.video_fps > 0 else 0
            
            # For paused state, use the stored pause position
            if not was_stopped and self.paused:
                start_time = self._pause_position
                self.paused = False
            
            # Start audio if available
            if self.clip and self.clip.audio:
                try:
                    if was_stopped:
                        # If we were stopped, start from the calculated position
                        self._start_audio(start_time)
                    else:
                        # If we were paused, resume audio from the pause position
                        self._resume_audio()
                    debug("Audio playback started")
                except Exception as e:
                    error(f"Failed to start audio playback: {e}")

    def pause(self):
        """Pause playback"""
        with self._lock:
            if self.playing and not self.stopped:
                # Calculate and store the current position when pausing
                # Calculate the current position in the video
                elapsed_time = time.time() - self.last_frame_time
                frames_advanced = int(elapsed_time * self.video_fps)
                current_frame = self.current_frame + frames_advanced
                self._pause_position = current_frame / self.video_fps if self.video_fps > 0 else 0
                
                # Pause audio
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
            
            # Stop audio
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
            # When seeking, restart audio at the appropriate position
            if self.clip and self.clip.audio:
                try:
                    # Calculate the time position for seeking
                    seek_time = (frame_number / self.video_fps) if self.video_fps > 0 else 0
                    
                    # Update the pause position to the new location
                    self._pause_position = seek_time
                    
                    # If currently playing, restart audio at new position
                    if self.playing and not self.paused:
                        self._start_audio(seek_time)
                    elif self.paused:
                        # If paused, update the stored position to the new seek position
                        self._pause_position = seek_time
                except Exception as e:
                    error(f"Failed to seek audio: {e}")

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
                    self._stop_audio_process()
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
            self._pause_position = 0
            
            # Stop audio process
            self._stop_audio_process()
            
            # Close clip
            if self.clip:
                self.clip.close()