import cv2
import time
import os
import threading
import pygame
from PySide6.QtCore import QThread, Signal
from log import debug, error 

class VideoPlayerThread(QThread):
    """Video player thread"""
    frame_ready = Signal(object)
    playback_finished = Signal()
    video_info_ready = Signal(dict)  # Emit video information
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
        # Add exit flag
        self.exiting = False
        self.target_frame = -1
        self._closed = True  # Track whether resources have been released
        self._lock = threading.RLock()  # Reentrant lock for resource protection
        
        # Audio support
        self.audio_initialized = False
        try:
            pygame.mixer.init()
            self.pygame = pygame
            self.audio_initialized = True
        except ImportError:
            self.pygame = None
            error("Pygame not installed. Audio will not be available.")
        except Exception as e:
            self.pygame = None
            error(f"Failed to initialize audio: {e}")

    def load_video(self, file_path):
        """Load video file"""
        try:
            debug(f"Attempting to load video: {file_path}")
            
            # Always release existing video capture before loading a new one
            with self._lock:
                if self.cap is not None and not self._closed:
                    try:
                        self.cap.release()
                    except Exception as e:
                        error(f"Error releasing video capture: {e}")
                    finally:
                        self.cap = None
                        self._closed = True
            
            # Stop any currently playing audio
            if self.audio_initialized and self.pygame:
                try:
                    self.pygame.mixer.music.stop()
                except:
                    pass
            
            with self._lock:
                # Create new capture
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    error(f"Cannot open video file: {file_path}")
                    self.cap = None
                    self._closed = True
                    return False

                self.current_file = file_path
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._closed = False  # Mark resources as active

            # Ensure frame rate is valid
            if self.video_fps <= 0:
                self.video_fps = 30  # Default value

            # Calculate video duration
            if self.video_fps > 0 and self.total_frames > 0:
                self.video_duration = self.total_frames / self.video_fps
            else:
                self.video_duration = 0

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

            # Load audio track if available
            if self.audio_initialized and self.pygame:
                try:
                    self.pygame.mixer.music.load(file_path)
                except Exception as e:
                    error(f"Failed to load audio track: {e}")

            # Emit video information
            self.video_info_ready.emit(video_info)
            debug(f"Successfully loaded video: {file_path}, total frames: {self.total_frames}")

            return True
        except Exception as e:
            error(f"Failed to load video: {e}")
            with self._lock:
                self._safe_release_capture()
            return False

    def play(self):
        """Start playback"""
        with self._lock:
            self.playing = True
            self.paused = False
            self.stopped = False

        # Start audio if available
        if self.audio_initialized and self.pygame:
            try:
                # Calculate position to resume audio
                position = self.get_position()
                if position > 0:
                    # For simplicity, we restart the audio from the beginning
                    # More advanced implementations could use sound positioning libraries
                    self.pygame.mixer.music.play(start=position * self.video_duration)
                else:
                    self.pygame.mixer.music.play()
            except Exception as e:
                error(f"Failed to play audio: {e}")

    def pause(self):
        """Pause playback"""
        debug(f"Pausing playback for video: {self.current_file}")
        with self._lock:
            self.paused = True

        # Pause audio if available
        if self.audio_initialized and self.pygame:
            try:
                self.pygame.mixer.music.pause()
            except Exception as e:
                error(f"Failed to pause audio: {e}")

    def stop(self):
        """Stop playback"""
        debug(f"Stopping playback for video: {self.current_file}")
        with self._lock:
            self.playing = False
            self.paused = False
            self.stopped = True
            self.current_frame = 0
            if self.cap and not self._closed:
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception as e:
                    error(f"Error resetting video position: {e}")

        # Stop audio if available
        if self.audio_initialized and self.pygame:
            try:
                self.pygame.mixer.music.stop()
            except Exception as e:
                error(f"Failed to stop audio: {e}")

    def get_position(self):
        """Get current playback position"""
        with self._lock:
            if self.cap and not self._closed and self.total_frames > 0:
                return self.current_frame / self.total_frames
        return 0

    def _safe_release_capture(self):
        """Safely release video capture resources with multiple safety checks"""
        try:
            with self._lock:
                if self.cap is not None:
                    try:
                        if not self._closed:
                            debug("Releasing video capture")
                            self.cap.release()
                    except Exception as e:
                        error(f"Error releasing video capture: {e}")
                    finally:
                        self.cap = None
                        self._closed = True
                else:
                    # Even if cap is None, mark as closed
                    self._closed = True
        except Exception as e:
            error(f"Error in _safe_release_capture: {e}")
        finally:
            with self._lock:
                self.cap = None
                self._closed = True

    def run(self):
        """Main playback loop"""
        while not self.exiting:
            # Handle seek requests
            target_frame_handled = False
            with self._lock:
                if (self.target_frame >= 0 and self.cap is not None and not self._closed and 
                    0 <= self.target_frame < self.total_frames):
                    try:
                        if self.cap is not None:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
                            self.current_frame = self.target_frame
                            target_frame_handled = True
                            
                            # Seek audio if available
                            if self.audio_initialized and self.pygame:
                                try:
                                    position = self.target_frame / self.total_frames
                                    # Restart audio at new position
                                    self.pygame.mixer.music.stop()
                                    if self.playing and not self.paused:
                                        self.pygame.mixer.music.play(start=position * self.video_duration)
                                except Exception as e:
                                    error(f"Failed to seek audio: {e}")
                    except Exception as e:
                        error(f"Error seeking to frame {self.target_frame}: {e}")
            
            if target_frame_handled:
                with self._lock:
                    self.target_frame = -1  # Reset target frame

            # Check various states
            stopped_state = False
            paused_state = False
            playing_state = False
            with self._lock:
                stopped_state = self.stopped
                paused_state = self.paused
                playing_state = self.playing

            if stopped_state:
                time.sleep(0.1)
                continue

            if not playing_state or paused_state:
                debug("Playback paused, waiting...")
                time.sleep(0.1)
                continue

            # Check if capture is available
            cap_available = False
            with self._lock:
                cap_available = self.cap is not None and not self._closed
                if cap_available:
                    try:
                        cap_available = self.cap.isOpened()
                    except:
                        cap_available = False
                        
            if cap_available:
                try:
                    ret, frame = None, None
                    cap_valid = False
                    with self._lock:
                        cap_valid = self.cap is not None and not self._closed
                    
                    if cap_valid:
                        try:
                            ret, frame = self.cap.read()
                        except Exception as e:
                            error(f"Error reading frame: {e}")
                            ret = False
                            
                    if ret and frame is not None:
                        with self._lock:
                            self.current_frame += 1
                            
                        # Emit frame without holding the lock
                        self.frame_ready.emit(frame)

                        # Control playback speed
                        sleep_time = 1.0 / self.video_fps if self.video_fps > 0 else 0.033
                        time.sleep(sleep_time)

                        # Check if playback is finished
                        finished = False
                        total_frames = 0
                        with self._lock:
                            total_frames = self.total_frames
                            if self.current_frame >= total_frames:
                                finished = True
                                
                        if finished and total_frames > 0:
                            debug("Playback finished")
                            with self._lock:
                                self.playing = False
                                self.stopped = True
                                self.current_frame = 0
                                if self.cap is not None and not self._closed:
                                    try:
                                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                    except Exception as e:
                                        error(f"Error resetting video position: {e}")
                            self.playback_finished.emit()
                            
                            # Stop audio if available
                            if self.audio_initialized and self.pygame:
                                try:
                                    self.pygame.mixer.music.stop()
                                except Exception as e:
                                    error(f"Failed to stop audio: {e}")
                    else:
                        # Playback finished or error occurred
                        debug("Playback finished (no more frames or error)")
                        with self._lock:
                            self.playing = False
                            self.stopped = True
                            self.current_frame = 0
                            
                            if self.cap is not None and not self._closed:
                                try:
                                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                except Exception as e:
                                    error(f"Error resetting video position: {e}")
                        self.playback_finished.emit()
                        
                        # Stop audio if available
                        if self.audio_initialized and self.pygame:
                            try:
                                self.pygame.mixer.music.stop()
                            except Exception as e:
                                error(f"Failed to stop audio: {e}")
                except Exception as e:
                    error(f"Error during video playback: {e}")
                    with self._lock:
                        self.playing = False
                        self.stopped = True
                    self.playback_finished.emit()
                    
                    # Stop audio if available
                    if self.audio_initialized and self.pygame:
                        try:
                            self.pygame.mixer.music.stop()
                        except Exception as e:
                            error(f"Failed to stop audio: {e}")
            else:
                debug("Video capture not opened or failed")
                time.sleep(0.1)

        # Clean up resources when exiting
        self._safe_release_capture()
        
        # Stop audio if available
        if self.audio_initialized and self.pygame:
            try:
                self.pygame.mixer.music.stop()
                self.pygame.mixer.quit()
            except Exception as e:
                error(f"Failed to quit audio mixer: {e}")
        
        debug("Video player thread exited")

    def shutdown(self):
        """Safely shut down thread"""
        debug("Shutting down video player thread")
        with self._lock:
            self.exiting = True
            self.playing = False
            self.paused = False
            self.stopped = True

    def seek(self, frame_number):
        """Seek to specific frame"""
        debug(f"Requesting seek to frame: {frame_number}")