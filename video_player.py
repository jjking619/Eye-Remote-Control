import cv2
import time
import os
from PySide6.QtCore import QThread, Signal


class VideoPlayerThread(QThread):
    """Video player thread"""
    frame_ready = Signal(object)
    playback_finished = Signal()
    video_info_ready = Signal(dict)  # Emit video information
    seek_requested = Signal(int)  # New signal: request to seek to specific frame

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
        self.target_frame = -1  # New: store target frame number

    def load_video(self, file_path):
        """Load video file"""
        try:
            # If a video is already loaded, release it first
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                print(f"Cannot open video file: {file_path}")
                return False

            self.current_file = file_path
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

            # Emit video information
            self.video_info_ready.emit(video_info)

            return True
        except Exception as e:
            print(f"Failed to load video: {e}")
            return False

    def play(self):
        """Start playback"""
        self.playing = True
        self.paused = False
        self.stopped = False

    def pause(self):
        """Pause playback"""
        self.paused = True

    def stop(self):
        """Stop playback"""
        self.playing = False
        self.paused = False
        self.stopped = True
        self.current_frame = 0
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_position(self):
        """Get current playback position"""
        if self.cap and self.total_frames > 0:
            return self.current_frame / self.total_frames
        return 0

    def run(self):
        """Main playback loop"""
        while not self.exiting:
            # Handle seek requests
            if self.target_frame >= 0 and self.cap and 0 <= self.target_frame < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
                self.current_frame = self.target_frame
                self.target_frame = -1  # Reset target frame

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

                    # Control playback speed
                    sleep_time = 1.0 / self.video_fps
                    time.sleep(sleep_time)

                    # Check if playback is finished
                    if self.current_frame >= self.total_frames:
                        self.playing = False
                        self.stopped = True
                        self.current_frame = 0
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.playback_finished.emit()
                else:
                    # Playback finished
                    self.playing = False
                    self.stopped = True
                    self.current_frame = 0
                    if self.cap:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.playback_finished.emit()
            else:
                time.sleep(0.1)

        # Clean up resources
        if self.cap:
            self.cap.release()
            self.cap = None

        print("Video player thread exited")

    def shutdown(self):
        """Safely shut down thread"""
        self.exiting = True
        self.playing = False
        self.paused = False
        self.stopped = True

    def seek(self, frame_number):
        """Seek to specific frame"""
        if self.cap and 0 <= frame_number < self.total_frames:
            self.target_frame = frame_number