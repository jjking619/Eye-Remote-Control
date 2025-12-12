import cv2
import time
from PySide6.QtCore import QThread, Signal
from eye_detector import MediaPipeEyeDetector


class VideoCaptureThread(QThread):
    frame_ready = Signal(object)
    detection_status = Signal(dict)  # Emit detection status
    fps_updated = Signal(float)  # Emit FPS updates
    command_detected = Signal(str)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.detecting = True
        self.show_landmarks = True

        # Add exit flag
        self.exiting = False

        # Component initialization
        self.eye_detector = MediaPipeEyeDetector()

        # FPS calculation
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_command = None
        self.last_face_detected_time = time.time()

    def find_available_camera(self):
        """Automatically detect available camera"""
        # First try the default cameras (0-9)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    return i
            cap.release()
            
        # If no default cameras found, try higher indexes (10-20)
        # Some systems may have cameras with higher indexes
        for i in range(10, 21):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    return i
            cap.release()
            
        # Try platform-specific camera paths for Linux systems
        # These paths are common for USB and integrated cameras on Linux
        linux_camera_paths = [
            "/dev/video0", "/dev/video1", "/dev/video2", 
            "/dev/video3", "/dev/video4", "/dev/video5"
        ]
        
        for path in linux_camera_paths:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    # Return the index part of the path
                    return int(path.replace("/dev/video", ""))
            cap.release()
            
        return None

    def start_capture(self, camera_id=None):
        if camera_id is None:
            camera_id = self.find_available_camera()
            if camera_id is None:
                raise Exception("No available camera device found")

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
        """Improved stop method"""
        self.running = False
        self.exiting = True

        # Wait for thread to finish, but set timeout
        if self.isRunning():
            self.wait(2000)  # Wait up to 2 seconds

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
                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:  # Update once per second
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                    self.fps_updated.emit(self.fps)

                processed_frame = frame.copy()
                detection_result = {}

                # Process frame if detection is enabled
                if self.detecting:
                    try:
                        # Detect eye state
                        detection_result = self.eye_detector.detect_eyes_state(processed_frame)

                        # Emit detection status
                        self.detection_status.emit(detection_result)

                        # Handle actions according to new control logic
                        # When playing video, continue playing if eyes are gazing at screen, 
                        command = None
                        face_detected = detection_result.get('face_detected', False)

                        if face_detected:
                            # Update last face detected time
                            self.last_face_detected_time = current_time

                            # Check if eyes are closed
                            eyes_closed = detection_result.get('eyes_closed', False)

                            # Check if user is gazing
                            is_gazing = detection_result.get('is_gazing', False)

                            # Pause if eyes are closed or not gazing
                            # The eye detector already handles short blinks appropriately
                            if eyes_closed or not is_gazing:
                                command = "pause"
                            else:
                                command = "play"
                        else:
                            # Pause video if no face detected for over 1 second
                            if current_time - self.last_face_detected_time > 1.0:
                                command = "pause"

                        # Draw landmarks (optional)
                        if self.show_landmarks and face_detected:
                            self.eye_detector.draw_landmarks(processed_frame, detection_result)

                        # Emit command signal
                        if command and command != self.last_command:
                            self.command_detected.emit(command)
                            self.last_command = command

                    except Exception as e:
                        print(f"Detection error: {e}")
                        # Emit empty status to indicate detection failure
                        self.detection_status.emit({})
                else:
                    # If detection is disabled, emit empty status
                    self.detection_status.emit({})

                # Emit frame ready signal
                self.frame_ready.emit(processed_frame)

                time.sleep(0.03)  # ~30 FPS

        # Clean up resources
        if self.cap:
            self.cap.release()
            self.cap = None

        # Release MediaPipe resources
        try:
            self.eye_detector.close()
        except:
            pass

        print("Camera thread exited")
        self.finished.emit()
