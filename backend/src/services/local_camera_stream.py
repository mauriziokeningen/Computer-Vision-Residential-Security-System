import cv2
import time
import threading
from typing import Optional


class LocalCameraStreamService:
    """
    Backend-owned local webcam service.
    Keeps a single shared capture session and serves the latest JPEG frame.
    """

    def __init__(self):
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._starting = False
        self._source_index = 0
        self._last_frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._first_frame_event = threading.Event()

    def _open_capture(self, source_index: int) -> cv2.VideoCapture:
        cap = None

        # Try DirectShow first on Windows
        if hasattr(cv2, "CAP_DSHOW"):
            try:
                cap = cv2.VideoCapture(source_index, cv2.CAP_DSHOW)
            except Exception:
                cap = None

        # Fallback to default backend
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(source_index)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open local webcam source {source_index}. "
                "Make sure no other application is locking the camera."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def start(self, source_index: int = 0, warmup_timeout: float = 4.0) -> None:
        """
        Start the webcam only once.
        Reuses the same backend-owned capture across all MJPEG clients.
        """
        with self._lock:
            if self._running and self._source_index == source_index:
                return
            if self._starting and self._source_index == source_index:
                return

        with self._start_lock:
            with self._lock:
                if self._running and self._source_index == source_index:
                    return
                if self._starting and self._source_index == source_index:
                    return
                self._starting = True
                self._source_index = source_index

            self.stop()

            cap = self._open_capture(source_index)

            with self._lock:
                self._capture = cap
                self._running = True
                self._last_frame = None
                self._first_frame_event.clear()
                self._thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._thread.start()

            got_first_frame = self._first_frame_event.wait(timeout=warmup_timeout)

            with self._lock:
                self._starting = False

            if not got_first_frame:
                self.stop()
                raise RuntimeError(
                    "The backend opened the webcam but could not read frames from it. "
                    "Close any app that may be using the camera and try again."
                )

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._starting = False

            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None

            self._thread = None
            self._last_frame = None
            self._first_frame_event.clear()

    def _reader_loop(self) -> None:
        consecutive_failures = 0

        while True:
            with self._lock:
                running = self._running
                capture = self._capture

            if not running:
                break

            if capture is None:
                time.sleep(0.05)
                continue

            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 20:
                    with self._lock:
                        self._last_frame = None
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            success, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            if not success:
                time.sleep(0.01)
                continue

            with self._lock:
                self._last_frame = jpeg.tobytes()
                self._first_frame_event.set()

            time.sleep(0.03)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "starting": self._starting,
                "source_index": self._source_index,
                "has_frame": self._last_frame is not None,
            }

    def mjpeg_generator(self):
        try:
            while True:
                with self._lock:
                    frame = self._last_frame
                    running = self._running

                if not running:
                    break

                if frame is None:
                    time.sleep(0.05)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
                )

                time.sleep(0.05)
        except GeneratorExit:
            return


local_camera_stream_service = LocalCameraStreamService()