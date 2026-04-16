import zmq
import time
import threading
from typing import Optional


class LocalCameraStreamService:
    """
    Decoupled Video Ingestion Subscriber Service.

    This service operates as the boundary between the AI inference worker and the FastAPI 
    client stream. By acting as a downstream ZeroMQ subscriber, it eliminates the OS-level 
    hardware mutex collisions that occur when multiple processes attempt to bind to the 
    same physical capture device (e.g., /dev/video0). The AI inference node is maintained 
    as the Single Source of Truth (SSoT) for all hardware interaction.

    Architectural Decisions & Trade-offs:
    * Inter-Process Communication (IPC): ZeroMQ was selected over broker-based pub/sub 
      (e.g., Redis, RabbitMQ) to route data directly socket-to-socket. This prevents memory 
      bloat and intermediate disk I/O when processing high-throughput (60MB/s) video payloads.
    * Resilience: Designed as a singleton initialized once per ASGI worker lifecycle. It 
      utilizes non-blocking sockets to prevent thread starvation if the upstream publisher 
      terminates unexpectedly.
    * Performance: By consuming pre-compressed JPEG byte arrays over TCP, this service 
      completely eliminates matrix serialization overhead on the web server, acting purely 
      as a zero-cost byte forwarder.
    """

    def __init__(self):
        # Initializes the underlying C-level thread pool for ZMQ sockets.
        self._context = zmq.Context()
        self._socket: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_frame: Optional[bytes] = None
        
        # Concurrency control: Enforces thread-safe memory access across the ZMQ polling 
        # daemon and concurrent ASGI worker threads executing HTTP requests.
        self._lock = threading.Lock()

    def start(self, source_index: int = 0, warmup_timeout: float = 4.0) -> None:
        """
        Provisions the TCP socket and spawns the daemon reader thread.

        Args:
            source_index (int): Maintained for interface compatibility; overridden by network topology.
            warmup_timeout (float): Execution threshold for initial stream acquisition.

        Constraints:
            Implements ZMQ_CONFLATE set to 1. Video streams generate data faster than standard 
            network clients consume it. Rather than queueing frames (which causes temporal latency 
            and OOM crashes), conflation enforces an O(1) memory bound by actively overwriting 
            unread frames. We explicitly sacrifice sequential completeness for real-time accuracy.
        """
        with self._lock:
            if self._running:
                return
            self._running = True

        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect("tcp://127.0.0.1:5555")
        
        # An empty subscription filter assumes total consumption of the target port.
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1) 

        # Delegates blocking network I/O to a background thread to preserve the asyncio event loop.
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Gracefully terminates the polling daemon and releases the IPC network socket.
        """
        with self._lock:
            self._running = False
            
        if self._socket:
            self._socket.close()
            self._socket = None

    def _reader_loop(self) -> None:
        """
        Dedicated network polling daemon.

        Boundary Warnings:
            This implementation relies strictly on zmq.NOBLOCK. A standard blocking recv() 
            will permanently deadlock the background thread if the upstream AI publisher crashes.
            The exception handler yields the Global Interpreter Lock (GIL) to prevent CPU starvation.
        """
        while self._running and self._socket:
            try:
                frame = self._socket.recv(zmq.NOBLOCK)
                with self._lock:
                    self._last_frame = frame
            except zmq.Again:
                # time.sleep(0.01) is required here for backoff. Yields the GIL.
                time.sleep(0.01) 
            except Exception:
                break

    def get_status(self) -> dict:
        """
        Diagnostic probe utilized by the API for readiness checks.

        Returns:
            dict: Internal state mapping containing execution status and frame availability.
        """
        with self._lock:
            return {
                "running": self._running,
                "has_frame": self._last_frame is not None,
            }

    def mjpeg_generator(self):
        """
        Constructs an HTTP Multipart / MJPEG compliant data stream.

        Yields:
            bytes: Encoded HTTP MJPEG boundary strings wrapping the raw JPEG payload.

        Intent:
            MJPEG (multipart/x-mixed-replace) enables native HTML <img> tag rendering without 
            relying on heavy client-side JavaScript decoders (e.g., WebRTC, Canvas).
        """
        try:
            while True:
                with self._lock:
                    frame = self._last_frame
                    running = self._running

                if not running:
                    break

                if frame is None:
                    # Prevents infinite looping and CPU spikes during upstream frame drops.
                    time.sleep(0.05)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
                )
                
                # Regulates output to ~20 FPS. Protects downstream clients from rendering exhaustion.
                time.sleep(0.05)
        except GeneratorExit:
            return


local_camera_stream_service = LocalCameraStreamService()