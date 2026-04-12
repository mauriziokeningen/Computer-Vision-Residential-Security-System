import React, { useEffect, useRef, useState } from "react";

type CameraDevice = {
  deviceId: string;
  label: string;
};

export default function WebcamFeed({
  className,
  onFrame,
  fps = 8,
}: {
  className?: string;
  onFrame?: (canvas: HTMLCanvasElement, width: number, height: number) => void;
  fps?: number;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");

  useEffect(() => {
    let mounted = true;

    async function loadDevices() {
      try {
        const tempStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        tempStream.getTracks().forEach((t) => t.stop());

        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const cameras = allDevices
          .filter((d) => d.kind === "videoinput")
          .map((d) => ({
            deviceId: d.deviceId,
            label: d.label || `Cámara ${d.deviceId.slice(0, 6)}`,
          }));

        if (!mounted) return;

        setDevices(cameras);

        if (cameras.length > 0) {
          const externalCam =
            cameras.find((c) =>
              c.label.toLowerCase().includes("usb") ||
              c.label.toLowerCase().includes("hd") ||
              c.label.toLowerCase().includes("webcam")
            ) ?? cameras[cameras.length - 1];

          setSelectedDeviceId(externalCam.deviceId);
        }
      } catch (err) {
        console.error("Error enumerando cámaras:", err);
      }
    }

    loadDevices();

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    let stream: MediaStream | null = null;

    async function startCamera() {
      if (!selectedDeviceId) return;

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: selectedDeviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        console.error("Error iniciando cámara:", err);
      }
    }

    startCamera();

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [selectedDeviceId]);

  useEffect(() => {
    if (!onFrame) return;

    const id = setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      const w = 640;
      const h = Math.round((video.videoHeight / video.videoWidth) * w);
      if (!Number.isFinite(h) || h <= 0) return;

      canvas.width = w;
      canvas.height = h;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, w, h);
      onFrame(canvas, w, h);
    }, Math.max(80, Math.round(1000 / fps)));

    return () => clearInterval(id);
  }, [onFrame, fps]);

  return (
    <div className={`${className} relative`}>
      {devices.length > 1 && (
        <div className="absolute top-2 right-2 z-20">
          <select
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            className="bg-white/90 text-black text-xs rounded px-2 py-1 border"
          >
            {devices.map((cam) => (
              <option key={cam.deviceId} value={cam.deviceId}>
                {cam.label}
              </option>
            ))}
          </select>
        </div>
      )}

      <video
        ref={videoRef}
        className="h-full w-full object-contain -scale-x-100 bg-black"
        playsInline
        muted
      />
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}