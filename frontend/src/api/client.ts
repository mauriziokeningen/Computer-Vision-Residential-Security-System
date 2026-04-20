import { ApiCamera } from '../types';

const API = '/api';

export async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const isFormData = opts?.body instanceof FormData;
  const res = await fetch(`${API}${path}`, {
    headers: isFormData ? undefined : { 'Content-Type': 'application/json' },
    ...opts,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {}
    throw new Error(detail);
  }

  return res.json();
}

export async function ensureLocalWebcam(): Promise<ApiCamera> {
  return apiFetch<ApiCamera>('/cameras/local-webcam/ensure', {
    method: 'POST',
  });
}

export function localWebcamStreamUrl() {
  // Appending the current epoch timestamp forces the browser to bypass its
  // internal cache and initialize a new HTTP boundary stream upon every component mount.
  return `/api/cameras/local-webcam/stream?source=0&t=${Date.now()}`;
}

export function buildWsUrl(path: string) {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';

  // import.meta.env.DEV is natively injected by Vite.
  // This guarantees we hit the FastAPI backend port locally, whether you
  // access the app via 'localhost', '127.0.0.1', or '192.168.x.x'.
  if (import.meta.env.DEV) {
    return `${wsProtocol}://${window.location.hostname}:8000${path}`;
  }

  // In production, rely on standard unified ingress routing.
  return `${wsProtocol}://${window.location.host}${path}`;
}

