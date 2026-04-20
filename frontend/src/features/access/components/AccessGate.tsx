import React, { useState, useEffect, useCallback } from 'react';
import {
  Camera,
  ShieldCheck,
  CircleX,
  Loader2,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { Badge } from '../../../components/ui/badge';
import { Progress } from '../../../components/ui/progress';
import { ErrorBanner } from '../../../components/ui/banner';

import { ApiCamera } from '../../../types';
import { apiFetch, ensureLocalWebcam, localWebcamStreamUrl } from '../../../api/client';

export function AccessGate({ onRegisterVisitor }: { onRegisterVisitor: () => void }) {
  const [cameras, setCameras] = useState<ApiCamera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');

    try {
      await ensureLocalWebcam();
      const cameraData = await apiFetch<ApiCamera[]>('/cameras/?limit=20');
      setCameras(cameraData);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    return () => {
      fetch('/api/cameras/local-webcam/stop', { method: 'POST' }).catch(() => {});
    };
  }, [load]);

  const activeLocalCamera = cameras.find(
    (camera) => camera.ip_address === 'local://0' && camera.status === 'ACTIVE'
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Acceso – Torre A</CardTitle>
          <CardDescription>Verification view fed by backend-owned cameras</CardDescription>
        </CardHeader>

        <CardContent className="space-y-3">
          {error && (
            <ErrorBanner msg={`Error loading cameras: ${error}`} onClose={() => setError('')} />
          )}

          {loading ? (
            <div className="aspect-video rounded-xl border bg-slate-50 grid place-items-center text-slate-500">
              <div className="flex items-center gap-2 text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading access camera…
              </div>
            </div>
          ) : activeLocalCamera ? (
            <div className="aspect-video rounded-xl overflow-hidden border bg-black">
              <img
                src={localWebcamStreamUrl()}
                alt="Backend-owned access stream"
                className="h-full w-full object-contain"
                onError={() =>
                  setError(
                    'Could not load backend webcam stream. Make sure no other app is locking the laptop camera.'
                  )
                }
              />
            </div>
          ) : (
            <div className="aspect-video rounded-xl border border-dashed bg-slate-50 text-slate-500 grid place-items-center p-6 text-center">
              <div className="space-y-2">
                <Camera className="h-8 w-8 mx-auto" />
                <p className="font-medium">No access control camera is active.</p>
                <p className="text-sm">
                  The backend could not expose the local webcam yet. Once device 0 is available,
                  the processed stream will render here.
                </p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-3 gap-2 text-xs">
            {['Iluminación', 'Enfoque', 'Alineación'].map((k, i) => (
              <div key={k} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span>{k}</span>
                  <span>{70 + i * 10}%</span>
                </div>
                <Progress value={70 + i * 10} />
              </div>
            ))}
          </div>
        </CardContent>

        <CardFooter className="flex gap-2">
          <Button variant="default" className="gap-2">
            <ShieldCheck className="h-4 w-4" />
            Permitir
          </Button>
          <Button variant="destructive" className="gap-2">
            <CircleX className="h-4 w-4" />
            Denegar
          </Button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Result</CardTitle>
          <CardDescription>Current access decision and fallback actions</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex justify-between"><span>Nombre</span><span className="font-medium">—</span></div>
          <div className="flex justify-between"><span>Similitud</span><span className="font-medium">0.41</span></div>
          <div className="flex justify-between"><span>Umbral</span><span>0.52</span></div>
          <div className="flex justify-between">
            <span>Veredicto</span>
            <Badge variant="outline" className="rounded-full">Desconocido</Badge>
          </div>
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button variant="outline" onClick={onRegisterVisitor}>Registrar visitante</Button>
          <Button variant="outline">Crear incidente</Button>
        </CardFooter>
      </Card>
    </div>
  );
}

