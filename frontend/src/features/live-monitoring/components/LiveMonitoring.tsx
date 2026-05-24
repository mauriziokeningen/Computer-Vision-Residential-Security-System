import React, { useState, useEffect, useCallback } from 'react';
import { Camera, Video, RefreshCw, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { ScrollArea } from '../../../components/ui/scroll-area';
import { Badge } from '../../../components/ui/badge';
import { ErrorBanner } from '../../../components/ui/banner';
import { StatusBadge } from '../../../components/ui/status-badges';
import { ApiCamera, ApiAlert } from '../../../types';
import { apiFetch, ensureLocalWebcam, localWebcamStreamUrl } from '../../../api/client';
import { formatTime } from '../../../lib/format';
import { useLanguage } from '../../../i18n/LanguageContext';
import { translateAlertMessage } from '../../../lib/translateAlert';

export function LiveMonitoring() {
  const { t, locale } = useLanguage();
  const [cameras, setCameras] = useState<ApiCamera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [recentAlerts, setRecentAlerts] = useState<ApiAlert[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      await ensureLocalWebcam();
      const [cameraData, alertData] = await Promise.all([
        apiFetch<ApiCamera[]>('/cameras/?limit=20'),
        apiFetch<ApiAlert[]>('/alerts/?limit=10'),
      ]);
      setCameras(cameraData);
      setRecentAlerts(alertData);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    return () => { fetch('/api/cameras/local-webcam/stop', { method: 'POST' }).catch(() => {}); };
  }, [load]);

  const activeLocalCamera = cameras.find(
    (camera) => camera.ip_address === 'local://0' && camera.status === 'ACTIVE'
  );

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Card className="xl:col-span-2">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle><Video className="h-5 w-5 inline mr-2" />{t.live.title}</CardTitle>
              <CardDescription>{t.live.titleDesc}</CardDescription>
            </div>
            <Button variant="outline" className="gap-2" onClick={load}>
              <RefreshCw className="h-4 w-4" />{t.live.refresh}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && <ErrorBanner msg={`${t.live.errorLoading}: ${error}`} onClose={() => setError('')} />}
          {loading ? (
            <div className="flex items-center gap-2 text-slate-500 text-sm p-4">
              <Loader2 className="h-4 w-4 animate-spin" />{t.live.loading}
            </div>
          ) : activeLocalCamera ? (
            <div className="aspect-video rounded-xl overflow-hidden border bg-black">
              <img
                src={localWebcamStreamUrl()}
                alt="Backend-owned local webcam stream"
                className="h-full w-full object-contain"
                onError={() => setError(t.live.errorStream)}
              />
            </div>
          ) : (
            <div className="aspect-video rounded-xl border border-dashed bg-slate-50 text-slate-500 grid place-items-center p-6 text-center">
              <div className="space-y-2">
                <Camera className="h-8 w-8 mx-auto" />
                <p className="font-medium">{t.live.noFeed}</p>
                <p className="text-sm">{t.live.noFeedDesc}</p>
              </div>
            </div>
          )}
          <div className="grid md:grid-cols-2 gap-3">
            {cameras.length === 0 ? (
              <div className="rounded-lg border p-4 text-sm text-slate-500">{t.live.noCameras}</div>
            ) : cameras.map((camera) => (
              <div key={camera.id} className="rounded-lg border bg-white p-3">
                <div className="flex items-center justify-between">
                  <div className="font-medium text-sm">{camera.location}</div>
                  <Badge className={`${camera.status === 'ACTIVE' ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-700'} rounded-full`}>
                    {camera.status}
                  </Badge>
                </div>
                <div className="text-xs text-slate-500 mt-1">{camera.ip_address}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t.live.recentTitle}</CardTitle>
          <CardDescription>{t.live.recentDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-80 pr-2">
            <div className="space-y-3">
              {recentAlerts.length === 0 ? (
                <div className="p-3 rounded-lg border bg-white text-sm text-slate-500">
                  {t.live.noAlerts}
                </div>
              ) : recentAlerts.map((alert) => (
                <div key={alert.id} className="p-3 rounded-lg border bg-white">
                  <div className="flex items-center justify-between gap-2">
                    <StatusBadge status={alert.status} />
                    <span className="text-xs text-slate-500">{formatTime(alert.created_at)}</span>
                  </div>
                  <div className="text-sm mt-2 text-slate-700">
                    {translateAlertMessage(alert.message, locale)}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}