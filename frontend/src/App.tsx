import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  Bell,
  Camera,
  ChevronRight,
  DoorOpen,
  Gauge,
  ListChecks,
  PersonStanding,
  Settings,
  ShieldCheck,
  Siren,
  UserPlus,
  Users,
  Video,
  Wand2,
  Search,
  CheckCircle2,
  CircleX,
  Clock,
  Cpu,
  Database,
  RefreshCw,
  Loader2,
  XCircle,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { ScrollArea } from './components/ui/scroll-area';
import { Switch } from './components/ui/switch';
import { Badge } from './components/ui/badge';
import { Select, NativeSelect, SelectItem } from './components/ui/select';
import { Progress } from './components/ui/progress';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

type AlertStatus = 'UNREAD' | 'ACKNOWLEDGED' | 'RESOLVED';

interface ApiAlert {
  id: string;
  incident_id: string | null;
  message: string;
  status: AlertStatus;
  created_at: string;
  resolved_at: string | null;
}

interface ApiIncident {
  id: string;
  created_at: string;
  incident_metadata: {
    rule_triggered?: string;
    priority?: string;
    module?: string;
    camera_id?: string;
    timestamp?: string;
    detections?: any[];
  };
}

interface ApiPerson {
  id: string;
  full_name: string;
  person_type: string;
  building?: string | null;
  apartment?: string | null;
  phone?: string | null;
  email?: string | null;
  valid_from?: string | null;
  valid_until?: string | null;
  created_at: string;
}

interface ApiCamera {
  id: string;
  location: string;
  ip_address: string;
  status: string;
}

interface EvidenceFile {
  object_name: string;
  size: number;
  last_modified?: string | null;
  content_type?: string | null;
}

interface AlertCounts {
  unread: number;
  acknowledged: number;
  resolved: number;
}

const API = '/api';

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
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

async function ensureLocalWebcam(): Promise<ApiCamera> {
  return apiFetch<ApiCamera>('/cameras/local-webcam/ensure', {
    method: 'POST',
  });
}

function localWebcamStreamUrl() {
  return '/api/cameras/local-webcam/stream?source=0';
}

function buildWsUrl(path: string) {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${wsProtocol}://${window.location.host}${path}`;
}

function formatTime(iso: string) {
  const date = new Date(iso);
  return date.toLocaleString('es-MX', {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: '2-digit',
  });
}

function priorityToSeverity(p?: string): 'low' | 'high' | 'critical' {
  if (p === 'CRITICAL') return 'critical';
  if (p === 'HIGH' || p === 'MEDIUM') return 'high';
  return 'low';
}

function ErrorBanner({ msg, onClose }: { msg: string; onClose: () => void }) {
  return (
    <div className="flex items-center gap-3 bg-red-50 border border-red-200 text-red-800 rounded-lg px-4 py-3 text-sm">
      <XCircle className="h-4 w-4 shrink-0" />
      <span className="flex-1">{msg}</span>
      <button onClick={onClose} className="text-red-400 hover:text-red-600">✕</button>
    </div>
  );
}

function SuccessBanner({ msg, onClose }: { msg: string; onClose: () => void }) {
  return (
    <div className="flex items-center gap-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg px-4 py-3 text-sm">
      <CheckCircle2 className="h-4 w-4 shrink-0" />
      <span className="flex-1">{msg}</span>
      <button onClick={onClose} className="text-emerald-400 hover:text-emerald-600">✕</button>
    </div>
  );
}

function SeverityBadge({ level }: { level: 'low' | 'high' | 'critical' }) {
  const map: Record<string, string> = {
    low: 'bg-slate-100 text-slate-800',
    high: 'bg-amber-100 text-amber-800',
    critical: 'bg-red-100 text-red-800',
  };
  const label: Record<string, string> = {
    low: 'Baja',
    high: 'Alta',
    critical: 'Crítica',
  };
  return <Badge className={`${map[level]} rounded-full`}>{label[level]}</Badge>;
}

function StatusBadge({ status }: { status: AlertStatus }) {
  const map: Record<AlertStatus, string> = {
    UNREAD: 'bg-red-100 text-red-800',
    ACKNOWLEDGED: 'bg-amber-100 text-amber-800',
    RESOLVED: 'bg-emerald-100 text-emerald-800',
  };
  const label: Record<AlertStatus, string> = {
    UNREAD: 'Sin leer',
    ACKNOWLEDGED: 'Atendida',
    RESOLVED: 'Resuelta',
  };
  return <Badge className={`${map[status]} rounded-full`}>{label[status]}</Badge>;
}

function KPI({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: any;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-slate-100">
            <Icon className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <div className="text-xs text-slate-500">{label}</div>
            <div className="text-xl font-semibold">{value}</div>
            <div className="text-xs text-slate-500">{sub}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function Header({
  unreadCount,
  searchQuery,
  onSearchChange,
  onOpenAlerts,
}: {
  unreadCount: number;
  searchQuery: string;
  onSearchChange: (value: string) => void;
  onOpenAlerts: () => void;
}) {
  return (
    <div className="flex flex-col gap-3 py-4 md:flex-row md:items-center md:justify-between">
      <div className="flex items-center gap-3">
        <Siren className="h-7 w-7" />
        <div>
          <h1 className="text-xl font-semibold leading-none">Seguridad UH · Panel</h1>
          <p className="text-sm text-slate-500">Monitoreo en tiempo real · TT2</p>
        </div>
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="relative min-w-[260px]">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <Input
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Buscar alertas, incidentes o personas"
            className="pl-9"
          />
        </div>
        <Button onClick={onOpenAlerts} className="gap-2 relative">
          <Bell className="h-4 w-4" />Notificaciones
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1 bg-red-500 text-white text-[10px] rounded-full h-4 w-4 flex items-center justify-center">
              {unreadCount > 9 ? '9+' : unreadCount}
            </span>
          )}
        </Button>
      </div>
    </div>
  );
}

function Dashboard({ alertCounts }: { alertCounts: AlertCounts }) {
  const [recentAlerts, setRecentAlerts] = useState<ApiAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [statsSeries, setStatsSeries] = useState<{ t: string; a: number }[]>([]);

  useEffect(() => {
    apiFetch<ApiAlert[]>('/alerts/?limit=20')
      .then((data) => {
        setRecentAlerts(data);
        const now = new Date();
        const buckets: Record<string, number> = {};
        for (let h = 7; h >= 0; h--) {
          const d = new Date(now);
          d.setHours(now.getHours() - h);
          buckets[String(d.getHours()).padStart(2, '0')] = 0;
        }
        data.forEach((a) => {
          const h = String(new Date(a.created_at).getHours()).padStart(2, '0');
          if (h in buckets) buckets[h]++;
        });
        setStatsSeries(Object.entries(buckets).map(([t, a]) => ({ t, a })));
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const total = alertCounts.unread + alertCounts.acknowledged + alertCounts.resolved;

  return (
    <div className="space-y-4">
      {error && <ErrorBanner msg={`Error cargando alertas: ${error}`} onClose={() => setError('')} />}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPI icon={AlertTriangle} label="Sin leer" value={String(alertCounts.unread)} sub="Requieren atención" />
        <KPI icon={ShieldCheck} label="Atendidas" value={String(alertCounts.acknowledged)} sub="En seguimiento" />
        <KPI icon={CircleX} label="Resueltas" value={String(alertCounts.resolved)} sub="Cerradas" />
        <KPI icon={Cpu} label="Total" value={String(total)} sub="Todas las alertas" />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle>Tendencia de alertas (hoy)</CardTitle>
            <CardDescription>Frecuencia por hora</CardDescription>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={statsSeries}>
                <XAxis dataKey="t" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Line type="monotone" dataKey="a" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Estado de alertas</CardTitle>
            <CardDescription>Distribución actual</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {[
              { label: 'Sin leer', value: alertCounts.unread, color: 'bg-red-500' },
              { label: 'Atendidas', value: alertCounts.acknowledged, color: 'bg-amber-500' },
              { label: 'Resueltas', value: alertCounts.resolved, color: 'bg-emerald-500' },
            ].map((s) => (
              <div key={s.label} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <div className={`h-2 w-2 rounded-full ${s.color}`} />
                    <span>{s.label}</span>
                  </div>
                  <span className="font-medium">{s.value}</span>
                </div>
                <Progress value={total > 0 ? (s.value / total) * 100 : 0} />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Últimas alertas</CardTitle>
          <CardDescription>Datos reales desde la API</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center gap-2 text-slate-500 text-sm p-4">
              <Loader2 className="h-4 w-4 animate-spin" />Cargando alertas…
            </div>
          ) : recentAlerts.length === 0 ? (
            <div className="text-slate-500 text-sm p-4">
              No hay alertas. Simula un evento en la sección Incidentes.
            </div>
          ) : (
            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
              {recentAlerts.slice(0, 6).map((a) => (
                <motion.div
                  key={a.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <Card
                    className="border-l-4"
                    style={{
                      borderLeftColor:
                        a.status === 'UNREAD'
                          ? '#ef4444'
                          : a.status === 'ACKNOWLEDGED'
                          ? '#f59e0b'
                          : '#10b981',
                    }}
                  >
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <StatusBadge status={a.status} />
                        <CardDescription className="text-xs flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatTime(a.created_at)}
                        </CardDescription>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0 text-xs text-slate-600">
                      {a.message.length > 60 ? a.message.slice(0, 60) + '…' : a.message}
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function LiveMonitoring() {
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
    return () => {
      fetch('/api/cameras/local-webcam/stop', { method: 'POST' }).catch(() => {});
    };
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
              <CardTitle>
                <Video className="h-5 w-5 inline mr-2" />
                Live monitoring
              </CardTitle>
              <CardDescription>
                Backend-owned ingestion only. The browser does not capture hardware directly.
              </CardDescription>
            </div>
            <Button variant="outline" className="gap-2" onClick={load}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {error && (
            <ErrorBanner
              msg={`Error loading live monitoring data: ${error}`}
              onClose={() => setError('')}
            />
          )}

          {loading ? (
            <div className="flex items-center gap-2 text-slate-500 text-sm p-4">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading camera status…
            </div>
          ) : activeLocalCamera ? (
            <div className="aspect-video rounded-xl overflow-hidden border bg-black">
              <img
                src={localWebcamStreamUrl()}
                alt="Backend-owned local webcam stream"
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
                <p className="font-medium">No active backend-owned video feed available.</p>
                <p className="text-sm">
                  The frontend no longer opens the laptop camera directly. A backend-owned local webcam
                  feed will appear here once the backend can access device 0.
                </p>
              </div>
            </div>
          )}

          <div className="grid md:grid-cols-2 gap-3">
            {cameras.length === 0 ? (
              <div className="rounded-lg border p-4 text-sm text-slate-500">
                No cameras are registered yet.
              </div>
            ) : cameras.map((camera) => (
              <div key={camera.id} className="rounded-lg border bg-white p-3">
                <div className="flex items-center justify-between">
                  <div className="font-medium text-sm">{camera.location}</div>
                  <Badge
                    className={`${
                      camera.status === 'ACTIVE'
                        ? 'bg-emerald-100 text-emerald-800'
                        : 'bg-slate-100 text-slate-700'
                    } rounded-full`}
                  >
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
          <CardTitle>Recent security activity</CardTitle>
          <CardDescription>Latest alert feed from the backend</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-80 pr-2">
            <div className="space-y-3">
              {recentAlerts.length === 0 ? (
                <div className="p-3 rounded-lg border bg-white text-sm text-slate-500">
                  No alerts have been generated yet.
                </div>
              ) : recentAlerts.map((alert) => (
                <div key={alert.id} className="p-3 rounded-lg border bg-white">
                  <div className="flex items-center justify-between gap-2">
                    <StatusBadge status={alert.status} />
                    <span className="text-xs text-slate-500">{formatTime(alert.created_at)}</span>
                  </div>
                  <div className="text-sm mt-2 text-slate-700">{alert.message}</div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

function Alerts({ query = '' }: { query?: string }) {
  const [alerts, setAlerts] = useState<ApiAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [updating, setUpdating] = useState<string | null>(null);
  const [updateError, setUpdateError] = useState('');

  const load = useCallback(() => {
    setLoading(true);
    setError('');
    const q = statusFilter !== 'all' ? `?status=${statusFilter}&limit=50` : '?limit=50';
    apiFetch<ApiAlert[]>(`/alerts/${q}`)
      .then(setAlerts)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [statusFilter]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const ws = new WebSocket(buildWsUrl('/ws/alerts'));
    ws.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        if (m.event_type === 'NEW_ALERT' || m.event_type === 'ALERT_STATUS_CHANGED') load();
      } catch {}
    };
    return () => ws.close();
  }, [load]);

  const updateStatus = async (id: string, status: AlertStatus) => {
    setUpdating(id);
    setUpdateError('');
    try {
      await apiFetch(`/alerts/${id}/status`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      });
      load();
    } catch (e: any) {
      setUpdateError(`No se pudo actualizar: ${e.message}`);
    } finally {
      setUpdating(null);
    }
  };

  const normalizedQuery = query.trim().toLowerCase();
  const filteredAlerts = alerts.filter((alert) =>
    !normalizedQuery ||
    alert.message.toLowerCase().includes(normalizedQuery) ||
    alert.status.toLowerCase().includes(normalizedQuery)
  );

  return (
    <div className="space-y-4">
      {error && <ErrorBanner msg={`Error cargando alertas: ${error}`} onClose={() => setError('')} />}
      {updateError && <ErrorBanner msg={updateError} onClose={() => setUpdateError('')} />}

      <div className="flex flex-col md:flex-row gap-3 md:items-end">
        <div className="flex-1">
          <Label>Estado</Label>
          <select
            className="w-full border rounded-md px-3 py-2 text-sm mt-1"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">Todos</option>
            <option value="UNREAD">Sin leer</option>
            <option value="ACKNOWLEDGED">Atendidas</option>
            <option value="RESOLVED">Resueltas</option>
          </select>
        </div>
        <Button variant="secondary" className="gap-2" onClick={load}>
          <RefreshCw className="h-4 w-4" />Actualizar
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-slate-500 text-sm p-4">
          <Loader2 className="h-4 w-4 animate-spin" />Cargando alertas…
        </div>
      ) : filteredAlerts.length === 0 ? (
        <div className="text-slate-500 text-sm p-6 border rounded-xl text-center">
          {alerts.length === 0 ? 'No hay alertas con ese filtro.' : 'No alerts match the current search.'}
        </div>
      ) : (
        <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-3">
          {filteredAlerts.map((a) => (
            <Card key={a.id}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <StatusBadge status={a.status} />
                  <CardDescription className="text-xs">
                    <Clock className="h-3 w-3 inline mr-1" />
                    {formatTime(a.created_at)}
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent className="text-sm text-slate-700">{a.message}</CardContent>
              <CardFooter className="flex gap-2 flex-wrap">
                {a.status === 'UNREAD' && (
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={updating === a.id}
                    onClick={() => updateStatus(a.id, 'ACKNOWLEDGED')}
                    className="gap-2"
                  >
                    {updating === a.id ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <CheckCircle2 className="h-4 w-4" />
                    )}
                    Atender
                  </Button>
                )}
                {a.status === 'ACKNOWLEDGED' && (
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={updating === a.id}
                    onClick={() => updateStatus(a.id, 'RESOLVED')}
                    className="gap-2"
                  >
                    {updating === a.id ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <CheckCircle2 className="h-4 w-4" />
                    )}
                    Resolver
                  </Button>
                )}
                {a.status === 'RESOLVED' && (
                  <span className="text-xs text-slate-400 self-center">Resuelta ✓</span>
                )}
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

function IncidentsList({ query = '' }: { query?: string }) {
  const [incidents, setIncidents] = useState<ApiIncident[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<ApiIncident | null>(null);
  const [simulating, setSimulating] = useState<string | null>(null);
  const [simError, setSimError] = useState('');
  const [simSuccess, setSimSuccess] = useState('');
  const [evidenceFiles, setEvidenceFiles] = useState<EvidenceFile[]>([]);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState('');
  const [evidenceUrl, setEvidenceUrl] = useState('');
  const [selectedEvidenceName, setSelectedEvidenceName] = useState('');

  const load = () => {
    setLoading(true);
    setError('');
    apiFetch<ApiIncident[]>('/incidents/?limit=30')
      .then(setIncidents)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (!selected) {
      setEvidenceFiles([]);
      setEvidenceUrl('');
      setSelectedEvidenceName('');
      setEvidenceError('');
      return;
    }

    setEvidenceLoading(true);
    setEvidenceError('');

    apiFetch<EvidenceFile[]>(`/evidence/incident/${selected.id}`)
      .then(async (files) => {
        setEvidenceFiles(files);
        if (!files.length) {
          setEvidenceUrl('');
          return;
        }
        const preferred =
          files.find((file) => /\.(jpg|jpeg|png|webp|mp4|webm)$/i.test(file.object_name)) ?? files[0];
        setSelectedEvidenceName(preferred.object_name);
        const urlData = await apiFetch<{ url: string }>(
          `/evidence/url?object_name=${encodeURIComponent(preferred.object_name)}`
        );
        setEvidenceUrl(urlData.url);
      })
      .catch((e) => setEvidenceError(e.message))
      .finally(() => setEvidenceLoading(false));
  }, [selected]);

  const simulate = async (module: string) => {
    setSimulating(module);
    setSimError('');
    setSimSuccess('');

    const detections =
      module === 'face'
        ? [{ name: 'unknown_person', confidence: 0.85 }]
        : module === 'weapons'
        ? [{ class: 'knife', confidence: 0.91 }]
        : [{ action: 'punch', confidence: 0.78 }];

    try {
      const result = await apiFetch<any>('/incidents/simulate', {
        method: 'POST',
        body: JSON.stringify({ module, camera_id: 'cam-demo-01', detections }),
      });
      setSimSuccess(`✓ Incidente creado — Regla: ${result.rule_triggered} · Prioridad: ${result.priority}`);
      load();
    } catch (e: any) {
      setSimError(`Error al simular: ${e.message}`);
    } finally {
      setSimulating(null);
    }
  };

  const normalizedQuery = query.trim().toLowerCase();
  const filteredIncidents = incidents.filter((incident) => {
    const meta = incident.incident_metadata ?? {};
    const haystack = [
      meta.module,
      meta.rule_triggered,
      meta.priority,
      meta.camera_id,
      incident.id,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();
    return !normalizedQuery || haystack.includes(normalizedQuery);
  });

  if (selected) {
    const meta = selected.incident_metadata;
    const isVideo = /\.(mp4|webm)$/i.test(selectedEvidenceName);
    const isImage = /\.(jpg|jpeg|png|webp)$/i.test(selectedEvidenceName);

    return (
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card className="xl:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Incidente</CardTitle>
                <CardDescription>
                  {meta.camera_id} · {formatTime(selected.created_at)}
                </CardDescription>
              </div>
              <Button variant="ghost" onClick={() => setSelected(null)}>← Volver</Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {evidenceLoading ? (
              <div className="aspect-video rounded-xl border bg-slate-50 grid place-items-center text-slate-500">
                <div className="flex items-center gap-2 text-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />Loading evidence…
                </div>
              </div>
            ) : evidenceError ? (
              <div className="aspect-video rounded-xl border border-red-200 bg-red-50 grid place-items-center text-red-700 p-6 text-center">
                <div className="space-y-2">
                  <XCircle className="h-8 w-8 mx-auto" />
                  <p className="font-medium">Evidence could not be loaded.</p>
                  <p className="text-sm">{evidenceError}</p>
                </div>
              </div>
            ) : evidenceUrl && isImage ? (
              <div className="aspect-video rounded-xl border overflow-hidden bg-black">
                <img src={evidenceUrl} alt="Incident evidence" className="h-full w-full object-contain" />
              </div>
            ) : evidenceUrl && isVideo ? (
              <div className="aspect-video rounded-xl border overflow-hidden bg-black">
                <video src={evidenceUrl} controls className="h-full w-full object-contain" />
              </div>
            ) : (
              <div className="aspect-video rounded-xl border border-dashed bg-slate-50 grid place-items-center text-slate-500 p-6 text-center">
                <div className="space-y-2">
                  <Database className="h-8 w-8 mx-auto" />
                  <p className="font-medium">No evidence file is available for this incident yet.</p>
                  <p className="text-sm">
                    The incident exists, but there is no saved MinIO evidence to render in this view.
                  </p>
                </div>
              </div>
            )}

            <div className="grid md:grid-cols-3 gap-3 text-xs">
              {[['Módulo', meta.module], ['Regla', meta.rule_triggered], ['Prioridad', meta.priority]].map(([k, v]) => (
                <div key={k} className="rounded-lg border p-2">
                  <div className="font-medium">{k}</div>
                  <div className="text-slate-500">{v ?? '–'}</div>
                </div>
              ))}
            </div>
          </CardContent>
          <CardFooter className="flex gap-2">
            <Button className="gap-2"><Bell className="h-4 w-4" />Notificar</Button>
            <Button variant="outline" className="gap-2"><Database className="h-4 w-4" />Guardar evidencia</Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Metadatos</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span>Prioridad</span>
              <SeverityBadge level={priorityToSeverity(meta.priority)} />
            </div>
            <div className="flex justify-between">
              <span>Cámara</span>
              <span>{meta.camera_id ?? '–'}</span>
            </div>
            <div className="flex justify-between">
              <span>ID</span>
              <span className="text-xs text-slate-400">{selected.id.slice(0, 8)}…</span>
            </div>

            <div>
              <div className="font-medium mb-2">Evidence objects</div>
              {evidenceFiles.length === 0 ? (
                <div className="text-xs text-slate-500">No evidence objects listed.</div>
              ) : (
                <div className="space-y-2">
                  {evidenceFiles.map((file) => (
                    <div key={file.object_name} className="rounded-md border p-2 text-xs">
                      <div className="font-medium break-all">{file.object_name}</div>
                      <div className="text-slate-500">{Math.round((file.size ?? 0) / 1024)} KB</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Incidentes</h2>
          <p className="text-sm text-slate-500">Eventos detectados por los módulos de IA</p>
        </div>
        <Button variant="outline" className="gap-2" onClick={load}>
          <RefreshCw className="h-4 w-4" />Actualizar
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Simular evento</CardTitle>
          <CardDescription>Crea un incidente + alerta sin necesitar los módulos de IA activos</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {simError && <ErrorBanner msg={simError} onClose={() => setSimError('')} />}
          {simSuccess && <SuccessBanner msg={simSuccess} onClose={() => setSimSuccess('')} />}
          <div className="flex gap-3 flex-wrap">
            {([['face', 'Persona desconocida'], ['weapons', 'Arma'], ['pose', 'Agresión']] as [string, string][]).map(([m, label]) => (
              <Button
                key={m}
                variant="outline"
                size="sm"
                disabled={simulating !== null}
                onClick={() => simulate(m)}
                className="gap-2"
              >
                {simulating === m ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wand2 className="h-3 w-3" />}
                {simulating === m ? 'Simulando…' : label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {error && <ErrorBanner msg={`Error cargando incidentes: ${error}`} onClose={() => setError('')} />}

      {loading ? (
        <div className="flex items-center gap-2 text-slate-500 text-sm p-4">
          <Loader2 className="h-4 w-4 animate-spin" />Cargando…
        </div>
      ) : filteredIncidents.length === 0 ? (
        <div className="text-slate-500 text-sm p-6 border rounded-xl text-center">
          {incidents.length === 0 ? 'No hay incidentes. Usa el simulador para crear uno.' : 'No incidents match the current search.'}
        </div>
      ) : (
        <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-3">
          {filteredIncidents.map((inc) => {
            const meta = inc.incident_metadata;
            return (
              <Card key={inc.id} className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => setSelected(inc)}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm">{meta.module?.toUpperCase() ?? 'EVENTO'}</CardTitle>
                    <SeverityBadge level={priorityToSeverity(meta.priority)} />
                  </div>
                  <CardDescription className="text-xs flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatTime(inc.created_at)} · {meta.camera_id ?? '–'}
                  </CardDescription>
                </CardHeader>
                <CardContent className="text-xs text-slate-500">Regla: {meta.rule_triggered ?? '–'}</CardContent>
                <CardFooter>
                  <Button variant="ghost" size="sm" className="gap-1 text-xs">
                    Ver detalle <ChevronRight className="h-3 w-3" />
                  </Button>
                </CardFooter>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Residents({ query = '' }: { query?: string }) {
  const [persons, setPersons] = useState<ApiPerson[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState('');
  const [createSuccess, setCreateSuccess] = useState('');
  const [form, setForm] = useState({
    full_name: '',
    person_type: 'RESIDENT',
    building: '',
    apartment: '',
    phone: '',
    email: '',
    valid_from: '',
    valid_until: '',
  });
  const [enrollSlots, setEnrollSlots] = useState<(File | null)[]>([null, null, null]);
  const [enrollTarget, setEnrollTarget] = useState<string | null>(null);
  const [enrolling, setEnrolling] = useState(false);
  const [enrollMsg, setEnrollMsg] = useState('');
  const [enrollError, setEnrollError] = useState('');

  const load = () => {
    setLoadError('');
    apiFetch<ApiPerson[]>('/persons/')
      .then(setPersons)
      .catch((e) => setLoadError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const createPerson = async () => {
    if (!form.full_name.trim()) return;

    setCreating(true);
    setCreateError('');
    setCreateSuccess('');

    try {
      await apiFetch('/persons/', {
        method: 'POST',
        body: JSON.stringify({
          full_name: form.full_name.trim(),
          person_type: form.person_type,
          building: form.building || null,
          apartment: form.apartment || null,
          phone: form.phone || null,
          email: form.email || null,
          valid_from: form.valid_from || null,
          valid_until: form.valid_until || null,
        }),
      });

      setCreateSuccess(`✓ "${form.full_name}" registered successfully`);
      setForm({
        full_name: '',
        person_type: 'RESIDENT',
        building: '',
        apartment: '',
        phone: '',
        email: '',
        valid_from: '',
        valid_until: '',
      });
      load();
    } catch (e: any) {
      setCreateError(`Error al registrar: ${e.message}`);
    } finally {
      setCreating(false);
    }
  };

  const setSlotFile = (slotIndex: number, file: File | null) => {
    setEnrollSlots((current) =>
      current.map((currentFile, index) => (index === slotIndex ? file : currentFile))
    );
  };

  const clearEnrollmentState = () => {
    setEnrollTarget(null);
    setEnrollSlots([null, null, null]);
  };

  const enrollBiometrics = async (personId: string) => {
    const validFiles = enrollSlots.filter(Boolean) as File[];
    if (validFiles.length !== 3) {
      setEnrollError('Exactly 3 facial images are required to build the resident master vector.');
      return;
    }

    setEnrolling(true);
    setEnrollMsg('');
    setEnrollError('');

    const fd = new FormData();
    validFiles.forEach((file) => fd.append('files', file));

    try {
      const res = await fetch(`/api/persons/${personId}/enroll`, { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setEnrollMsg(data.message ?? 'Enrolamiento exitoso');
      load();
    } catch (e: any) {
      setEnrollError(`Error al enrolar: ${e.message}`);
    } finally {
      setEnrolling(false);
      clearEnrollmentState();
    }
  };

  const normalizedQuery = query.trim().toLowerCase();
  const filteredPersons = persons.filter((person) => {
    const haystack = [
      person.full_name,
      person.person_type,
      person.building,
      person.apartment,
      person.email,
      person.phone,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();

    return !normalizedQuery || haystack.includes(normalizedQuery);
  });

  const requiresLocation = form.person_type === 'RESIDENT' || form.person_type === 'VISITOR';
  const isVisitor = form.person_type === 'VISITOR';
  const stagedCount = enrollSlots.filter(Boolean).length;
  const canFinalizeEnrollment = stagedCount === 3;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>People and biometric enrollment</CardTitle>
          <CardDescription>
            Residents require an exact 3-image staging workflow before enrollment.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <Label>Registered people</Label>
            {loadError && <ErrorBanner msg={loadError} onClose={() => setLoadError('')} />}
            {enrollMsg && <SuccessBanner msg={enrollMsg} onClose={() => setEnrollMsg('')} />}
            {enrollError && <ErrorBanner msg={enrollError} onClose={() => setEnrollError('')} />}

            {loading ? (
              <div className="flex items-center gap-2 text-slate-500 text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />Cargando…
              </div>
            ) : filteredPersons.length === 0 ? (
              <div className="text-slate-500 text-sm border rounded-lg p-3">
                {persons.length === 0 ? 'No hay personas registradas.' : 'No registered people match the current search.'}
              </div>
            ) : (
              <ScrollArea className="h-80">
                <div className="space-y-2 pr-2">
                  {filteredPersons.map((person) => (
                    <div key={person.id} className="rounded-xl border p-3 bg-white">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="font-medium">{person.full_name}</div>
                          <div className="text-xs text-slate-500">
                            {person.person_type}
                            {person.building ? ` · ${person.building}` : ''}
                            {person.apartment ? ` / ${person.apartment}` : ''}
                          </div>
                          {(person.valid_from || person.valid_until) && (
                            <div className="text-[11px] text-slate-400">
                              {person.valid_from ? `from ${new Date(person.valid_from).toLocaleString('es-MX')}` : 'open start'}
                              {' · '}
                              {person.valid_until ? `until ${new Date(person.valid_until).toLocaleString('es-MX')}` : 'open end'}
                            </div>
                          )}
                        </div>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            setEnrollTarget(person.id);
                            setEnrollSlots([null, null, null]);
                            setEnrollMsg('');
                            setEnrollError('');
                          }}
                        >
                          Enroll
                        </Button>
                      </div>

                      {enrollTarget === person.id && (
                        <div className="mt-3 space-y-3 border-t pt-3">
                          <div className="flex items-center justify-between">
                            <Label className="text-xs">Exact triple capture / staging buffer</Label>
                            <Badge className="rounded-full bg-slate-100 text-slate-700">
                              {stagedCount}/3 ready
                            </Badge>
                          </div>

                          <div className="grid gap-3">
                            {enrollSlots.map((file, slotIndex) => (
                              <div key={slotIndex} className="rounded-lg border p-3 bg-slate-50">
                                <div className="flex items-center justify-between gap-3">
                                  <div>
                                    <div className="font-medium text-sm">Slot {slotIndex + 1}</div>
                                    <div className="text-xs text-slate-500">
                                      {file ? file.name : 'No frame selected yet'}
                                    </div>
                                  </div>
                                  <div className="flex gap-2">
                                    <label className="inline-flex">
                                      <input
                                        type="file"
                                        accept="image/jpeg,image/png"
                                        className="hidden"
                                        onChange={(e) => setSlotFile(slotIndex, e.target.files?.[0] ?? null)}
                                      />
                                      <span className="inline-flex items-center rounded-md border px-3 py-1 text-xs cursor-pointer bg-white hover:bg-slate-100">
                                        {file ? 'Retake' : 'Capture'}
                                      </span>
                                    </label>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      disabled={!file}
                                      onClick={() => setSlotFile(slotIndex, null)}
                                    >
                                      Delete
                                    </Button>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>

                          <div className="flex gap-2 mt-1">
                            <Button
                              size="sm"
                              disabled={enrolling || !canFinalizeEnrollment}
                              onClick={() => enrollBiometrics(person.id)}
                              className="gap-2"
                            >
                              {enrolling ? (
                                <Loader2 className="h-3 w-3 animate-spin" />
                              ) : (
                                <UserPlus className="h-3 w-3" />
                              )}
                              {enrolling ? 'Processing…' : 'Finalize enrollment'}
                            </Button>
                            <Button size="sm" variant="ghost" onClick={clearEnrollmentState}>
                              Cancel
                            </Button>
                          </div>

                          {!canFinalizeEnrollment && (
                            <div className="text-xs text-amber-700">
                              Finalize remains disabled until exactly 3 valid images are staged.
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </div>

          <div className="space-y-3">
            <Label>Register new person</Label>
            {createError && <ErrorBanner msg={createError} onClose={() => setCreateError('')} />}
            {createSuccess && <SuccessBanner msg={createSuccess} onClose={() => setCreateSuccess('')} />}

            <div>
              <Label className="text-xs text-slate-500">Full name</Label>
              <Input
                placeholder="Ej: Ana García"
                value={form.full_name}
                onChange={(e) => setForm({ ...form, full_name: e.target.value })}
                onKeyDown={(e) => e.key === 'Enter' && createPerson()}
              />
            </div>

            <div>
              <Label className="text-xs text-slate-500">Type</Label>
              <select
                className="w-full border rounded-md px-3 py-2 text-sm mt-1"
                value={form.person_type}
                onChange={(e) => setForm({ ...form, person_type: e.target.value })}
              >
                <option value="RESIDENT">Resident</option>
                <option value="VISITOR">Visitor</option>
                <option value="STAFF">Staff</option>
              </select>
            </div>

            {requiresLocation && (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs text-slate-500">Building</Label>
                  <Input
                    value={form.building}
                    onChange={(e) => setForm({ ...form, building: e.target.value })}
                    placeholder="Tower A"
                  />
                </div>
                <div>
                  <Label className="text-xs text-slate-500">Apartment</Label>
                  <Input
                    value={form.apartment}
                    onChange={(e) => setForm({ ...form, apartment: e.target.value })}
                    placeholder="301"
                  />
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs text-slate-500">Phone</Label>
                <Input
                  value={form.phone}
                  onChange={(e) => setForm({ ...form, phone: e.target.value })}
                  placeholder="+52..."
                />
              </div>
              <div>
                <Label className="text-xs text-slate-500">Email</Label>
                <Input
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  placeholder="person@example.com"
                />
              </div>
            </div>

            {isVisitor && (
              <div className="grid grid-cols-1 gap-3">
                <div>
                  <Label className="text-xs text-slate-500">Valid from</Label>
                  <Input
                    type="datetime-local"
                    value={form.valid_from}
                    onChange={(e) => setForm({ ...form, valid_from: e.target.value })}
                  />
                </div>
                <div>
                  <Label className="text-xs text-slate-500">Valid until</Label>
                  <Input
                    type="datetime-local"
                    value={form.valid_until}
                    onChange={(e) => setForm({ ...form, valid_until: e.target.value })}
                  />
                </div>
              </div>
            )}

            <Button
              className="w-full gap-2"
              disabled={creating || !form.full_name.trim()}
              onClick={createPerson}
            >
              {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <UserPlus className="h-4 w-4" />}
              {creating ? 'Registrando…' : 'Register person'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function AccessGate({ onRegisterVisitor }: { onRegisterVisitor: () => void }) {
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

function SettingsPanel() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Card>
        <CardHeader>
          <CardTitle>General</CardTitle>
          <CardDescription>Preferencias del sistema</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">Notificaciones push</div>
              <div className="text-xs text-slate-500">Enviar a residentes y guardias</div>
            </div>
            <Switch defaultChecked />
          </div>
          <div>
            <Label>Zona horaria</Label>
            <Select defaultValue="America/Mexico_City">
              <NativeSelect>
                <SelectItem value="America/Mexico_City">America/Mexico_City</SelectItem>
                <SelectItem value="UTC">UTC</SelectItem>
              </NativeSelect>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Reconocimiento facial</CardTitle>
          <CardDescription>Umbrales y calidad</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Umbral de similitud</Label>
            <Input type="number" defaultValue={0.52} step="0.01" />
            <p className="text-xs text-slate-500">Mayor = más estricto</p>
          </div>
          <div className="space-y-1">
            <Label>Normalización</Label>
            <Select defaultValue="ArcFace">
              <NativeSelect>
                <SelectItem value="ArcFace">Hiperesfera (ArcFace)</SelectItem>
                <SelectItem value="CosFace">CosFace</SelectItem>
              </NativeSelect>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Análisis corporal</CardTitle>
          <CardDescription>Acciones y ventanas</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Ventana temporal (frames)</Label>
            <Input type="number" defaultValue={64} />
          </div>
          <div className="space-y-1">
            <Label>Clases monitoreadas</Label>
            <div className="flex flex-wrap gap-1">
              {['Empujón', 'Golpe', 'Caída', 'Cuchillo', 'Pistola'].map((c) => (
                <Badge key={c} variant="outline" className="rounded-full">
                  {c}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<'dashboard' | 'live' | 'alerts' | 'incidents' | 'residents' | 'access' | 'settings'>(() => {
    const saved = window.localStorage.getItem('uh_security_active_tab');
    return (saved as any) || 'dashboard';
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [alertCounts, setAlertCounts] = useState<AlertCounts>({ unread: 0, acknowledged: 0, resolved: 0 });
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    window.localStorage.setItem('uh_security_active_tab', tab);
  }, [tab]);

  useEffect(() => {
    const ws = new WebSocket(buildWsUrl('/ws/alerts'));
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event_type === 'ALERT_COUNT_UPDATE') {
          setAlertCounts({
            unread: msg.data.unread,
            acknowledged: msg.data.acknowledged,
            resolved: msg.data.resolved,
          });
        }
      } catch {}
    };
    return () => ws.close();
  }, []);

  useEffect(() => {
    Promise.all([
      apiFetch<{ count: number }>('/alerts/count?status=UNREAD'),
      apiFetch<{ count: number }>('/alerts/count?status=ACKNOWLEDGED'),
      apiFetch<{ count: number }>('/alerts/count?status=RESOLVED'),
    ])
      .then(([u, a, r]) => setAlertCounts({ unread: u.count, acknowledged: a.count, resolved: r.count }))
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-slate-50">
      <div className="max-w-7xl mx-auto p-4">
        <Header
          unreadCount={alertCounts.unread}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onOpenAlerts={() => setTab('alerts')}
        />

        <div className="flex items-center gap-2 mb-3 text-xs text-slate-500">
          <div className={`h-2 w-2 rounded-full ${wsConnected ? 'bg-emerald-500' : 'bg-red-400'}`} />
          {wsConnected ? 'Backend connected in real time' : 'No real-time backend connection available'}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[240px_1fr] gap-4">
          <div className="hidden lg:block">
            <Card className="sticky top-4">
              <CardHeader>
                <CardTitle className="text-base">Navegación</CardTitle>
                <CardDescription>Secciones del sistema</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <nav className="grid gap-1">
                  {[
                    { id: 'dashboard', icon: Gauge, label: 'Dashboard' },
                    { id: 'live', icon: Camera, label: 'Monitoreo en vivo' },
                    { id: 'alerts', icon: AlertTriangle, label: 'Alertas', badge: alertCounts.unread },
                    { id: 'incidents', icon: ListChecks, label: 'Incidentes' },
                    { id: 'residents', icon: Users, label: 'Residentes' },
                    { id: 'access', icon: DoorOpen, label: 'Control de Acceso' },
                    { id: 'settings', icon: Settings, label: 'Configuración' },
                  ].map((it) => (
                    <Button
                      key={it.id}
                      variant={tab === it.id ? 'secondary' : 'ghost'}
                      className="justify-start gap-2 relative"
                      onClick={() => setTab(it.id as any)}
                    >
                      <it.icon className="h-4 w-4" />
                      {it.label}
                      {it.badge && it.badge > 0 ? (
                        <span className="ml-auto bg-red-500 text-white text-[10px] rounded-full h-4 w-4 flex items-center justify-center">
                          {it.badge > 9 ? '9+' : it.badge}
                        </span>
                      ) : null}
                    </Button>
                  ))}
                </nav>
              </CardContent>
            </Card>
          </div>

          <div>
            {tab === 'dashboard' && <Dashboard alertCounts={alertCounts} />}
            {tab === 'live' && <LiveMonitoring />}
            {tab === 'alerts' && <Alerts query={searchQuery} />}
            {tab === 'incidents' && <IncidentsList query={searchQuery} />}
            {tab === 'residents' && <Residents query={searchQuery} />}
            {tab === 'access' && <AccessGate onRegisterVisitor={() => setTab('residents')} />}
            {tab === 'settings' && <SettingsPanel />}
          </div>
        </div>
      </div>
    </div>
  );
}