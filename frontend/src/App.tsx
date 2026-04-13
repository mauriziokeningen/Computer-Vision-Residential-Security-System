import React, { useState, useEffect, useRef, useCallback } from 'react';
import WebcamFeed from './components/WebcamFeed';
import { motion } from 'framer-motion';
import {
  AlertTriangle, Bell, Camera, ChevronRight, DoorOpen, Gauge, ListChecks,
  PersonStanding, Settings, ShieldCheck, Siren, UserPlus, Users, Video,
  Wand2, Search, CheckCircle2, CircleX, Clock, Cpu, Database, Eye,
  RefreshCw, Loader2, XCircle,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { ScrollArea } from './components/ui/scroll-area';
import { Switch } from './components/ui/switch';
import { Badge } from './components/ui/badge';
import { Select, NativeSelect, SelectItem } from './components/ui/select';
import { Progress } from './components/ui/progress';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// ─── Types ────────────────────────────────────────────────────────────────────

type AlertStatus = 'UNREAD' | 'ACKNOWLEDGED' | 'RESOLVED';

interface ApiAlert {
  id: string; incident_id: string | null; message: string;
  status: AlertStatus; created_at: string; resolved_at: string | null;
}
interface ApiIncident {
  id: string; created_at: string;
  incident_metadata: { rule_triggered?: string; priority?: string; module?: string; camera_id?: string; timestamp?: string; detections?: any[]; };
}
interface ApiPerson { id: string; full_name: string; person_type: string; created_at: string; }
interface AlertCounts { unread: number; acknowledged: number; resolved: number; }

// ─── API helpers ──────────────────────────────────────────────────────────────

const API = '/api';

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { const body = await res.json(); detail = body.detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// ─── Error Banner ─────────────────────────────────────────────────────────────

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

// ─── Shared UI helpers ────────────────────────────────────────────────────────

function priorityToSeverity(p?: string): 'low' | 'high' | 'critical' {
  if (p === 'CRITICAL') return 'critical'; if (p === 'HIGH') return 'high'; return 'low';
}
function SeverityBadge({ level }: { level: 'low' | 'high' | 'critical' }) {
  const m = { low: 'bg-slate-100 text-slate-800', high: 'bg-amber-100 text-amber-800', critical: 'bg-red-100 text-red-800' };
  const l = { low: 'Baja', high: 'Alta', critical: 'Crítica' };
  return <Badge className={`${m[level]} rounded-full`}>{l[level]}</Badge>;
}
function StatusBadge({ status }: { status: AlertStatus }) {
  const m: Record<AlertStatus, string> = { UNREAD: 'bg-red-100 text-red-800', ACKNOWLEDGED: 'bg-amber-100 text-amber-800', RESOLVED: 'bg-emerald-100 text-emerald-800' };
  const l: Record<AlertStatus, string> = { UNREAD: 'Sin leer', ACKNOWLEDGED: 'Atendida', RESOLVED: 'Resuelta' };
  return <Badge className={`${m[status]} rounded-full`}>{l[status]}</Badge>;
}
function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit' });
}

// ─── Header ───────────────────────────────────────────────────────────────────

function Header({ unreadCount }: { unreadCount: number }) {
  return (
    <div className="flex items-center justify-between py-4">
      <div className="flex items-center gap-3">
        <Siren className="h-7 w-7" />
        <div>
          <h1 className="text-xl font-semibold leading-none">Seguridad UH · Panel</h1>
          <p className="text-sm text-slate-500">Monitoreo en tiempo real · TT2</p>
        </div>
      </div>
      <div className="flex gap-2 items-center">
        <Button variant="outline" className="gap-2"><Search className="h-4 w-4" />Buscar</Button>
        <Button className="gap-2 relative">
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

// ─── Root App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState<'dashboard' | 'live' | 'alerts' | 'incidents' | 'residents' | 'access' | 'settings'>('dashboard');
  const [alertCounts, setAlertCounts] = useState<AlertCounts>({ unread: 0, acknowledged: 0, resolved: 0 });
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/alerts`);
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event_type === 'ALERT_COUNT_UPDATE')
          setAlertCounts({ unread: msg.data.unread, acknowledged: msg.data.acknowledged, resolved: msg.data.resolved });
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
        <Header unreadCount={alertCounts.unread} />
        <div className="flex items-center gap-2 mb-3 text-xs text-slate-500">
          <div className={`h-2 w-2 rounded-full ${wsConnected ? 'bg-emerald-500' : 'bg-red-400'}`} />
          {wsConnected ? 'Backend conectado en tiempo real' : 'Sin conexión al backend — ¿está corriendo en puerto 8000?'}
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
                    <Button key={it.id} variant={tab === it.id ? 'secondary' : 'ghost'} className="justify-start gap-2 relative" onClick={() => setTab(it.id as any)}>
                      <it.icon className="h-4 w-4" />{it.label}
                      {it.badge && it.badge > 0 ? (
                        <span className="ml-auto bg-red-500 text-white text-[10px] rounded-full h-4 w-4 flex items-center justify-center">{it.badge > 9 ? '9+' : it.badge}</span>
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
            {tab === 'alerts' && <Alerts />}
            {tab === 'incidents' && <IncidentsList />}
            {tab === 'residents' && <Residents />}
            {tab === 'access' && <AccessGate />}
            {tab === 'settings' && <SettingsPanel />}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── KPI ──────────────────────────────────────────────────────────────────────

function KPI({ icon: Icon, label, value, sub }: { icon: any; label: string; value: string; sub: string }) {
  return (
    <Card><CardContent className="p-4">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-xl bg-slate-100"><Icon className="h-5 w-5" /></div>
        <div className="flex-1">
          <div className="text-xs text-slate-500">{label}</div>
          <div className="text-xl font-semibold">{value}</div>
          <div className="text-xs text-slate-500">{sub}</div>
        </div>
      </div>
    </CardContent></Card>
  );
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

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
          const d = new Date(now); d.setHours(now.getHours() - h);
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
          <CardHeader><CardTitle>Tendencia de alertas (hoy)</CardTitle><CardDescription>Frecuencia por hora</CardDescription></CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={statsSeries}>
                <XAxis dataKey="t" /><YAxis allowDecimals={false} /><Tooltip />
                <Line type="monotone" dataKey="a" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Estado de alertas</CardTitle><CardDescription>Distribución actual</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            {[
              { label: 'Sin leer', value: alertCounts.unread, color: 'bg-red-500' },
              { label: 'Atendidas', value: alertCounts.acknowledged, color: 'bg-amber-500' },
              { label: 'Resueltas', value: alertCounts.resolved, color: 'bg-emerald-500' },
            ].map((s) => (
              <div key={s.label} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2"><div className={`h-2 w-2 rounded-full ${s.color}`} /><span>{s.label}</span></div>
                  <span className="font-medium">{s.value}</span>
                </div>
                <Progress value={total > 0 ? (s.value / total) * 100 : 0} />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader><CardTitle>Últimas alertas</CardTitle><CardDescription>Datos reales desde la API</CardDescription></CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center gap-2 text-slate-500 text-sm p-4"><Loader2 className="h-4 w-4 animate-spin" />Cargando alertas…</div>
          ) : recentAlerts.length === 0 ? (
            <div className="text-slate-500 text-sm p-4">No hay alertas. Simula un evento en la sección Incidentes.</div>
          ) : (
            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
              {recentAlerts.slice(0, 6).map((a) => (
                <motion.div key={a.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
                  <Card className="border-l-4" style={{ borderLeftColor: a.status === 'UNREAD' ? '#ef4444' : a.status === 'ACKNOWLEDGED' ? '#f59e0b' : '#10b981' }}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <StatusBadge status={a.status} />
                        <CardDescription className="text-xs flex items-center gap-1"><Clock className="h-3 w-3" />{formatTime(a.created_at)}</CardDescription>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0 text-xs text-slate-600">{a.message.length > 60 ? a.message.slice(0, 60) + '…' : a.message}</CardContent>
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

// ─── Live Monitoring ──────────────────────────────────────────────────────────

function LiveMonitoring() {
  const wsRef = useRef<WebSocket | null>(null);
  const [dets, setDets] = useState<any[]>([]);
  const [frameSize, setFrameSize] = useState({ width: 640, height: 480 });
  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed'>('connecting');

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`);
    wsRef.current = ws;
    ws.onopen = () => setWsStatus('open');
    ws.onclose = () => setWsStatus('closed');
    ws.onmessage = (ev) => { try { const d = JSON.parse(ev.data); setDets(d.detections ?? []); } catch {} };
    return () => ws.close();
  }, []);

  const handleFrame = useCallback((canvas: HTMLCanvasElement, width: number, height: number) => {
    setFrameSize({ width, height });
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.7) }));
  }, []);

  const weaponDets = dets.filter((d) => { const n = d.name?.toLowerCase?.() ?? ''; return n.includes('knife') || n.includes('pistol') || n.includes('gun'); });

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Card className="xl:col-span-2">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div><CardTitle><Video className="h-5 w-5 inline mr-2" />Cámara – Acceso Principal</CardTitle><CardDescription>Detección en tiempo real</CardDescription></div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <div className={`h-2 w-2 rounded-full ${wsStatus === 'open' ? 'bg-emerald-500' : wsStatus === 'connecting' ? 'bg-amber-400' : 'bg-red-500'}`} />
              {wsStatus === 'open' ? 'WS conectado' : wsStatus === 'connecting' ? 'Conectando…' : 'Sin conexión'}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative aspect-video rounded-xl bg-black overflow-hidden">
            <WebcamFeed className="absolute inset-0" onFrame={handleFrame} fps={8} />
            <div className="absolute inset-0 pointer-events-none">
              {dets.map((d, i) => {
                const [x1, y1, x2, y2] = d.bbox ?? [0, 0, 0, 0];
                const n = d.name?.toLowerCase?.() ?? '';
                if (!n.includes('knife') && !n.includes('pistol') && !n.includes('gun')) return null;
                return (
                  <div key={i} className="absolute border-2 border-red-500"
                    style={{ left: `${((frameSize.width - x2) / frameSize.width) * 100}%`, top: `${(y1 / frameSize.height) * 100}%`, width: `${((x2 - x1) / frameSize.width) * 100}%`, height: `${((y2 - y1) / frameSize.height) * 100}%` }}>
                    <div className="absolute -top-6 left-0 bg-red-600 text-white text-[10px] px-2 py-1 rounded">{d.name} #{d.track_id ?? '-'} · {Math.round((d.conf ?? 0) * 100)}%</div>
                  </div>
                );
              })}
            </div>
            <div className="absolute top-4 left-4 space-y-2">
              {weaponDets.slice(0, 4).map((d, i) => (
                <div key={i} className="backdrop-blur bg-white/70 border px-3 py-1 rounded-lg text-xs border-red-500 text-red-600">{d.name} #{d.track_id ?? '-'} · conf {Math.round((d.conf ?? 0) * 100)}%</div>
              ))}
            </div>
          </div>
        </CardContent>
        <CardFooter className="flex items-center gap-3">
          <Button variant="outline" className="gap-2"><Eye className="h-4 w-4" />Pantalla completa</Button>
          <Button className="gap-2"><Bell className="h-4 w-4" />Enviar alerta</Button>
        </CardFooter>
      </Card>
      <Card>
        <CardHeader><CardTitle>Detecciones activas</CardTitle><CardDescription>Armas en el frame actual</CardDescription></CardHeader>
        <CardContent>
          <ScrollArea className="h-72 pr-2">
            <div className="space-y-3">
              {weaponDets.length === 0 ? (
                <div className="p-3 rounded-lg border bg-white text-sm text-slate-500">Sin detecciones por el momento.</div>
              ) : weaponDets.map((d, i) => (
                <div key={i} className="p-3 rounded-lg border bg-white">
                  <div className="flex items-center justify-between"><div className="font-medium text-sm">{d.name} #{d.track_id ?? '-'}</div><Badge className="bg-red-100 text-red-800 rounded-full">Crítica</Badge></div>
                  <div className="text-xs text-slate-500">confianza: {Math.round((d.conf ?? 0) * 100)}%</div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

// ─── Alerts ───────────────────────────────────────────────────────────────────

function Alerts() {
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

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/alerts`);
    ws.onmessage = (ev) => {
      try { const m = JSON.parse(ev.data); if (m.event_type === 'NEW_ALERT' || m.event_type === 'ALERT_STATUS_CHANGED') load(); } catch {}
    };
    return () => ws.close();
  }, [load]);

  const updateStatus = async (id: string, status: AlertStatus) => {
    setUpdating(id);
    setUpdateError('');
    try {
      await apiFetch(`/alerts/${id}/status`, { method: 'PATCH', body: JSON.stringify({ status }) });
      load();
    } catch (e: any) {
      setUpdateError(`No se pudo actualizar: ${e.message}`);
    } finally {
      setUpdating(null);
    }
  };

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
        <Button variant="secondary" className="gap-2" onClick={load}><RefreshCw className="h-4 w-4" />Actualizar</Button>
      </div>
      {loading ? (
        <div className="flex items-center gap-2 text-slate-500 text-sm p-4"><Loader2 className="h-4 w-4 animate-spin" />Cargando alertas…</div>
      ) : alerts.length === 0 ? (
        <div className="text-slate-500 text-sm p-6 border rounded-xl text-center">No hay alertas con ese filtro.</div>
      ) : (
        <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-3">
          {alerts.map((a) => (
            <Card key={a.id}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <StatusBadge status={a.status} />
                  <CardDescription className="text-xs"><Clock className="h-3 w-3 inline mr-1" />{formatTime(a.created_at)}</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="text-sm text-slate-700">{a.message}</CardContent>
              <CardFooter className="flex gap-2 flex-wrap">
                {a.status === 'UNREAD' && (
                  <Button size="sm" variant="outline" disabled={updating === a.id} onClick={() => updateStatus(a.id, 'ACKNOWLEDGED')} className="gap-2">
                    {updating === a.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}Atender
                  </Button>
                )}
                {a.status === 'ACKNOWLEDGED' && (
                  <Button size="sm" variant="outline" disabled={updating === a.id} onClick={() => updateStatus(a.id, 'RESOLVED')} className="gap-2">
                    {updating === a.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}Resolver
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

// ─── Incidents ────────────────────────────────────────────────────────────────

function IncidentsList() {
  const [incidents, setIncidents] = useState<ApiIncident[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<ApiIncident | null>(null);
  const [simulating, setSimulating] = useState<string | null>(null);
  const [simError, setSimError] = useState('');
  const [simSuccess, setSimSuccess] = useState('');

  const load = () => {
    setLoading(true);
    setError('');
    apiFetch<ApiIncident[]>('/incidents/?limit=30')
      .then(setIncidents)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };
  useEffect(() => { load(); }, []);

  const simulate = async (module: string) => {
    setSimulating(module);
    setSimError('');
    setSimSuccess('');
    const detections =
      module === 'face' ? [{ name: 'unknown_person', confidence: 0.85 }]
      : module === 'weapons' ? [{ class: 'knife', confidence: 0.91 }]
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

  if (selected) {
    const meta = selected.incident_metadata;
    return (
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card className="xl:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div><CardTitle>Incidente</CardTitle><CardDescription>{meta.camera_id} · {formatTime(selected.created_at)}</CardDescription></div>
              <Button variant="ghost" onClick={() => setSelected(null)}>← Volver</Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="aspect-video rounded-xl bg-slate-200 grid place-items-center text-slate-500">
              <PersonStanding className="h-10 w-10 opacity-70" /><p className="text-sm">Evidencia de video</p>
            </div>
            <div className="grid md:grid-cols-3 gap-3 text-xs">
              {[['Módulo', meta.module], ['Regla', meta.rule_triggered], ['Prioridad', meta.priority]].map(([k, v]) => (
                <div key={k} className="rounded-lg border p-2"><div className="font-medium">{k}</div><div className="text-slate-500">{v ?? '–'}</div></div>
              ))}
            </div>
          </CardContent>
          <CardFooter className="flex gap-2">
            <Button className="gap-2"><Bell className="h-4 w-4" />Notificar</Button>
            <Button variant="outline" className="gap-2"><Database className="h-4 w-4" />Guardar evidencia</Button>
          </CardFooter>
        </Card>
        <Card>
          <CardHeader><CardTitle>Metadatos</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between"><span>Prioridad</span><SeverityBadge level={priorityToSeverity(meta.priority)} /></div>
            <div className="flex justify-between"><span>Cámara</span><span>{meta.camera_id ?? '–'}</span></div>
            <div className="flex justify-between"><span>ID</span><span className="text-xs text-slate-400">{selected.id.slice(0, 8)}…</span></div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div><h2 className="text-lg font-semibold">Incidentes</h2><p className="text-sm text-slate-500">Eventos detectados por los módulos de IA</p></div>
        <Button variant="outline" className="gap-2" onClick={load}><RefreshCw className="h-4 w-4" />Actualizar</Button>
      </div>

      {/* Simulador */}
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
              <Button key={m} variant="outline" size="sm" disabled={simulating !== null} onClick={() => simulate(m)} className="gap-2">
                {simulating === m ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wand2 className="h-3 w-3" />}
                {simulating === m ? 'Simulando…' : label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {error && <ErrorBanner msg={`Error cargando incidentes: ${error}`} onClose={() => setError('')} />}

      {loading ? (
        <div className="flex items-center gap-2 text-slate-500 text-sm p-4"><Loader2 className="h-4 w-4 animate-spin" />Cargando…</div>
      ) : incidents.length === 0 ? (
        <div className="text-slate-500 text-sm p-6 border rounded-xl text-center">No hay incidentes. Usa el simulador para crear uno.</div>
      ) : (
        <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-3">
          {incidents.map((inc) => {
            const meta = inc.incident_metadata;
            return (
              <Card key={inc.id} className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => setSelected(inc)}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between"><CardTitle className="text-sm">{meta.module?.toUpperCase() ?? 'EVENTO'}</CardTitle><SeverityBadge level={priorityToSeverity(meta.priority)} /></div>
                  <CardDescription className="text-xs flex items-center gap-1"><Clock className="h-3 w-3" />{formatTime(inc.created_at)} · {meta.camera_id ?? '–'}</CardDescription>
                </CardHeader>
                <CardContent className="text-xs text-slate-500">Regla: {meta.rule_triggered ?? '–'}</CardContent>
                <CardFooter><Button variant="ghost" size="sm" className="gap-1 text-xs">Ver detalle <ChevronRight className="h-3 w-3" /></Button></CardFooter>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── Residents ────────────────────────────────────────────────────────────────

function Residents() {
  const [persons, setPersons] = useState<ApiPerson[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState('');
  const [createSuccess, setCreateSuccess] = useState('');
  const [form, setForm] = useState({ full_name: '', person_type: 'RESIDENT' });
  const [enrollFiles, setEnrollFiles] = useState<FileList | null>(null);
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
  useEffect(() => { load(); }, []);

  const createPerson = async () => {
    if (!form.full_name.trim()) return;
    setCreating(true);
    setCreateError('');
    setCreateSuccess('');
    try {
      await apiFetch('/persons/', { method: 'POST', body: JSON.stringify(form) });
      setCreateSuccess(`✓ "${form.full_name}" registrado correctamente`);
      setForm({ full_name: '', person_type: 'RESIDENT' });
      load();
    } catch (e: any) {
      setCreateError(`Error al registrar: ${e.message}`);
    } finally {
      setCreating(false);
    }
  };

  const enrollBiometrics = async (personId: string) => {
    if (!enrollFiles || enrollFiles.length === 0) return;
    setEnrolling(true);
    setEnrollMsg('');
    setEnrollError('');
    const fd = new FormData();
    for (let i = 0; i < Math.min(enrollFiles.length, 3); i++) fd.append('files', enrollFiles[i]);
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
      setEnrollTarget(null);
      setEnrollFiles(null);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader><CardTitle>Personas enroladas</CardTitle><CardDescription>Registro biométrico para reconocimiento facial</CardDescription></CardHeader>
        <CardContent className="grid md:grid-cols-2 gap-6">

          {/* Lista */}
          <div className="space-y-3">
            <Label>Personas registradas</Label>
            {loadError && <ErrorBanner msg={loadError} onClose={() => setLoadError('')} />}
            {enrollMsg && <SuccessBanner msg={enrollMsg} onClose={() => setEnrollMsg('')} />}
            {enrollError && <ErrorBanner msg={enrollError} onClose={() => setEnrollError('')} />}
            {loading ? (
              <div className="flex items-center gap-2 text-slate-500 text-sm"><Loader2 className="h-4 w-4 animate-spin" />Cargando…</div>
            ) : persons.length === 0 ? (
              <div className="text-slate-500 text-sm border rounded-lg p-3">No hay personas registradas.</div>
            ) : (
              <ScrollArea className="h-72">
                <div className="space-y-2 pr-2">
                  {persons.map((p) => (
                    <div key={p.id} className="rounded-xl border p-3 bg-white">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">{p.full_name}</div>
                          <div className="text-xs text-slate-500">{p.person_type} · {formatTime(p.created_at)}</div>
                        </div>
                        <Button size="sm" variant="outline" onClick={() => { setEnrollTarget(p.id); setEnrollMsg(''); setEnrollError(''); }}>Enrolar</Button>
                      </div>
                      {enrollTarget === p.id && (
                        <div className="mt-3 space-y-2 border-t pt-3">
                          <Label className="text-xs">Subir 1–3 fotos del rostro (JPG/PNG)</Label>
                          <input
                            type="file"
                            accept="image/jpeg,image/png"
                            multiple
                            className="text-xs w-full"
                            onChange={(e) => setEnrollFiles(e.target.files)}
                          />
                          <div className="flex gap-2 mt-1">
                            <Button size="sm" disabled={enrolling || !enrollFiles || enrollFiles.length === 0} onClick={() => enrollBiometrics(p.id)} className="gap-2">
                              {enrolling ? <Loader2 className="h-3 w-3 animate-spin" /> : <UserPlus className="h-3 w-3" />}
                              {enrolling ? 'Procesando…' : 'Confirmar'}
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => setEnrollTarget(null)}>Cancelar</Button>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </div>

          {/* Formulario */}
          <div className="space-y-3">
            <Label>Registrar nueva persona</Label>
            {createError && <ErrorBanner msg={createError} onClose={() => setCreateError('')} />}
            {createSuccess && <SuccessBanner msg={createSuccess} onClose={() => setCreateSuccess('')} />}
            <div>
              <Label className="text-xs text-slate-500">Nombre completo</Label>
              <Input
                placeholder="Ej: Ana García"
                value={form.full_name}
                onChange={(e) => setForm({ ...form, full_name: e.target.value })}
                onKeyDown={(e) => e.key === 'Enter' && createPerson()}
              />
            </div>
            <div>
              <Label className="text-xs text-slate-500">Tipo</Label>
              <select
                className="w-full border rounded-md px-3 py-2 text-sm mt-1"
                value={form.person_type}
                onChange={(e) => setForm({ ...form, person_type: e.target.value })}
              >
                <option value="RESIDENT">Residente</option>
                <option value="VISITOR">Visitante</option>
                <option value="STAFF">Personal</option>
              </select>
            </div>
            <Button
              className="w-full gap-2"
              disabled={creating || !form.full_name.trim()}
              onClick={createPerson}
            >
              {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <UserPlus className="h-4 w-4" />}
              {creating ? 'Registrando…' : 'Registrar persona'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ─── Access Gate ──────────────────────────────────────────────────────────────

function AccessGate() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader><CardTitle>Acceso – Torre A</CardTitle><CardDescription>Verificación facial en vivo</CardDescription></CardHeader>
        <CardContent className="space-y-3">
          <div className="aspect-video rounded-xl bg-slate-200 grid place-items-center text-slate-500">
            <Camera className="h-8 w-8" /><p className="text-xs">Frente de acceso (conectar cámara IP)</p>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            {['Iluminación', 'Enfoque', 'Alineación'].map((k, i) => (
              <div key={k} className="space-y-1">
                <div className="flex items-center justify-between"><span>{k}</span><span>{70 + i * 10}%</span></div>
                <Progress value={70 + i * 10} />
              </div>
            ))}
          </div>
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button variant="default" className="gap-2"><ShieldCheck className="h-4 w-4" />Permitir</Button>
          <Button variant="destructive" className="gap-2"><CircleX className="h-4 w-4" />Denegar</Button>
        </CardFooter>
      </Card>
      <Card>
        <CardHeader><CardTitle>Resultado</CardTitle><CardDescription>Comparación por similitud coseno (ArcFace)</CardDescription></CardHeader>
        <CardContent className="space-y-2 text-sm">
          {[['Nombre', '—'], ['Similitud', '0.41'], ['Umbral', '0.52']].map(([l, v]) => (
            <div key={l} className="flex justify-between"><span>{l}</span><span className="font-medium">{v}</span></div>
          ))}
          <div className="flex justify-between"><span>Veredicto</span><Badge variant="outline" className="rounded-full">Desconocido</Badge></div>
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button variant="outline">Registrar visitante</Button>
          <Button variant="outline">Crear incidente</Button>
        </CardFooter>
      </Card>
    </div>
  );
}

// ─── Settings ─────────────────────────────────────────────────────────────────

function SettingsPanel() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Card>
        <CardHeader><CardTitle>General</CardTitle><CardDescription>Preferencias del sistema</CardDescription></CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div><div className="font-medium">Notificaciones push</div><div className="text-xs text-slate-500">Enviar a residentes y guardias</div></div>
            <Switch defaultChecked />
          </div>
          <div>
            <Label>Zona horaria</Label>
            <select className="w-full border rounded-md px-3 py-2 text-sm mt-1">
              <option value="America/Mexico_City">America/Mexico_City</option>
              <option value="UTC">UTC</option>
            </select>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Reconocimiento facial</CardTitle><CardDescription>Umbrales y calidad</CardDescription></CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1"><Label>Umbral de similitud</Label><Input type="number" defaultValue={0.52} step="0.01" /><p className="text-xs text-slate-500">Mayor = más estricto</p></div>
          <div className="space-y-1"><Label>Normalización</Label>
            <select className="w-full border rounded-md px-3 py-2 text-sm mt-1">
              <option value="ArcFace">Hiperesfera (ArcFace)</option>
              <option value="CosFace">CosFace</option>
            </select>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Análisis corporal</CardTitle><CardDescription>Acciones y ventanas</CardDescription></CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1"><Label>Ventana temporal (frames)</Label><Input type="number" defaultValue={64} /></div>
          <div className="space-y-1"><Label>Clases monitoreadas</Label>
            <div className="flex flex-wrap gap-1">{['Empujón', 'Golpe', 'Caída', 'Cuchillo', 'Pistola'].map((c) => <Badge key={c} variant="outline" className="rounded-full">{c}</Badge>)}</div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
