import React, { useState } from 'react';
import WebcamFeed from './components/WebcamFeed';
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
  Eye,
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

// --- Fake data ---
type AlertItem = {
  id: string;
  type: string;
  time: string;
  location: string;
  severity: 'low' | 'high' | 'critical';
  details: string;
  modules: string[];
  status: string;
};

const alerts: AlertItem[] = [
  {
    id: 'AL-01782',
    type: 'Desconocido',
    time: '15:32',
    location: 'Acceso Torre A',
    severity: 'high',
    details: 'Rostro no registrado. Intento de acceso.',
    modules: ['Facial'],
    status: 'Pendiente',
  },
  {
    id: 'AL-01781',
    type: 'Pelea',
    time: '14:58',
    location: 'Patio central',
    severity: 'critical',
    details: 'Golpes y empujones detectados.',
    modules: ['Pose'],
    status: 'En curso',
  },
  {
    id: 'AL-01780',
    type: 'Arma (cuchillo)',
    time: '14:05',
    location: 'Estacionamiento B2',
    severity: 'critical',
    details: 'Detección de objeto punzocortante.',
    modules: ['Objetos'],
    status: 'Escalada',
  },
  {
    id: 'AL-01779',
    type: 'Acceso permitido',
    time: '13:41',
    location: 'Acceso Torre B',
    severity: 'low',
    details: 'Residente: D. Orozco.',
    modules: ['Facial'],
    status: 'Cerrado',
  },
];

const statsSeries = [
  { t: '08', a: 2 },
  { t: '09', a: 1 },
  { t: '10', a: 3 },
  { t: '11', a: 2 },
  { t: '12', a: 4 },
  { t: '13', a: 2 },
  { t: '14', a: 5 },
  { t: '15', a: 3 },
];

const feedOverlays = [
  {
    label: 'Rostro: Desconocido',
    conf: 0.87,
    className: 'border-red-500 text-red-600',
  },
  {
    label: 'Pose: Empujón',
    conf: 0.81,
    className: 'border-amber-500 text-amber-600',
  },
  {
    label: 'Objeto: Cuchillo',
    conf: 0.92,
    className: 'border-red-500 text-red-600',
  },
];

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

function Header() {
  return (
    <div className="flex items-center justify-between py-4">
      <div className="flex items-center gap-3">
        <Siren className="h-7 w-7" />
        <div>
          <h1 className="text-xl font-semibold leading-none">
            Seguridad UH · Panel
          </h1>
          <p className="text-sm text-slate-500">Monitoreo en tiempo real · Prototipo</p>
        </div>
      </div>
      <div className="flex gap-2 items-center">
        <Button variant="outline" className="gap-2">
          <Search className="h-4 w-4" />
          Buscar
        </Button>
        <Button className="gap-2">
          <Bell className="h-4 w-4" />
          Notificaciones
        </Button>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<
    'dashboard' | 'live' | 'alerts' | 'incidents' | 'residents' | 'access' | 'settings'
  >('dashboard');

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-slate-50">
      <div className="max-w-7xl mx-auto p-4">
        <Header />

        <div className="grid grid-cols-1 lg:grid-cols-[240px_1fr] gap-4">
          {/* Sidebar */}
          <div className="hidden lg:block">
            <Card className="sticky top-4">
              <CardHeader>
                <CardTitle className="text-base">Navegación</CardTitle>
                <CardDescription>Secciones del prototipo</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <nav className="grid gap-1">
                  {[
                    { id: 'dashboard', icon: Gauge, label: 'Dashboard' },
                    { id: 'live', icon: Camera, label: 'Monitoreo en vivo' },
                    { id: 'alerts', icon: AlertTriangle, label: 'Alertas' },
                    { id: 'incidents', icon: ListChecks, label: 'Incidentes' },
                    { id: 'residents', icon: Users, label: 'Residentes' },
                    { id: 'access', icon: DoorOpen, label: 'Control de Acceso' },
                    { id: 'settings', icon: Settings, label: 'Configuración' },
                  ].map((it) => (
                    <Button
                      key={it.id}
                      variant={tab === it.id ? 'secondary' : 'ghost'}
                      className="justify-start gap-2"
                      onClick={() => setTab(it.id as any)}
                    >
                      <it.icon className="h-4 w-4" />
                      {it.label}
                    </Button>
                  ))}
                </nav>
              </CardContent>
            </Card>
          </div>

          {/* Content */}
          <div>
            {tab === 'dashboard' && <Dashboard />}
            {tab === 'live' && <LiveMonitoring />}
            {tab === 'alerts' && <Alerts />}
            {tab === 'incidents' && <IncidentDetail />}
            {tab === 'residents' && <Residents />}
            {tab === 'access' && <AccessGate />}
            {tab === 'settings' && <SettingsPanel />}
          </div>
        </div>
      </div>
    </div>
  );
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

function Dashboard() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPI icon={AlertTriangle} label="Alertas hoy" value="13" sub="3 críticas, 4 altas" />
        <KPI icon={ShieldCheck} label="Accesos permitidos" value="128" sub="Últimas 24 h" />
        <KPI icon={CircleX} label="Accesos bloqueados" value="12" sub="Rostros desconocidos" />
        <KPI icon={Cpu} label="Latencia promedio" value="87 ms" sub="Procesamiento en borde" />
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
                <Line type="monotone" dataKey="a" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Estado de servicios</CardTitle>
            <CardDescription>Módulos del sistema</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {[
              { name: 'Reconocimiento facial', ok: true },
              { name: 'Análisis de pose', ok: true },
              { name: 'Detección de objetos', ok: true },
              { name: 'Base de datos', ok: true },
              { name: 'Notificaciones', ok: false },
            ].map((s) => (
              <div key={s.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${s.ok ? 'bg-emerald-500' : 'bg-red-500'}`} />
                  <span>{s.name}</span>
                </div>
                <Badge variant={s.ok ? 'default' : 'destructive'}>{s.ok ? 'OK' : 'Fallo'}</Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Últimas alertas</CardTitle>
          <CardDescription>Eventos recibidos en tiempo real</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
            {alerts.map((a) => (
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
                      a.severity === 'critical'
                        ? '#ef4444'
                        : a.severity === 'high'
                        ? '#f59e0b'
                        : '#94a3b8',
                  }}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">{a.type}</CardTitle>
                      <SeverityBadge level={a.severity} />
                    </div>
                    <CardDescription className="flex items-center gap-2">
                      <Clock className="h-4 w-4" />
                      {a.time} · {a.location}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0 text-sm space-y-2">
                    <div className="text-slate-500">{a.details}</div>
                    <div className="flex flex-wrap gap-1">
                      {a.modules.map((m) => (
                        <Badge key={m} variant="outline" className="rounded-full">
                          {m}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                  <CardFooter className="pt-0">
                    <Button variant="ghost" size="sm" className="gap-2">
                      Ver detalle
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function LiveMonitoring() {
  const wsRef = React.useRef<WebSocket | null>(null);
  const [dets, setDets] = React.useState<any[]>([]);
  const [frameSize, setFrameSize] = React.useState({ width: 640, height: 480 });

  React.useEffect(() => {
    const ws = new WebSocket('ws://127.0.0.1:8000/ws');
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      setDets(data.detections ?? []);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
    };

    return () => ws.close();
  }, []);

  const handleFrame = (
    canvas: HTMLCanvasElement,
    width: number,
    height: number
  ) => {
    setFrameSize({ width, height });

    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const jpg = canvas.toDataURL('image/jpeg', 0.7);
    ws.send(JSON.stringify({ image: jpg }));
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle>
            <Video className="h-5 w-5 inline mr-2" />
            Cámara – Acceso Principal
          </CardTitle>
          <CardDescription>Detección y tracking en tiempo real</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative aspect-video rounded-xl bg-black overflow-hidden">
            <WebcamFeed className="absolute inset-0" onFrame={handleFrame} fps={8} />

            <div className="absolute inset-0 pointer-events-none">
              {dets.map((d, i) => {
                const [x1, y1, x2, y2] = d.bbox ?? [0, 0, 0, 0];

                const mirroredX1 = frameSize.width - x2;

                const left = `${(mirroredX1 / frameSize.width) * 100}%`;
                const top = `${(y1 / frameSize.height) * 100}%`;
                const width = `${((x2 - x1) / frameSize.width) * 100}%`;
                const height = `${((y2 - y1) / frameSize.height) * 100}%`;

                const name = d.name?.toLowerCase?.() ?? "";
                const isWeapon = name.includes("knife") || name.includes("pistol");

                if (!isWeapon) return null;

                return (
                  <div
                    key={i}
                    className="absolute border-2 border-red-500 shadow-[0_0_0_1px_rgba(255,255,255,0.15)]"
                    style={{ left, top, width, height }}
                  >
                    <div className="absolute -top-6 left-0 bg-red-600 text-white text-[10px] px-2 py-1 rounded">
                      {d.name} #{d.track_id ?? "-"} · {Math.round((d.conf ?? 0) * 100)}%
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="absolute top-4 left-4 space-y-2">
              {dets
                .filter((d) => {
                  const name = d.name?.toLowerCase?.() ?? "";
                  return name.includes("knife") || name.includes("pistol");
                })
                .slice(0, 4)
                .map((d, i) => (
                  <div
                    key={i}
                    className="backdrop-blur bg-white/70 border px-3 py-1 rounded-lg text-xs border-red-500 text-red-600"
                  >
                    {d.name} #{d.track_id ?? "-"} · conf {Math.round((d.conf ?? 0) * 100)}%
                  </div>
                ))}
            </div>
          </div>
        </CardContent>
        <CardFooter className="flex items-center gap-3">
          <Button variant="outline" className="gap-2">
            <Eye className="h-4 w-4" />
            Ver en pantalla completa
          </Button>
          <Button className="gap-2">
            <Bell className="h-4 w-4" />
            Enviar alerta
          </Button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Eventos recientes</CardTitle>
          <CardDescription>Detecciones de armas</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-72 pr-2">
            <div className="space-y-3">
              {dets.filter((d) => {
                const name = d.name?.toLowerCase?.() ?? "";
                return name.includes("knife") || name.includes("pistol");
              }).length === 0 ? (
                <div className="p-3 rounded-lg border bg-white text-sm text-slate-500">
                  Sin detecciones por el momento.
                </div>
              ) : (
                dets
                  .filter((d) => {
                    const name = d.name?.toLowerCase?.() ?? "";
                    return name.includes("knife") || name.includes("pistol");
                  })
                  .map((d, i) => (
                    <div key={i} className="p-3 rounded-lg border bg-white">
                      <div className="flex items-center justify-between">
                        <div className="font-medium text-sm">
                          {d.name} #{d.track_id ?? "-"}
                        </div>
                        <Badge className="bg-red-100 text-red-800 rounded-full">
                          Crítica
                        </Badge>
                      </div>
                      <div className="text-xs text-slate-500">
                        confianza: {Math.round((d.conf ?? 0) * 100)}%
                      </div>
                      <div className="text-xs mt-1">
                        Bounding box: [{d.bbox?.map((v: number) => Math.round(v)).join(", ")}]
                      </div>
                    </div>
                  ))
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

function Alerts() {
  return (
    <div className="space-y-4">
      <div className="flex flex-col md:flex-row gap-3 md:items-end">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 flex-1">
          <div>
            <Label>Severidad</Label>
            <Select defaultValue="all">
              <NativeSelect>
                <SelectItem value="all">Todas</SelectItem>
                <SelectItem value="critical">Crítica</SelectItem>
                <SelectItem value="high">Alta</SelectItem>
                <SelectItem value="low">Baja</SelectItem>
              </NativeSelect>
            </Select>
          </div>
          <div>
            <Label>Módulo</Label>
            <Select defaultValue="all">
              <NativeSelect>
                <SelectItem value="all">Todos</SelectItem>
                <SelectItem value="facial">Facial</SelectItem>
                <SelectItem value="pose">Pose</SelectItem>
                <SelectItem value="objetos">Objetos</SelectItem>
              </NativeSelect>
            </Select>
          </div>
          <div>
            <Label>Fecha</Label>
            <Input type="date" />
          </div>
        </div>
        <Button variant="secondary" className="gap-2">
          <Search className="h-4 w-4" /> Filtrar
        </Button>
      </div>

      <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-3">
        {alerts.map((a) => (
          <Card key={a.id}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">{a.type}</CardTitle>
                <SeverityBadge level={a.severity} />
              </div>
              <CardDescription>
                {a.id} · {a.time} · {a.location}
              </CardDescription>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <div className="text-slate-500">{a.details}</div>
              <div className="flex flex-wrap gap-1">
                {a.modules.map((m) => (
                  <Badge key={m} variant="outline" className="rounded-full">
                    {m}
                  </Badge>
                ))}
              </div>
            </CardContent>
            <CardFooter className="flex gap-2">
              <Button size="sm" variant="outline" className="gap-2">
                <CheckCircle2 className="h-4 w-4" /> Atender
              </Button>
              <Button size="sm" variant="outline" className="gap-2">
                <Wand2 className="h-4 w-4" /> Generar reporte
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}

function IncidentDetail() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle>Incidente · AL-01781</CardTitle>
          <CardDescription>Pelea detectada – Patio central – 14:58</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="aspect-video rounded-xl bg-slate-200 grid place-items-center text-slate-500">
            <PersonStanding className="h-10 w-10 opacity-70" />
            <p className="text-sm">Clip recortado (8s) · Esqueleto superpuesto</p>
          </div>
          <div className="grid md:grid-cols-3 gap-3">
            {['Empujón', 'Caída', 'Golpe'].map((t, i) => (
              <div key={i} className="rounded-lg border p-2 text-xs">
                <div className="font-medium">{t}</div>
                <div className="text-slate-500">conf {80 - i * 7}%</div>
              </div>
            ))}
          </div>
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button className="gap-2" variant="default">
            <Bell className="h-4 w-4" /> Notificar a seguridad
          </Button>
          <Button className="gap-2" variant="outline">
            <Database className="h-4 w-4" /> Guardar evidencia
          </Button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Resumen</CardTitle>
          <CardDescription>Metadatos del evento</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span>Severidad</span>
            <SeverityBadge level="critical" />
          </div>
          <div className="flex justify-between">
            <span>Duración</span>
            <span>00:08</span>
          </div>
          <div className="flex justify-between">
            <span>Personas</span>
            <span>2</span>
          </div>
          <div className="flex justify-between">
            <span>Cámara</span>
            <span>Patio-03</span>
          </div>
          <div className="flex justify-between">
            <span>Estado</span>
            <span className="text-amber-700">En curso</span>
          </div>
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button size="sm" variant="outline">
            Asignar
          </Button>
          <Button size="sm" variant="outline">
            Cerrar
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}

function Residents() {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Residentes</CardTitle>
          <CardDescription>Enrolamiento y lista</CardDescription>
        </CardHeader>
        <CardContent className="grid md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <Label>Buscar</Label>
            <Input placeholder="Nombre, depto, ID" />
            <div className="rounded-xl border p-3 bg-white">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">David Orozco</div>
                  <div className="text-xs text-slate-500">Depto B-302 · 2 registros</div>
                </div>
                <Button size="sm" variant="outline">
                  Ver
                </Button>
              </div>
            </div>
            <div className="rounded-xl border p-3 bg-white">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Armando García</div>
                  <div className="text-xs text-slate-500">Depto A-101 · 3 registros</div>
                </div>
                <Button size="sm" variant="outline">
                  Ver
                </Button>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <Label>Nuevo enrolamiento</Label>
            <div className="aspect-video rounded-xl bg-slate-200 grid place-items-center text-slate-500">
              <UserPlus className="h-8 w-8" />
              <p className="text-xs">Coloque el rostro frente a la cámara</p>
            </div>
            <Button className="w-full">Capturar rostro</Button>
            <div className="grid grid-cols-2 gap-2">
              <Input placeholder="Nombre" />
              <Input placeholder="Departamento" />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function AccessGate() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Acceso – Torre A</CardTitle>
          <CardDescription>Verificación facial en vivo</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="aspect-video rounded-xl bg-slate-200 grid place-items-center text-slate-500">
            <Camera className="h-8 w-8" />
            <p className="text-xs">Frente de acceso</p>
          </div>
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
            <ShieldCheck className="h-4 w-4" /> Permitir
          </Button>
          <Button variant="destructive" className="gap-2">
            <CircleX className="h-4 w-4" /> Denegar
          </Button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Resultado</CardTitle>
          <CardDescription>Comparación por similitud coseno</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span>Nombre</span>
            <span className="font-medium">—</span>
          </div>
          <div className="flex justify-between">
            <span>Similitud</span>
            <span className="font-medium">0.41</span>
          </div>
          <div className="flex justify-between">
            <span>Umbral</span>
            <span>0.52</span>
          </div>
          <div className="flex justify-between">
            <span>Veredicto</span>
            <Badge variant="outline" className="rounded-full">
              Desconocido
            </Badge>
          </div>
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button variant="outline">Registrar visitante</Button>
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
