import React from 'react';
import {
  AlertTriangle,
  Bell,
  Camera,
  DoorOpen,
  Gauge,
  ListChecks,
  Settings,
  Siren,
  Users,
  Search,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { AlertCounts } from '../types';

export type TabId =
  | 'dashboard'
  | 'live'
  | 'alerts'
  | 'incidents'
  | 'residents'
  | 'access'
  | 'settings';

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

export function SidebarLayout({
  tab,
  setTab,
  searchQuery,
  setSearchQuery,
  alertCounts,
  wsConnected,
  children,
}: {
  tab: TabId;
  setTab: (t: TabId) => void;
  searchQuery: string;
  setSearchQuery: (s: string) => void;
  alertCounts: AlertCounts;
  wsConnected: boolean;
  children: React.ReactNode;
}) {
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
                      onClick={() => setTab(it.id as TabId)}
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

          <div>{children}</div>
        </div>
      </div>
    </div>
  );
}

