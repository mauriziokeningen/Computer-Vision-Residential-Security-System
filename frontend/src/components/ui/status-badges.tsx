import { Badge } from './badge';
import { AlertStatus } from '../../types';

export function SeverityBadge({ level }: { level: 'low' | 'high' | 'critical' }) {
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

export function StatusBadge({ status }: { status: AlertStatus }) {
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

