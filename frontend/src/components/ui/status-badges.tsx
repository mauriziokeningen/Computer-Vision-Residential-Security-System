import { Badge } from './badge';
import { AlertStatus } from '../../types';
import { useLanguage } from '../../i18n/LanguageContext';

export function SeverityBadge({ level }: { level: 'low' | 'high' | 'critical' }) {
  const { locale } = useLanguage();

  const map: Record<string, string> = {
    low: 'bg-slate-100 text-slate-800',
    high: 'bg-amber-100 text-amber-800',
    critical: 'bg-red-100 text-red-800',
  };

  const label: Record<string, Record<string, string>> = {
    es: { low: 'Baja', high: 'Alta', critical: 'Crítica' },
    en: { low: 'Low',  high: 'High', critical: 'Critical' },
  };

  return (
    <Badge className={`${map[level]} rounded-full`}>
      {label[locale][level]}
    </Badge>
  );
}

export function StatusBadge({ status }: { status: AlertStatus }) {
  const { locale } = useLanguage();

  const map: Record<AlertStatus, string> = {
    UNREAD:       'bg-red-100 text-red-800',
    ACKNOWLEDGED: 'bg-amber-100 text-amber-800',
    RESOLVED:     'bg-emerald-100 text-emerald-800',
  };

  const label: Record<string, Record<AlertStatus, string>> = {
    es: { UNREAD: 'Sin leer', ACKNOWLEDGED: 'Atendida', RESOLVED: 'Resuelta' },
    en: { UNREAD: 'Unread',   ACKNOWLEDGED: 'Acknowledged', RESOLVED: 'Resolved' },
  };

  return (
    <Badge className={`${map[status]} rounded-full`}>
      {label[locale][status]}
    </Badge>
  );
}