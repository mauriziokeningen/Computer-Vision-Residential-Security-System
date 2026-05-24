import React, { useState, useEffect, useCallback } from 'react';
import { CheckCircle2, Clock, RefreshCw, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { Label } from '../../../components/ui/label';
import { ErrorBanner } from '../../../components/ui/banner';
import { StatusBadge } from '../../../components/ui/status-badges';
import { ApiAlert, AlertStatus } from '../../../types';
import { apiFetch, buildWsUrl } from '../../../api/client';
import { formatTime } from '../../../lib/format';
import { useLanguage } from '../../../i18n/LanguageContext';
import { translateAlertMessage } from '../../../lib/translateAlert';

export function Alerts({ query = '' }: { query?: string }) {
  const { t, locale } = useLanguage();
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
    let ws: WebSocket;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let isMounted = true;
    const connect = () => {
      if (!isMounted) return;
      ws = new WebSocket(buildWsUrl('/ws/alerts'));
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.event_type === 'NEW_ALERT' || m.event_type === 'ALERT_STATUS_CHANGED') load();
        } catch {}
      };
      ws.onclose = () => { if (isMounted) reconnectTimeout = setTimeout(connect, 3000); };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => { isMounted = false; clearTimeout(reconnectTimeout); if (ws) ws.close(); };
  }, [load]);

  const updateStatus = async (id: string, status: AlertStatus) => {
    setUpdating(id);
    setUpdateError('');
    try {
      await apiFetch(`/alerts/${id}/status`, { method: 'PATCH', body: JSON.stringify({ status }) });
      load();
    } catch (e: any) {
      setUpdateError(`${t.alerts.errorUpdating}: ${e.message}`);
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
      {error && <ErrorBanner msg={`${t.alerts.errorLoading}: ${error}`} onClose={() => setError('')} />}
      {updateError && <ErrorBanner msg={updateError} onClose={() => setUpdateError('')} />}

      <div className="flex flex-col md:flex-row gap-3 md:items-end">
        <div className="flex-1">
          <Label>{t.alerts.filterLabel}</Label>
          <select className="w-full border rounded-md px-3 py-2 text-sm mt-1" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">{t.alerts.filterAll}</option>
            <option value="UNREAD">{t.alerts.filterUnread}</option>
            <option value="ACKNOWLEDGED">{t.alerts.filterAcknowledged}</option>
            <option value="RESOLVED">{t.alerts.filterResolved}</option>
          </select>
        </div>
        <Button variant="secondary" className="gap-2" onClick={load}>
          <RefreshCw className="h-4 w-4" />{t.alerts.refresh}
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-slate-500 text-sm p-4">
          <Loader2 className="h-4 w-4 animate-spin" />{t.alerts.loading}
        </div>
      ) : filteredAlerts.length === 0 ? (
        <div className="text-slate-500 text-sm p-6 border rounded-xl text-center">
          {alerts.length === 0 ? t.alerts.emptyFilter : t.alerts.emptySearch}
        </div>
      ) : (
        <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-3">
          {filteredAlerts.map((a) => (
            <Card key={a.id}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <StatusBadge status={a.status} />
                  <CardDescription className="text-xs">
                    <Clock className="h-3 w-3 inline mr-1" />{formatTime(a.created_at)}
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent className="text-sm text-slate-700">
                {translateAlertMessage(a.message, locale)}
              </CardContent>
              <CardFooter className="flex gap-2 flex-wrap">
                {a.status === 'UNREAD' && (
                  <Button size="sm" variant="outline" disabled={updating === a.id} onClick={() => updateStatus(a.id, 'ACKNOWLEDGED')} className="gap-2">
                    {updating === a.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                    {t.alerts.acknowledge}
                  </Button>
                )}
                {a.status === 'ACKNOWLEDGED' && (
                  <Button size="sm" variant="outline" disabled={updating === a.id} onClick={() => updateStatus(a.id, 'RESOLVED')} className="gap-2">
                    {updating === a.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                    {t.alerts.resolve}
                  </Button>
                )}
                {a.status === 'RESOLVED' && (
                  <span className="text-xs text-slate-400 self-center">{t.alerts.resolved}</span>
                )}
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}