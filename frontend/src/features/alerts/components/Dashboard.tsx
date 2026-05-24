import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, ShieldCheck, CircleX, Clock, Cpu, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../../components/ui/card';
import { Progress } from '../../../components/ui/progress';
import { ErrorBanner } from '../../../components/ui/banner';
import { KPI } from '../../../components/ui/kpi';
import { StatusBadge } from '../../../components/ui/status-badges';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { ApiAlert, AlertCounts } from '../../../types';
import { apiFetch } from '../../../api/client';
import { parseApiDate, formatTime } from '../../../lib/format';
import { useLanguage } from '../../../i18n/LanguageContext';
import { translateAlertMessage } from '../../../lib/translateAlert';

export function Dashboard({ alertCounts }: { alertCounts: AlertCounts }) {
  const { t, locale } = useLanguage();
  const [recentAlerts, setRecentAlerts] = useState<ApiAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [statsSeries, setStatsSeries] = useState<{ t: string; a: number }[]>([]);

  const buildHourlySeries = useCallback((alerts: ApiAlert[]) => {
    const now = new Date();
    const buckets = Array.from({ length: 8 }, (_, i) => {
      const start = new Date(now);
      start.setMinutes(0, 0, 0);
      start.setHours(start.getHours() - (7 - i));
      return {
        startMs: start.getTime(),
        endMs: start.getTime() + 60 * 60 * 1000,
        t: start.toLocaleTimeString('es-MX', { hour: '2-digit', hour12: false }),
        a: 0,
      };
    });
    alerts.forEach((alert) => {
      const alertMs = parseApiDate(alert.created_at).getTime();
      const bucket = buckets.find((b) => alertMs >= b.startMs && alertMs < b.endMs);
      if (bucket) bucket.a += 1;
    });
    setStatsSeries(buckets.map(({ t, a }) => ({ t, a })));
  }, []);

  const loadDashboardAlerts = useCallback(async (showLoader = false) => {
    if (showLoader) setLoading(true);
    setError('');
    try {
      const data = await apiFetch<ApiAlert[]>('/alerts/?limit=100');
      setRecentAlerts(data);
      buildHourlySeries(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [buildHourlySeries]);

  useEffect(() => { loadDashboardAlerts(true); }, [loadDashboardAlerts]);
  useEffect(() => {
    loadDashboardAlerts(false);
  }, [alertCounts.unread, alertCounts.acknowledged, alertCounts.resolved, loadDashboardAlerts]);

  const total = alertCounts.unread + alertCounts.acknowledged + alertCounts.resolved;

  return (
    <div className="space-y-4">
      {error && <ErrorBanner msg={`${t.dashboard.errorLoading}: ${error}`} onClose={() => setError('')} />}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPI icon={AlertTriangle} label={t.dashboard.kpiUnread} value={String(alertCounts.unread)} sub={t.dashboard.kpiUnreadSub} />
        <KPI icon={ShieldCheck} label={t.dashboard.kpiAcknowledged} value={String(alertCounts.acknowledged)} sub={t.dashboard.kpiAcknowledgedSub} />
        <KPI icon={CircleX} label={t.dashboard.kpiResolved} value={String(alertCounts.resolved)} sub={t.dashboard.kpiResolvedSub} />
        <KPI icon={Cpu} label={t.dashboard.kpiTotal} value={String(total)} sub={t.dashboard.kpiTotalSub} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle>{t.dashboard.trendTitle}</CardTitle>
            <CardDescription>{t.dashboard.trendDesc}</CardDescription>
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
            <CardTitle>{t.dashboard.statusTitle}</CardTitle>
            <CardDescription>{t.dashboard.statusDesc}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {[
              { label: t.dashboard.kpiUnread, value: alertCounts.unread, color: 'bg-red-500' },
              { label: t.dashboard.kpiAcknowledged, value: alertCounts.acknowledged, color: 'bg-amber-500' },
              { label: t.dashboard.kpiResolved, value: alertCounts.resolved, color: 'bg-emerald-500' },
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
          <CardTitle>{t.dashboard.recentTitle}</CardTitle>
          <CardDescription>{t.dashboard.recentDesc}</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center gap-2 text-slate-500 text-sm p-4">
              <Loader2 className="h-4 w-4 animate-spin" />{t.dashboard.loading}
            </div>
          ) : recentAlerts.length === 0 ? (
            <div className="text-slate-500 text-sm p-4">{t.dashboard.empty}</div>
          ) : (
            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
              {recentAlerts.slice(0, 6).map((a) => {
                const msg = translateAlertMessage(a.message, locale);
                return (
                  <motion.div key={a.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
                    <Card className="border-l-4" style={{
                      borderLeftColor: a.status === 'UNREAD' ? '#ef4444' : a.status === 'ACKNOWLEDGED' ? '#f59e0b' : '#10b981',
                    }}>
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between">
                          <StatusBadge status={a.status} />
                          <CardDescription className="text-xs flex items-center gap-1">
                            <Clock className="h-3 w-3" />{formatTime(a.created_at)}
                          </CardDescription>
                        </div>
                      </CardHeader>
                      <CardContent className="pt-0 text-xs text-slate-600">
                        {msg.length > 60 ? msg.slice(0, 60) + '…' : msg}
                      </CardContent>
                    </Card>
                  </motion.div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}