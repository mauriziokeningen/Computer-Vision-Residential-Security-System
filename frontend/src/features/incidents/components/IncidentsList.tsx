import React, { useState, useEffect, useCallback } from 'react';
import { Bell, ChevronRight, Clock, Database, RefreshCw, Loader2, XCircle, Wand2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { ErrorBanner, SuccessBanner } from '../../../components/ui/banner';
import { SeverityBadge } from '../../../components/ui/status-badges';
import { ApiIncident, EvidenceFile } from '../../../types';
import { apiFetch } from '../../../api/client';
import { formatTime, priorityToSeverity } from '../../../lib/format';
import { useLanguage } from '../../../i18n/LanguageContext';

export function IncidentsList({ query = '', lastIncidentEvent = 0 }: { query?: string; lastIncidentEvent?: number }) {
  const { t } = useLanguage();
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

  const load = useCallback((showLoader = false) => {
    if (showLoader) setLoading(true);
    setError('');
    apiFetch<ApiIncident[]>('/incidents/?limit=30')
      .then(setIncidents)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(true); }, [load]);

  useEffect(() => {
    if (lastIncidentEvent > 0) load(false);
  }, [lastIncidentEvent, load]);

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
        if (!files.length) { setEvidenceUrl(''); return; }
        const preferred = files.find((file) => /\.(jpg|jpeg|png|webp|mp4|webm)$/i.test(file.object_name)) ?? files[0];
        setSelectedEvidenceName(preferred.object_name);
        const urlData = await apiFetch<{ url: string }>(`/evidence/url?object_name=${encodeURIComponent(preferred.object_name)}`);
        setEvidenceUrl(urlData.url);
      })
      .catch((e) => setEvidenceError(e.message))
      .finally(() => setEvidenceLoading(false));
  }, [selected]);

  const simulate = async (module: string) => {
    setSimulating(module);
    setSimError('');
    setSimSuccess('');
    const detections = module === 'face'
      ? [{ name: 'unknown_person', confidence: 0.85 }]
      : module === 'weapons'
      ? [{ class: 'knife', confidence: 0.91 }]
      : [{ action: 'punch', confidence: 0.78 }];
    try {
      const result = await apiFetch<any>('/incidents/simulate', {
        method: 'POST',
        body: JSON.stringify({ module, camera_id: 'cam-demo-01', detections }),
      });
      setSimSuccess(`✓ ${t.incidents.simulateTitle} — ${result.rule_triggered} · ${result.priority}`);
    } catch (e: any) {
      setSimError(`Error: ${e.message}`);
    } finally {
      setSimulating(null);
    }
  };

  const SIM_MODULES: [string, string][] = [
    ['face', t.incidents.simFace],
    ['weapons', t.incidents.simWeapons],
    ['pose', t.incidents.simPose],
  ];

  const normalizedQuery = query.trim().toLowerCase();
  const filteredIncidents = incidents.filter((incident) => {
    const meta = incident.incident_metadata ?? {};
    const haystack = [meta.module, meta.rule_triggered, meta.priority, meta.camera_id, incident.id]
      .filter(Boolean).join(' ').toLowerCase();
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
                <CardTitle>{t.incidents.title}</CardTitle>
                <CardDescription>{meta.camera_id} · {formatTime(selected.created_at)}</CardDescription>
              </div>
              <Button variant="ghost" onClick={() => setSelected(null)}>{t.incidents.back}</Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {evidenceLoading ? (
              <div className="aspect-video rounded-xl border bg-slate-50 grid place-items-center text-slate-500">
                <div className="flex items-center gap-2 text-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />{t.incidents.loadingEvidence}
                </div>
              </div>
            ) : evidenceError ? (
              <div className="aspect-video rounded-xl border border-red-200 bg-red-50 grid place-items-center text-red-700 p-6 text-center">
                <div className="space-y-2">
                  <XCircle className="h-8 w-8 mx-auto" />
                  <p className="font-medium">{t.incidents.evidenceError}</p>
                  <p className="text-sm">{evidenceError}</p>
                </div>
              </div>
            ) : evidenceUrl && isImage ? (
              <div className="aspect-video rounded-xl border overflow-hidden bg-black">
                <img src={evidenceUrl} alt="evidence" className="h-full w-full object-contain" />
              </div>
            ) : evidenceUrl && isVideo ? (
              <div className="aspect-video rounded-xl border overflow-hidden bg-black">
                <video src={evidenceUrl} controls className="h-full w-full object-contain" />
              </div>
            ) : (
              <div className="aspect-video rounded-xl border border-dashed bg-slate-50 grid place-items-center text-slate-500 p-6 text-center">
                <div className="space-y-2">
                  <Database className="h-8 w-8 mx-auto" />
                  <p className="font-medium">{t.incidents.noEvidence}</p>
                  <p className="text-sm">{t.incidents.noEvidenceDesc}</p>
                </div>
              </div>
            )}
            <div className="grid md:grid-cols-3 gap-3 text-xs">
              {[
                [t.incidents.labelModule, meta.module],
                [t.incidents.labelRule, meta.rule_triggered],
                [t.incidents.labelPriority, meta.priority],
              ].map(([k, v]) => (
                <div key={k} className="rounded-lg border p-2">
                  <div className="font-medium">{k}</div>
                  <div className="text-slate-500">{v ?? '–'}</div>
                </div>
              ))}
            </div>
          </CardContent>
          <CardFooter className="flex gap-2">
            <Button className="gap-2"><Bell className="h-4 w-4" />{t.incidents.notify}</Button>
            <Button variant="outline" className="gap-2"><Database className="h-4 w-4" />{t.incidents.saveEvidence}</Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader><CardTitle>{t.incidents.metadata}</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span>{t.incidents.priority}</span>
              <SeverityBadge level={priorityToSeverity(meta.priority)} />
            </div>
            <div className="flex justify-between">
              <span>{t.incidents.camera}</span>
              <span>{meta.camera_id ?? '–'}</span>
            </div>
            <div className="flex justify-between">
              <span>ID</span>
              <span className="text-xs text-slate-400">{selected.id.slice(0, 8)}…</span>
            </div>
            <div>
              <div className="font-medium mb-2">{t.incidents.evidenceObjects}</div>
              {evidenceFiles.length === 0 ? (
                <div className="text-xs text-slate-500">{t.incidents.noEvidence}</div>
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
          <h2 className="text-lg font-semibold">{t.incidents.title}</h2>
          <p className="text-sm text-slate-500">{t.incidents.subtitle}</p>
        </div>
        <Button variant="outline" className="gap-2" onClick={() => load(true)}>
          <RefreshCw className="h-4 w-4" />{t.incidents.refresh}
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t.incidents.simulateTitle}</CardTitle>
          <CardDescription>{t.incidents.simulateDesc}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {simError && <ErrorBanner msg={simError} onClose={() => setSimError('')} />}
          {simSuccess && <SuccessBanner msg={simSuccess} onClose={() => setSimSuccess('')} />}
          <div className="flex gap-3 flex-wrap">
            {SIM_MODULES.map(([m, label]) => (
              <Button key={m} variant="outline" size="sm" disabled={simulating !== null} onClick={() => simulate(m)} className="gap-2">
                {simulating === m ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wand2 className="h-3 w-3" />}
                {simulating === m ? t.incidents.simulating : label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {error && <ErrorBanner msg={error} onClose={() => setError('')} />}

      {loading ? (
        <div className="flex items-center gap-2 text-slate-500 text-sm p-4">
          <Loader2 className="h-4 w-4 animate-spin" />{t.incidents.loading}
        </div>
      ) : filteredIncidents.length === 0 ? (
        <div className="text-slate-500 text-sm p-6 border rounded-xl text-center">
          {incidents.length === 0 ? t.incidents.empty : t.incidents.emptySearch}
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
                    <Clock className="h-3 w-3" />{formatTime(inc.created_at)} · {meta.camera_id ?? '–'}
                  </CardDescription>
                </CardHeader>
                <CardContent className="text-xs text-slate-500">
                  {t.incidents.rule}: {meta.rule_triggered ?? '–'}
                </CardContent>
                <CardFooter>
                  <Button variant="ghost" size="sm" className="gap-1 text-xs">
                    {t.incidents.viewDetail} <ChevronRight className="h-3 w-3" />
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