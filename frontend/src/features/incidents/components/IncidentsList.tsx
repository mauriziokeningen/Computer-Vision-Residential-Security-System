import React, { useState, useEffect, useCallback } from 'react';
import {
  Bell,
  ChevronRight,
  Clock,
  Database,
  RefreshCw,
  Loader2,
  XCircle,
  Wand2,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { ErrorBanner, SuccessBanner } from '../../../components/ui/banner';
import { SeverityBadge } from '../../../components/ui/status-badges';

import { ApiIncident, EvidenceFile } from '../../../types';
import { apiFetch } from '../../../api/client';
import { formatTime, priorityToSeverity } from '../../../lib/format';

export function IncidentsList({ query = '', lastIncidentEvent = 0 }: { query?: string; lastIncidentEvent?: number }) {
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

  // Carga inicial
  useEffect(() => {
    load(true);
  }, [load]);

  // Re-fetch cuando el WebSocket global detecta NEW_ALERT
  // Esto dispara en TODAS las tabs simultáneamente — fix del bug #85
  useEffect(() => {
    if (lastIncidentEvent > 0) {
      load(false);
    }
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
      // ❌ load() removido — Tab A ahora espera el WebSocket igual que Tab B
      // Esto garantiza sincronización real entre tabs, no optimistic update local
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
        <Button variant="outline" className="gap-2" onClick={() => load(true)}>
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