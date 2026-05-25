import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  UserPlus,
  Loader2,
  Camera,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Label } from '../../../components/ui/label';
import { ScrollArea } from '../../../components/ui/scroll-area';
import { Badge } from '../../../components/ui/badge';
import { ErrorBanner, SuccessBanner } from '../../../components/ui/banner';

import { ApiPerson } from '../../../types';
import { apiFetch } from '../../../api/client';
import WebcamFeed from './WebcamFeed';

// Pose hints for the in-browser camera capture path. The backend treats the
// three files as an unordered set; these labels are UI affordances only, to
// nudge the operator into varying head angles for a richer master vector.
const POSE_HINTS = ['Front', 'Left', 'Right'] as const;

export function Residents({ query = '' }: { query?: string }) {
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
  const [isCapturing, setIsCapturing] = useState(false);

  // Toggle between the legacy file-upload path and the in-browser camera
  // capture path. Both produce the same File[] payload, so the submission
  // pipeline downstream is unchanged.
  const [captureMode, setCaptureMode] = useState<'upload' | 'camera'>('upload');

  // Latest live frame from WebcamFeed. WebcamFeed paints its hidden canvas
  // continuously; we keep a pointer here so captureFromCamera can snapshot
  // the most recent frame synchronously on click.
  const latestCanvasRef = useRef<HTMLCanvasElement | null>(null);

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

  // WebcamFeed pushes its hidden canvas here on every painted frame. Storing
  // the reference (not the pixel data) keeps this callback O(1) and stable
  // across re-renders.
  const handleFrame = useCallback((canvas: HTMLCanvasElement) => {
    latestCanvasRef.current = canvas;
  }, []);

  // Snapshot the latest webcam frame into a JPEG File and drop it into the
  // first empty slot. Quality 0.92 matches the backend evidence pipeline and
  // is well above the threshold where ArcFace landmark extraction degrades.
  const captureFromCamera = () => {

    if (isCapturing) return;
  
    const canvas = latestCanvasRef.current;
    if (!canvas) {
      setEnrollError('Camera frame not ready yet. Wait a moment and try again.');
      return;
    }

    const nextSlot = enrollSlots.findIndex((slot) => slot === null);
    if (nextSlot === -1) return; // buffer saturated

    setIsCapturing(true);

    canvas.toBlob(
      (blob) => {
        setIsCapturing(false);
        if (!blob) {
          setEnrollError('Failed to capture frame from camera.');
          return;
        }
        const poseHint = POSE_HINTS[nextSlot].toLowerCase();
        const file = new File(
          [blob],
          `enrollment_${poseHint}_${Date.now()}.jpg`,
          { type: 'image/jpeg' }
        );
        setSlotFile(nextSlot, file);
        setEnrollError('');
      },
      'image/jpeg',
      0.92
    );
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
  const nextSlotIndex = enrollSlots.findIndex((slot) => slot === null);
  const bufferSaturated = nextSlotIndex === -1;

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

                          {/* Capture mode toggle. Both modes populate the same
                              File[] buffer downstream; only the source of the
                              File objects differs. */}
                          <div className="flex gap-2 text-xs">
                            <button
                              type="button"
                              onClick={() => setCaptureMode('upload')}
                              className={`px-3 py-1 rounded-md border transition ${
                                captureMode === 'upload'
                                  ? 'bg-slate-900 text-white border-slate-900'
                                  : 'bg-white text-slate-700 hover:bg-slate-100'
                              }`}
                            >
                              Upload files
                            </button>
                            <button
                              type="button"
                              onClick={() => setCaptureMode('camera')}
                              className={`px-3 py-1 rounded-md border transition ${
                                captureMode === 'camera'
                                  ? 'bg-slate-900 text-white border-slate-900'
                                  : 'bg-white text-slate-700 hover:bg-slate-100'
                              }`}
                            >
                              Capture from camera
                            </button>
                          </div>

                          {/* Live webcam viewport (only mounted in camera mode
                              so getUserMedia is not held open while uploading). */}
                          {captureMode === 'camera' && (
                            <div className="space-y-2">
                              <div className="relative aspect-video rounded-lg overflow-hidden bg-black">
                                <WebcamFeed
                                  className="absolute inset-0"
                                  onFrame={handleFrame}
                                  fps={8}
                                />
                                {!bufferSaturated && (
                                  <div className="absolute bottom-2 left-2 right-2 bg-black/60 text-white text-xs rounded px-2 py-1">
                                    Next: <span className="font-semibold">{POSE_HINTS[nextSlotIndex]}</span>
                                    {' · '}
                                    {nextSlotIndex === 0 && 'Look straight at the camera'}
                                    {nextSlotIndex === 1 && 'Turn your head slightly to the left'}
                                    {nextSlotIndex === 2 && 'Turn your head slightly to the right'}
                                  </div>
                                )}
                              </div>
                              <Button
                                size="sm"
                                onClick={captureFromCamera}
                                disabled={bufferSaturated || enrolling}
                                className="gap-2"
                              >
                                <Camera className="h-3 w-3" />
                                {bufferSaturated
                                  ? 'Buffer full · retake a slot below'
                                  : `Capture ${POSE_HINTS[nextSlotIndex]} frame`}
                              </Button>
                            </div>
                          )}

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
                                    {/* Upload-mode controls. Hidden in camera
                                        mode so the operator isn't tempted to
                                        mix sources within one session. */}
                                    {captureMode === 'upload' && (
                                      <label className="inline-flex">
                                        <input
                                          type="file"
                                          accept="image/jpeg,image/png"
                                          className="hidden"
                                          onChange={(e) => setSlotFile(slotIndex, e.target.files?.[0] ?? null)}
                                          onClick={(e) => { (e.target as HTMLInputElement).value = ''; }}
                                        />
                                        <span className="inline-flex items-center rounded-md border px-3 py-1 text-xs cursor-pointer bg-white hover:bg-slate-100">
                                          {file ? 'Retake' : 'Capture'}
                                        </span>
                                      </label>
                                    )}
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