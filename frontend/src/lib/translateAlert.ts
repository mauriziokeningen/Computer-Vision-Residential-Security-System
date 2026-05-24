import type { Locale } from '../i18n/LanguageContext';

interface AlertTemplates {
  unknownPerson: (camera: string, confidence: string) => string;
  weaponDetected: (weapon: string, camera: string, confidence: string) => string;
  aggressionDetected: (action: string, camera: string, confidence: string) => string;
  fallDetected: (camera: string, confidence: string) => string;
}

const TEMPLATES: Record<Locale, AlertTemplates> = {
  es: {
    unknownPerson: (camera, confidence) =>
      `Persona desconocida detectada en ${camera} — Confianza: ${confidence}`,
    weaponDetected: (weapon, camera, confidence) =>
      `ARMA DETECTADA: ${weapon} en ${camera} — Confianza: ${confidence}`,
    aggressionDetected: (action, camera, confidence) =>
      `AGRESIÓN DETECTADA: ${action} en ${camera} — Confianza: ${confidence}`,
    fallDetected: (camera, confidence) =>
      `CAÍDA DETECTADA en ${camera} — Confianza: ${confidence}`,
  },
  en: {
    unknownPerson: (camera, confidence) =>
      `Unknown person detected at ${camera} — Confidence: ${confidence}`,
    weaponDetected: (weapon, camera, confidence) =>
      `WEAPON DETECTED: ${weapon} at ${camera} — Confidence: ${confidence}`,
    aggressionDetected: (action, camera, confidence) =>
      `AGGRESSION DETECTED: ${action} at ${camera} — Confidence: ${confidence}`,
    fallDetected: (camera, confidence) =>
      `FALL DETECTED at ${camera} — Confidence: ${confidence}`,
  },
};

function extractConfidence(message: string): string {
  const match = message.match(/[Cc]onfianza:\s*([\d.]+)/);
  return match ? `${match[1]}%` : '';
}

function extractCamera(message: string): string {
  const match = message.match(/en\s+([^\s(]+)/i);
  return match ? match[1] : '';
}

function extractWeapon(message: string): string {
  const match = message.match(/(?:ARMA DETECTADA|WEAPON DETECTED):\s*(\S+)/i);
  return match ? match[1] : 'unknown';
}

function extractAction(message: string): string {
  const match = message.match(/(?:AGRESION DETECTADA|AGGRESSION DETECTED):\s*(\S+)/i);
  return match ? match[1] : 'unknown';
}

export function translateAlertMessage(message: string, locale: Locale): string {
  const t = TEMPLATES[locale];
  const confidence = extractConfidence(message);
  const camera = extractCamera(message);

  if (/persona desconocida|unknown person/i.test(message))
    return t.unknownPerson(camera, confidence);

  if (/arma detectada|weapon detected/i.test(message))
    return t.weaponDetected(extractWeapon(message), camera, confidence);

  if (/agresion detectada|aggression detected/i.test(message))
    return t.aggressionDetected(extractAction(message), camera, confidence);

  if (/caida detectada|fall detected/i.test(message))
    return t.fallDetected(camera, confidence);

  return message;
}