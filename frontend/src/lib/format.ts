/**
 * Shared formatting / parsing helpers used across domain features.
 * These are pure utilities with no UI concerns — hence they live in lib/.
 */

export function parseApiDate(iso: string) {
  if (!iso) return new Date(NaN);
  // SQLAlchemy/Postgres often leaks naive timestamps with spaces instead of 'T'
  // This strictly polyfills it for WebKit/Safari engines to prevent 'Invalid Date'
  let normalized = iso.trim().replace(' ', 'T');
  if (!normalized.endsWith('Z') && !normalized.match(/[+-]\d{2}:\d{2}$/)) {
    normalized += 'Z'; // Assume UTC for system consistency
  }
  return new Date(normalized);
}

export function formatTime(iso: string) {
  const date = parseApiDate(iso); // Re-apply the polyfill here
  return date.toLocaleString('es-MX', {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: '2-digit',
  });
}

export function priorityToSeverity(p?: string): 'low' | 'high' | 'critical' {
  if (p === 'CRITICAL') return 'critical';
  if (p === 'HIGH' || p === 'MEDIUM') return 'high';
  return 'low';
}

