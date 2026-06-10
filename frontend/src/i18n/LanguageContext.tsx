import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from 'react';
import { es } from './locales/es';
import { en } from './locales/en';
import type { Dictionary } from './locales/es';

export type Locale = 'es' | 'en';

const COOKIE_KEY = 'uh_locale';
const DICTIONARIES: Record<Locale, Dictionary> = { es, en };

function getInitialLocale(): Locale {
  // 1. Cookie persistida
  const cookie = document.cookie
    .split('; ')
    .find((r) => r.startsWith(`${COOKIE_KEY}=`))
    ?.split('=')[1];
  if (cookie === 'es' || cookie === 'en') return cookie;

  // 2. Accept-Language del navegador (fallback)
  const lang = navigator.language.toLowerCase();
  if (lang.startsWith('es')) return 'es';

  // 3. Fallback a inglés (bedrock arquitectónico)
  return 'en';
}

interface LanguageContextValue {
  locale: Locale;
  t: Dictionary;
  setLocale: (l: Locale) => void;
}

const LanguageContext = createContext<LanguageContextValue | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(getInitialLocale);

  const setLocale = useCallback((l: Locale) => {
    // Persiste en cookie (1 año)
    document.cookie = `${COOKIE_KEY}=${l};path=/;max-age=31536000;SameSite=Lax`;
    setLocaleState(l);
  }, []);

  return (
    <LanguageContext.Provider value={{ locale, t: DICTIONARIES[locale], setLocale }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error('useLanguage must be used inside <LanguageProvider>');
  return ctx;
}

/** Helper para interpolación: t.alerts.unknownPersonDetected con {camera_id} */
export function interpolate(template: string, vars: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => vars[key] ?? `{${key}}`);
}