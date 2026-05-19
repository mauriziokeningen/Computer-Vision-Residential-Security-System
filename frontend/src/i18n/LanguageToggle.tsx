import React from 'react';
import { Globe } from 'lucide-react';
import { Button } from '../components/ui/button';
import { useLanguage, type Locale } from './LanguageContext';

const OPTIONS: { value: Locale; label: string }[] = [
  { value: 'es', label: 'ES' },
  { value: 'en', label: 'EN' },
];

export function LanguageToggle() {
  const { locale, setLocale } = useLanguage();

  return (
    <div className="flex items-center gap-1">
      <Globe className="h-4 w-4 text-slate-400" />
      {OPTIONS.map((opt) => (
        <Button
          key={opt.value}
          variant={locale === opt.value ? 'secondary' : 'ghost'}
          size="sm"
          className="h-7 px-2 text-xs font-semibold"
          onClick={() => setLocale(opt.value)}
          aria-pressed={locale === opt.value}
        >
          {opt.label}
        </Button>
      ))}
    </div>
  );
}