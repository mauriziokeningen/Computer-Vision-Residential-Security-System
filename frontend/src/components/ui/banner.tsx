import { XCircle, CheckCircle2 } from 'lucide-react';

export function ErrorBanner({ msg, onClose }: { msg: string; onClose: () => void }) {
  return (
    <div className="flex items-center gap-3 bg-red-50 border border-red-200 text-red-800 rounded-lg px-4 py-3 text-sm">
      <XCircle className="h-4 w-4 shrink-0" />
      <span className="flex-1">{msg}</span>
      <button onClick={onClose} className="text-red-400 hover:text-red-600">✕</button>
    </div>
  );
}

export function SuccessBanner({ msg, onClose }: { msg: string; onClose: () => void }) {
  return (
    <div className="flex items-center gap-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg px-4 py-3 text-sm">
      <CheckCircle2 className="h-4 w-4 shrink-0" />
      <span className="flex-1">{msg}</span>
      <button onClick={onClose} className="text-emerald-400 hover:text-emerald-600">✕</button>
    </div>
  );
}

