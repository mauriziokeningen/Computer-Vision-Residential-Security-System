import * as React from "react";
import { cn } from "../../lib/utils";

export function Progress({ value = 0, className }: { value: number; className?: string }) {
  return (
    <div className={cn("h-2 w-full rounded-full bg-slate-100 overflow-hidden", className)}>
      <div
        className="h-full bg-slate-900 rounded-full transition-all"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}
