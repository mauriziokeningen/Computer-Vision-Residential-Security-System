import * as React from "react";
import { cn } from "../../lib/utils";

export function Switch({
  defaultChecked,
  checked,
  onCheckedChange,
  className,
}: {
  defaultChecked?: boolean;
  checked?: boolean;
  onCheckedChange?: (v: boolean) => void;
  className?: string;
}) {
  const [internal, setInternal] = React.useState<boolean>(defaultChecked ?? false);
  const isControlled = typeof checked === "boolean";
  const val = isControlled ? checked! : internal;

  return (
    <button
      type="button"
      onClick={() => {
        const next = !val;
        if (!isControlled) setInternal(next);
        onCheckedChange?.(next);
      }}
      className={cn(
        "relative inline-flex h-6 w-11 items-center rounded-full border border-slate-200 bg-slate-100 transition-colors",
        val ? "bg-slate-900" : "bg-slate-100",
        className
      )}
      aria-checked={val}
      role="switch"
    >
      <span
        className={cn(
          "inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform",
          val ? "translate-x-5" : "translate-x-1"
        )}
      />
    </button>
  );
}
