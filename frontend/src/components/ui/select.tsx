import * as React from "react";
import { cn } from "../../lib/utils";

type Ctx = { value?: string; onValueChange?: (v: string) => void };
const SelectCtx = React.createContext<Ctx>({});

export function Select({
  defaultValue,
  value,
  onValueChange,
  children,
}: {
  defaultValue?: string;
  value?: string;
  onValueChange?: (v: string) => void;
  children: React.ReactNode;
}) {
  const [internal, setInternal] = React.useState(defaultValue ?? "");
  const isControlled = value !== undefined;
  const v = isControlled ? value : internal;

  return (
    <SelectCtx.Provider
      value={{
        value: v,
        onValueChange: (nv) => {
          if (!isControlled) setInternal(nv);
          onValueChange?.(nv);
        },
      }}
    >
      {children}
    </SelectCtx.Provider>
  );
}

export function SelectTrigger({
  className,
  children,
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("relative", className)}>
      {children}
    </div>
  );
}

export function SelectValue({ placeholder }: { placeholder?: string }) {
  // placeholder handled by native select
  return <span className="text-sm text-slate-500">{placeholder}</span>;
}

export function SelectContent({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

export function SelectItem({ value, children }: { value: string; children: React.ReactNode }) {
  // Rendered inside NativeSelect
  return <option value={value}>{children}</option>;
}

export function NativeSelect({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  const ctx = React.useContext(SelectCtx);
  return (
    <select
      value={ctx.value}
      onChange={(e) => ctx.onValueChange?.(e.target.value)}
      className={cn(
        "h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-slate-200",
        className
      )}
    >
      {children}
    </select>
  );
}
