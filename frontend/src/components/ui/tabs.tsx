import * as React from "react";
import { cn } from "../../lib/utils";

type TabsCtxType = { value: string; setValue: (v: string) => void };
const TabsCtx = React.createContext<TabsCtxType | null>(null);

export function Tabs({
  value,
  defaultValue,
  onValueChange,
  children,
}: {
  value?: string;
  defaultValue?: string;
  onValueChange?: (v: string) => void;
  children: React.ReactNode;
}) {
  const [internal, setInternal] = React.useState(defaultValue ?? "");
  const isControlled = value !== undefined;
  const v = isControlled ? value! : internal;

  const setValue = (nv: string) => {
    if (!isControlled) setInternal(nv);
    onValueChange?.(nv);
  };

  return <TabsCtx.Provider value={{ value: v, setValue }}>{children}</TabsCtx.Provider>;
}

export function TabsList({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("inline-flex h-10 items-center rounded-xl bg-slate-100 p-1", className)}
      {...props}
    />
  );
}

export function TabsTrigger({
  value,
  className,
  children,
}: {
  value: string;
  className?: string;
  children: React.ReactNode;
}) {
  const ctx = React.useContext(TabsCtx);
  if (!ctx) return null;
  const active = ctx.value === value;
  return (
    <button
      type="button"
      onClick={() => ctx.setValue(value)}
      className={cn(
        "h-8 px-3 rounded-lg text-sm transition-colors",
        active ? "bg-white shadow-sm" : "text-slate-600 hover:text-slate-900",
        className
      )}
    >
      {children}
    </button>
  );
}

export function TabsContent({
  value,
  className,
  children,
}: {
  value: string;
  className?: string;
  children: React.ReactNode;
}) {
  const ctx = React.useContext(TabsCtx);
  if (!ctx || ctx.value !== value) return null;
  return <div className={cn("mt-3", className)}>{children}</div>;
}
