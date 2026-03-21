import * as React from "react";
import { cn } from "../../lib/utils";

type Variant = "default" | "outline" | "destructive";

export function Badge({
  className,
  variant = "default",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: Variant }) {
  const v = {
    default: "bg-slate-900 text-white",
    outline: "border border-slate-200 bg-white text-slate-900",
    destructive: "bg-red-600 text-white",
  }[variant];

  return (
    <span
      className={cn(
        "inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-full",
        v,
        className
      )}
      {...props}
    />
  );
}
