import * as React from "react";
import { cn } from "../../lib/utils";

type Variant = "default" | "secondary" | "outline" | "ghost" | "destructive";
type Size = "default" | "sm" | "icon";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant="default", size="default", ...props }, ref) => {
    const v = {
      default: "bg-slate-900 text-white hover:bg-slate-800",
      secondary: "bg-slate-100 text-slate-900 hover:bg-slate-200",
      outline: "border border-slate-200 bg-white hover:bg-slate-50",
      ghost: "hover:bg-slate-100",
      destructive: "bg-red-600 text-white hover:bg-red-700",
    }[variant];

    const s = {
      default: "h-10 px-4 py-2 text-sm",
      sm: "h-9 px-3 text-sm",
      icon: "h-10 w-10",
    }[size];

    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-xl font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-slate-300 disabled:opacity-50 disabled:pointer-events-none",
          v,
          s,
          className
        )}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";
