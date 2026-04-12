import * as React from "react";
import { cn } from "../../lib/utils";

/** Versión simple: scroll nativo con estilos. */
export function ScrollArea({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("overflow-auto rounded-xl", className)}
      {...props}
    />
  );
}
