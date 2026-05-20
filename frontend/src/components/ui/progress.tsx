import { forwardRef, type HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface ProgressProps extends HTMLAttributes<HTMLDivElement> {
  value: number;       // 0–100
  max?: number;
  variant?: "teal" | "amber" | "red" | "slate";
  size?: "sm" | "md";
}

const barColors = {
  teal: "bg-teal-500",
  amber: "bg-amber-500",
  red: "bg-red-500",
  slate: "bg-slate-400",
};

const Progress = forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value, max = 100, variant = "teal", size = "sm", ...props }, ref) => {
    const pct = Math.min(100, Math.max(0, (value / max) * 100));
    return (
      <div
        ref={ref}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        className={cn(
          "w-full overflow-hidden rounded-full bg-slate-100",
          size === "sm" ? "h-1.5" : "h-2.5",
          className
        )}
        {...props}
      >
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500 ease-out",
            barColors[variant]
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    );
  }
);
Progress.displayName = "Progress";

export { Progress };
