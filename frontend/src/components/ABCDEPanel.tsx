import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { Detection } from "@/types/api";
import { ShieldCheck, Circle } from "lucide-react";

interface ABCDEPanelProps {
  detections: Detection[];
}

/* Letter labels for each ABCDE criterion */
const CRITERIA = [
  { key: "asymmetry_score", letter: "A", label: "Asymmetry", max: 2 },
  { key: "border_score", letter: "B", label: "Border", max: 2 },
  { key: "color_score", letter: "C", label: "Color", max: 3 },
  { key: "diameter_score", letter: "D", label: "Diameter", max: 2 },
  { key: "evolution_score", letter: "E", label: "Evolution", max: 2 },
] as const;

/* Color for risk level badges */
const riskColors: Record<string, "teal" | "warning" | "danger" | "outline"> = {
  Low: "teal",
  Moderate: "warning",
  High: "danger",
  Refer: "danger",
};

export function ABCDEPanel({ detections }: ABCDEPanelProps) {
  const withAbcde = detections.filter((d) => d.abcde);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldCheck className="w-3.5 h-3.5 text-teal-600" />
          ABCDE Analysis
        </CardTitle>
      </CardHeader>
      <CardContent>
        {withAbcde.length > 0 ? (
          <div className="flex flex-col gap-4">
            {withAbcde.map((det, i) => (
              <LesionABCDE key={det.track_id ?? `abcde-${i}`} det={det} index={i} />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center py-4">
            <Circle className="w-5 h-5 text-slate-300 mb-1.5" />
            <p className="text-xs text-slate-400">
              No morphological data yet
            </p>
            <p className="text-[10px] text-slate-400 mt-0.5">
              ABCDE scores appear after lesion detection
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ── Per-lesion ABCDE breakdown ──────────────────────── */

function LesionABCDE({ det, index }: { det: Detection; index: number }) {
  const abcde = det.abcde!;
  const risk = det.risk;

  return (
    <div className="rounded-lg border border-slate-100 bg-slate-25 p-3 fade-in">
      {/* Header: lesion name + risk badge */}
      <div className="flex items-center justify-between mb-2.5">
        <span className="text-xs font-semibold text-slate-700">
          {det.track_id != null ? `Lesion #${det.track_id}` : `Lesion ${index + 1}`}
        </span>
        {risk && (
          <Badge variant={riskColors[risk.level] ?? "outline"}>
            {risk.level}
            <span className="ml-1 opacity-70">({risk.total_score})</span>
          </Badge>
        )}
      </div>

      {/* ABCDE criteria bars */}
      <div className="flex flex-col gap-2">
        {CRITERIA.map(({ key, letter, label, max }) => {
          const value = abcde[key] as number;
          const pct = (value / max) * 100;
          const variant = pct >= 75 ? "red" : pct >= 40 ? "amber" : "teal";

          return (
            <div key={key} className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-md bg-slate-100 flex items-center justify-center text-[10px] font-bold text-slate-600 shrink-0">
                {letter}
              </span>
              <div className="flex-1 flex flex-col gap-0.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-500">{label}</span>
                  <span className="text-[10px] font-semibold text-slate-700 tabular-nums">
                    {value}/{max}
                  </span>
                </div>
                <Progress value={pct} variant={variant} size="sm" />
              </div>
            </div>
          );
        })}
      </div>

      {/* Diameter callout */}
      {abcde.diameter_mm > 0 && (
        <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-100 text-[10px]">
          <span className="text-slate-500">Est. diameter</span>
          <span className={`font-semibold tabular-nums ${abcde.diameter_mm >= 6 ? "text-red-600" : "text-slate-700"}`}>
            {abcde.diameter_mm.toFixed(1)} mm
            {abcde.diameter_mm >= 6 && " ⚠"}
          </span>
        </div>
      )}

      {/* Risk summary */}
      {risk && (
        <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">
          {risk.summary}
        </p>
      )}
    </div>
  );
}
