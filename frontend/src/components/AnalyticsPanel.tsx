import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { Detection, FrameAnalytics } from "@/types/api";
import {
  Crosshair,
  Clock,
  TrendingUp,
  Gauge,
  BarChart3,
  Layers,
} from "lucide-react";

interface AnalyticsPanelProps {
  analytics: FrameAnalytics | null;
  detections: Detection[];
  systemLogs?: Array<[string, string]>;
}

export function AnalyticsPanel({ analytics, detections, systemLogs = [] }: AnalyticsPanelProps) {
  const confirmed = detections.filter((d) => d.confirmed);
  const avgConf = analytics?.average_confidence ?? 0;
  const inferenceMs = analytics?.inference_ms ?? 0;
  const pipelineMs = analytics?.pipeline_ms ?? 0;
  const fps = analytics?.fps ?? 0;

  return (
    <aside className="w-[320px] shrink-0 flex flex-col gap-3 overflow-y-auto">
      {/* ── Lesion Count ────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Crosshair className="w-3.5 h-3.5 text-teal-600" />
            Lesions Detected
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-baseline gap-3">
            <span className="text-4xl font-bold text-slate-900 tabular-nums tracking-tight">
              {confirmed.length}
            </span>
            <div className="flex flex-col text-xs text-slate-500">
              <span>{detections.length} raw candidates</span>
              <span>
                {analytics?.total_confirmed_lesions ?? 0} total confirmed
              </span>
            </div>
          </div>
          {analytics && (
            <div className="flex gap-2 mt-3">
              <Badge variant="teal">
                {analytics.active_lesions} active
              </Badge>
              <Badge variant="outline">
                Frame {analytics.frame_index}
                {analytics.total_frames > 0 &&
                  ` / ${analytics.total_frames}`}
              </Badge>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── Confidence Scores ───────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-3.5 h-3.5 text-teal-600" />
            Confidence Scores
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {confirmed.length > 0 ? (
            confirmed.map((det, i) => {
              const pct = det.confidence * 100;
              const variant =
                pct >= 80 ? "teal" : pct >= 50 ? "amber" : "red";
              return (
                <div key={det.track_id ?? `c-${i}`} className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-600 font-medium">
                      {det.track_id != null
                        ? `Lesion #${det.track_id}`
                        : `Lesion ${i + 1}`}
                    </span>
                    <span className="font-semibold tabular-nums text-slate-800">
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={pct} variant={variant} size="sm" />
                </div>
              );
            })
          ) : (
            <div className="flex flex-col items-center py-4">
              <Layers className="w-5 h-5 text-slate-300 mb-1.5" />
              <p className="text-xs text-slate-400">No detections yet</p>
            </div>
          )}
          {confirmed.length > 0 && (
            <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-xs">
              <span className="text-slate-500">Average</span>
              <span className="font-semibold text-slate-700 tabular-nums">
                {(avgConf * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── Performance Metrics ─────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Gauge className="w-3.5 h-3.5 text-teal-600" />
            Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3">
            <MetricTile
              icon={<Clock className="w-3.5 h-3.5" />}
              label="Inference"
              value={`${inferenceMs.toFixed(0)}ms`}
            />
            <MetricTile
              icon={<TrendingUp className="w-3.5 h-3.5" />}
              label="Pipeline"
              value={`${pipelineMs.toFixed(0)}ms`}
            />
            <MetricTile
              icon={<Gauge className="w-3.5 h-3.5" />}
              label="FPS"
              value={fps > 0 ? fps.toFixed(1) : "—"}
            />
            <MetricTile
              icon={<BarChart3 className="w-3.5 h-3.5" />}
              label="Det. Freq."
              value={
                analytics
                  ? `${analytics.detection_frequency.toFixed(2)}/s`
                  : "—"
              }
            />
          </div>
        </CardContent>
      </Card>

      {/* ── Detection Log ───────────────────── */}
      <Card className="flex-1 min-h-0">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-teal-600" />
            Detection Log
          </CardTitle>
        </CardHeader>
        <CardContent className="max-h-48 overflow-y-auto">
          {systemLogs.length > 0 ? (
            <div className="flex flex-col gap-1.5">
              {[...systemLogs].reverse().map(([type, msg], i) => (
                <div
                  key={`log-${i}`}
                  className="flex items-start gap-2 text-xs py-1.5 px-2.5 rounded-md bg-slate-50 fade-in border border-slate-100"
                >
                  <span 
                    className={`w-1.5 h-1.5 rounded-full mt-1 shrink-0 ${
                      type === "detect" ? "bg-amber-500" :
                      type === "track" ? "bg-teal-500" : "bg-blue-400"
                    }`} 
                  />
                  <span className="text-slate-600 leading-snug">
                    {msg}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-400 text-center py-3">
              Awaiting inference…
            </p>
          )}
        </CardContent>
      </Card>
    </aside>
  );
}

/* ── Small Metric Tile ──────────────────────────── */
function MetricTile({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-lg bg-slate-50 px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-slate-400">{icon}
        <span className="text-[11px] font-medium">{label}</span>
      </div>
      <span className="text-lg font-semibold text-slate-800 tabular-nums leading-none">
        {value}
      </span>
    </div>
  );
}
