import type { Detection, StreamState } from "@/types/api";
import { MonitorUp, ImageOff } from "lucide-react";

interface ViewportProps {
  frameUrl: string | null;
  detections: Detection[];
  streamState: StreamState;
  frameWidth: number;
  frameHeight: number;
}

export function Viewport({
  frameUrl,
  detections,
  streamState,
  frameWidth,
  frameHeight,
}: ViewportProps) {
  const hasFrame = !!frameUrl;

  return (
    <div className="flex flex-col gap-2 flex-1 min-w-0">
      {/* Viewport label bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MonitorUp className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
            Primary Viewport
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          {hasFrame && (
            <span>
              {frameWidth}×{frameHeight}
            </span>
          )}
          {streamState === "streaming" && (
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 pulse-dot" />
              LIVE
            </span>
          )}
        </div>
      </div>

      {/* Viewport frame */}
      <div className="viewport-frame aspect-video w-full flex items-center justify-center relative">
        {hasFrame ? (
          <>
            <img
              src={frameUrl}
              alt="Endoscopic frame"
              className="w-full h-full object-contain"
              draggable={false}
            />

            {/* Bounding box overlays */}
            {detections.map((det, i) => {
              const [x1, y1, x2, y2] = det.bbox;
              const left = (x1 / frameWidth) * 100;
              const top = (y1 / frameHeight) * 100;
              const width = ((x2 - x1) / frameWidth) * 100;
              const height = ((y2 - y1) / frameHeight) * 100;

              return (
                <div
                  key={det.track_id ?? `det-${i}`}
                  className={`bbox-overlay ${!det.confirmed ? "pending" : ""}`}
                  style={{
                    left: `${left}%`,
                    top: `${top}%`,
                    width: `${width}%`,
                    height: `${height}%`,
                  }}
                >
                  <span
                    className="bbox-label"
                    style={{
                      backgroundColor: det.confirmed
                        ? "rgba(13, 148, 136, 0.9)"
                        : "rgba(217, 119, 6, 0.9)",
                    }}
                  >
                    {det.track_id != null ? `ID ${det.track_id}` : "Lesion"}{" "}
                    {(det.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}

            {/* Scanline overlay for live feed */}
            {streamState === "streaming" && (
              <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-transparent to-black/5" />
            )}
          </>
        ) : (
          /* Empty state */
          <div className="flex flex-col items-center gap-3 text-slate-500">
            <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center">
              <ImageOff className="w-6 h-6 text-slate-500" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-slate-400">
                No feed active
              </p>
              <p className="text-xs text-slate-600 mt-0.5">
                Upload an image or start a video stream
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
