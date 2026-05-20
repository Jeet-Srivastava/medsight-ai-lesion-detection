import { useRef } from "react";
import { Button } from "@/components/ui/button";
import type { StreamState } from "@/types/api";
import { Upload, Play, Square, Pause, SlidersHorizontal } from "lucide-react";

interface ControlBarProps {
  confidence: number;
  onConfidenceChange: (val: number) => void;
  streamState: StreamState;
  onUploadImage: (file: File) => void;
  onUploadVideo: (file: File) => void;
  onStartStream: () => void;
  onStopStream: () => void;
  onPauseStream: () => void;
}

export function ControlBar({
  confidence,
  onConfidenceChange,
  streamState,
  onUploadImage,
  onUploadVideo,
  onStartStream,
  onStopStream,
  onPauseStream,
}: ControlBarProps) {
  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  const isStreaming = streamState === "streaming";
  const isPaused = streamState === "paused";
  const isProcessing = streamState === "processing";

  return (
    <div className="border-t border-slate-200/80 bg-white/60 backdrop-blur-sm px-5 py-3 flex items-center gap-6 shrink-0">
      {/* ── Upload Controls ────────────────── */}
      <div className="flex items-center gap-2">
        <input
          ref={imageInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,.webp"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUploadImage(file);
            e.target.value = "";
          }}
        />
        <Button
          variant="outline"
          size="sm"
          onClick={() => imageInputRef.current?.click()}
          disabled={isStreaming || isProcessing}
          id="btn-upload-image"
        >
          <Upload className="w-3.5 h-3.5" />
          Image
        </Button>

        <input
          ref={videoInputRef}
          type="file"
          accept=".mp4,.mov,.avi,.mkv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUploadVideo(file);
            e.target.value = "";
          }}
        />
        <Button
          variant="outline"
          size="sm"
          onClick={() => videoInputRef.current?.click()}
          disabled={isStreaming || isProcessing}
          id="btn-upload-video"
        >
          <Upload className="w-3.5 h-3.5" />
          Video
        </Button>
      </div>

      {/* ── Divider ─── */}
      <div className="w-px h-6 bg-slate-200" />

      {/* ── Stream Controls ────────────────── */}
      <div className="flex items-center gap-2">
        {!isStreaming && !isPaused ? (
          <Button
            size="sm"
            onClick={onStartStream}
            disabled={isProcessing}
            id="btn-start-stream"
          >
            <Play className="w-3.5 h-3.5" />
            Start Stream
          </Button>
        ) : (
          <>
            <Button
              variant="secondary"
              size="sm"
              onClick={onPauseStream}
              id="btn-pause-stream"
            >
              <Pause className="w-3.5 h-3.5" />
              {isPaused ? "Resume" : "Pause"}
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={onStopStream}
              id="btn-stop-stream"
            >
              <Square className="w-3.5 h-3.5" />
              Stop
            </Button>
          </>
        )}
      </div>

      {/* ── Divider ─── */}
      <div className="w-px h-6 bg-slate-200" />

      {/* ── Confidence Threshold ───────────── */}
      <div className="flex items-center gap-3 flex-1 max-w-xs">
        <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400 shrink-0" />
        <div className="flex flex-col gap-1 flex-1">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium text-slate-500">
              Confidence Threshold
            </span>
            <span className="text-[11px] font-semibold text-teal-700 tabular-nums bg-teal-50 px-1.5 py-0.5 rounded">
              {(confidence * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min={5}
            max={95}
            step={5}
            value={confidence * 100}
            onChange={(e) => onConfidenceChange(Number(e.target.value) / 100)}
            className="w-full"
            id="slider-confidence"
          />
        </div>
      </div>
    </div>
  );
}
