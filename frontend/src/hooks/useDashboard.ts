import { useState, useCallback, useRef, useEffect } from "react";
import type {
  Detection,
  FrameAnalytics,
  StreamState,
  SystemStatus,
  PipelineResult,
  ClinicalReport,
} from "@/types/api";
import * as api from "@/api/client";

// ── Default state values ────────────────────────────────

const DEFAULT_STATUS: SystemStatus = {
  model_name: "YOLO11",
  model_path: "yolo11n.pt",
  device: "CPU",
  status: "idle",
  fp16_enabled: false,
};

/**
 * Core application state hook.
 * Manages all dashboard state and exposes handler functions
 * that the UI components bind to.
 */
export function useDashboard() {
  // System
  const [systemStatus, setSystemStatus] =
    useState<SystemStatus>(DEFAULT_STATUS);
  const [sessionId] = useState(() => generateSessionId());

  // Inference state
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [analytics, setAnalytics] = useState<FrameAnalytics | null>(null);

  const [streamState, setStreamState] = useState<StreamState>("idle");
  const [confidence, setConfidence] = useState(0.35);
  const [frameWidth, setFrameWidth] = useState(640);
  const [frameHeight, setFrameHeight] = useState(480);
  const [systemLogs, setSystemLogs] = useState<Array<[string, string]>>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Report & XAI state
  const [report, setReport] = useState<ClinicalReport | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [saliencyUrl, setSaliencyUrl] = useState<string | null>(null);

  // Stream polling ref
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Media capture refs for browser camera streaming
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const captureRef = useRef<number | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  // ── Fetch system status on mount ────────────────────
  useEffect(() => {
    api
      .fetchSystemStatus()
      .then(setSystemStatus)
      .catch(() => {
        /* backend not connected yet – use defaults */
      });
  }, []);

  // ── Handlers ────────────────────────────────────────

  const revokeFrameObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  const showLocalImage = useCallback(
    (file: File) => {
      revokeFrameObjectUrl();
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      setFrameUrl(url);

      const img = new Image();
      img.onload = () => {
        setFrameWidth(img.naturalWidth);
        setFrameHeight(img.naturalHeight);
      };
      img.src = url;
    },
    [revokeFrameObjectUrl]
  );

  const applyResult = useCallback((result: PipelineResult | { status: string }) => {
    if ("status" in result && result.status === "stopped") return;
    const pResult = result as PipelineResult;
    setDetections(pResult.confirmed_detections || []);
    setAnalytics(pResult.analytics || null);
    if (pResult.logs?.length) {
      setSystemLogs((prev) => [...prev, ...pResult.logs].slice(-200));
    }
    if (pResult.annotated_frame_b64) {
      revokeFrameObjectUrl();
      setFrameUrl(`data:image/jpeg;base64,${pResult.annotated_frame_b64}`);
    }
    if (pResult.frame_width && pResult.frame_height) {
      setFrameWidth(pResult.frame_width);
      setFrameHeight(pResult.frame_height);
    }
    // Clear stale report when new inference arrives
    setReport(null);
    setSaliencyUrl(null);
    setErrorMessage(null);
  }, [revokeFrameObjectUrl]);

  const handleUploadImage = useCallback(
    async (file: File) => {
      setStreamState("processing");
      setErrorMessage(null);
      try {
        const result = await api.uploadImageForInference(file, confidence);

        if (result.analytics) {
          applyResult(result);
        } else {
          showLocalImage(file);
        }

        setSystemStatus((prev) => ({ ...prev, status: "active" }));
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Analysis failed";
        setErrorMessage(message);
        setSystemLogs((prev) => {
          const next: Array<[string, string]> = [...prev, ["warn", message]];
          return next.slice(-200);
        });
        showLocalImage(file);
        setDetections([]);
        setAnalytics(null);
      } finally {
        setStreamState("idle");
      }
    },
    [confidence, applyResult, showLocalImage]
  );

  const startClientCapture = useCallback(() => {
    const canvasEl = document.createElement("canvas");
    const canvasCtx = canvasEl.getContext("2d");
    const isSending = { value: false };

    const captureLoop = async () => {
      try {
        const v = videoElRef.current;
        if (!v || v.readyState < 2) return;
        if (isSending.value) return;

        const desiredW = 480;
        const srcW = v.videoWidth || frameWidth;
        const srcH = v.videoHeight || frameHeight;
        const aspect = srcW && srcH ? srcW / srcH : frameWidth / frameHeight;
        const w = Math.min(desiredW, srcW || desiredW);
        const h = Math.max(1, Math.round(w / aspect));

        canvasEl.width = w;
        canvasEl.height = h;
        if (!canvasCtx) return;
        canvasCtx.drawImage(v, 0, 0, w, h);

        const blob: Blob | null = await new Promise((resolve) =>
          canvasEl.toBlob((b) => resolve(b), "image/jpeg", 0.6)
        );
        if (!blob) return;

        isSending.value = true;
        try {
          const result = await api.sendClientFrame(blob, confidence);
          applyResult(result);
        } catch {
          /* ignore per-frame errors */
        } finally {
          isSending.value = false;
        }
      } catch {
        /* ignore capture errors */
      }
    };

    if (captureRef.current) clearInterval(captureRef.current);
    captureRef.current = window.setInterval(captureLoop, 120);
  }, [applyResult, confidence, frameHeight, frameWidth]);

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const result = await api.fetchStreamFrame();
        applyResult(result);
      } catch (err: any) {
        if (err.message === "Stream ended") {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setStreamState("idle");
          setSystemStatus((prev) => ({ ...prev, status: "idle" }));
        }
      }
    }, 120);
  }, [applyResult]);

  const handleUploadVideo = useCallback(
    async (file: File) => {
      setStreamState("processing");
      try {
        await api.uploadVideoForInference(file, confidence);
        setSystemStatus((prev) => ({ ...prev, status: "active" }));
        setStreamState("streaming");
        startPolling();
      } catch {
        // Demo mode
        setSystemStatus((prev) => ({ ...prev, status: "active" }));
        setStreamState("idle");
      }
    },
    [confidence, startPolling]
  );

  const handleStartStream = useCallback(async () => {
    // Try browser camera first so the user can grant access in-browser.
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      mediaStreamRef.current = stream;

      const video = document.createElement("video");
      video.autoplay = true;
      video.playsInline = true;
      video.muted = true;
      video.srcObject = stream;
      // Ensure play() resolves
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      video.play();
      videoElRef.current = video;

      setStreamState("streaming");
      setSystemStatus((prev) => ({ ...prev, status: "active" }));

      startClientCapture();
    } catch {
      // Fallback: ask backend to use server webcam (existing behavior)
      try {
        await api.startStream(confidence);
        setStreamState("streaming");
        setSystemStatus((prev) => ({ ...prev, status: "active" }));
        startPolling();
      } catch {
        // Demo fallback
        setStreamState("streaming");
        setSystemStatus((prev) => ({ ...prev, status: "active" }));
      }
    }
  }, [confidence, startPolling, startClientCapture]);

  const handleStopStream = useCallback(async () => {
    // Stop any client-side capture
    if (captureRef.current) {
      clearInterval(captureRef.current);
      captureRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoElRef.current) {
      videoElRef.current.srcObject = null;
      videoElRef.current = null;
    }

    // Stop server-side stream if any
    try {
      await api.stopStream();
    } catch {
      /* ignore */
    }

    setStreamState("idle");
    setSystemStatus((prev) => ({ ...prev, status: "idle" }));
  }, []);

  const handlePauseStream = useCallback(() => {
    if (streamState === "streaming") {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
      if (captureRef.current) {
        clearInterval(captureRef.current);
        captureRef.current = null;
      }
      setStreamState("paused");
    } else if (streamState === "paused") {
      setStreamState("streaming");
      if (mediaStreamRef.current && videoElRef.current) {
        startClientCapture();
      } else {
        startPolling();
      }
    }
  }, [streamState, startPolling, startClientCapture]);

  const handleConfidenceChange = useCallback((val: number) => {
    setConfidence(val);
    api.updateConfidence(val).catch(() => {
      /* offline mode */
    });
  }, []);

  // ── Report & XAI handlers ─────────────────────────────

  const handleGenerateReport = useCallback(async () => {
    setReportLoading(true);
    try {
      const data = await api.fetchReport();
      setReport(data);
    } catch {
      /* no report available yet */
    } finally {
      setReportLoading(false);
    }
  }, []);

  const handleFetchSaliency = useCallback(async (index: number = 0) => {
    try {
      const data = await api.fetchSaliencyMap(index);
      if (data.saliency_frame_b64) {
        setSaliencyUrl(`data:image/jpeg;base64,${data.saliency_frame_b64}`);
      }
    } catch {
      /* saliency not available */
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (captureRef.current) clearInterval(captureRef.current);
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      revokeFrameObjectUrl();
    };
  }, [revokeFrameObjectUrl]);

  return {
    systemStatus,
    sessionId,
    frameUrl,
    detections,
    analytics,
    streamState,
    confidence,
    frameWidth,
    frameHeight,
    systemLogs,
    errorMessage,
    report,
    reportLoading,
    saliencyUrl,
    handleUploadImage,
    handleUploadVideo,
    handleStartStream,
    handleStopStream,
    handlePauseStream,
    handleConfidenceChange,
    handleGenerateReport,
    handleFetchSaliency,
    setErrorMessage,
  };
}

/* ── Helpers ─────────────────────────────────────────── */

function generateSessionId(): string {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let id = "";
  for (let i = 0; i < 8; i++) {
    id += chars[Math.floor(Math.random() * chars.length)];
  }
  return `MS-${id.slice(0, 4)}-${id.slice(4)}`;
}
