import { Header } from "@/components/Header";
import { Viewport } from "@/components/Viewport";
import { AnalyticsPanel } from "@/components/AnalyticsPanel";
import { ABCDEPanel } from "@/components/ABCDEPanel";
import { ReportPanel } from "@/components/ReportPanel";
import { ControlBar } from "@/components/ControlBar";
import { useDashboard } from "@/hooks/useDashboard";

export default function App() {
  const {
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
    report,
    reportLoading,
    handleUploadImage,
    handleUploadVideo,
    handleStartStream,
    handleStopStream,
    handlePauseStream,
    handleConfidenceChange,
    handleGenerateReport,
  } = useDashboard();

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* ── Top Header ──────────────────────── */}
      <Header systemStatus={systemStatus} sessionId={sessionId} />

      {/* ── Main Workspace ──────────────────── */}
      <main className="flex flex-1 min-h-0 gap-4 p-4">
        {/* Primary Viewport */}
        <Viewport
          frameUrl={frameUrl}
          detections={detections}
          streamState={streamState}
          frameWidth={frameWidth}
          frameHeight={frameHeight}
        />

        {/* Right Sidebar — scrollable stack of panels */}
        <aside className="w-[320px] shrink-0 flex flex-col gap-3 overflow-y-auto">
          {/* Existing analytics */}
          <AnalyticsPanel 
            analytics={analytics} 
            detections={detections} 
            systemLogs={systemLogs} 
          />

          {/* ABCDE Morphological Analysis */}
          <ABCDEPanel detections={detections} />

          {/* Clinical Report */}
          <ReportPanel
            report={report}
            onGenerateReport={handleGenerateReport}
            isLoading={reportLoading}
          />
        </aside>
      </main>

      {/* ── Control Bar ─────────────────────── */}
      <ControlBar
        confidence={confidence}
        onConfidenceChange={handleConfidenceChange}
        streamState={streamState}
        onUploadImage={handleUploadImage}
        onUploadVideo={handleUploadVideo}
        onStartStream={handleStartStream}
        onStopStream={handleStopStream}
        onPauseStream={handlePauseStream}
      />
    </div>
  );
}
