import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { ClinicalReport } from "@/types/api";
import { FileText, Download, AlertTriangle, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ReportPanelProps {
  report: ClinicalReport | null;
  onGenerateReport: () => void;
  isLoading: boolean;
}

export function ReportPanel({ report, onGenerateReport, isLoading }: ReportPanelProps) {
  const handleDownload = () => {
    if (!report) return;
    const json = JSON.stringify(report, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `medsight-report-${report.report_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="w-3.5 h-3.5 text-teal-600" />
          Clinical Report
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Generate button */}
        <Button
          variant="outline"
          size="sm"
          onClick={onGenerateReport}
          disabled={isLoading}
          className="w-full mb-3"
          id="btn-generate-report"
        >
          <FileText className="w-3.5 h-3.5" />
          {isLoading ? "Generating…" : "Generate Report"}
        </Button>

        {report ? (
          <div className="flex flex-col gap-3 fade-in">
            {/* Report header */}
            <div className="rounded-lg bg-slate-50 p-2.5 text-[10px] space-y-1">
              <div className="flex justify-between">
                <span className="text-slate-500">Report ID</span>
                <span className="font-mono text-slate-700">{report.report_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Time</span>
                <span className="text-slate-700">
                  {new Date(report.timestamp).toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Session</span>
                <span className="font-mono text-slate-700">{report.session_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Image</span>
                <span className="text-slate-700">
                  {report.image_dimensions[0]}×{report.image_dimensions[1]}
                </span>
              </div>
            </div>

            {/* Summary */}
            <div className="text-[11px] text-slate-600 leading-relaxed bg-blue-50/50 border border-blue-100 rounded-lg p-2.5">
              {report.summary}
            </div>

            {/* Findings count */}
            <div className="flex items-center gap-2">
              <Badge variant="teal">{report.total_detections} detected</Badge>
              <Badge variant="outline">{report.confirmed_detections} confirmed</Badge>
            </div>

            {/* Individual findings */}
            {report.findings.length > 0 && (
              <div className="flex flex-col gap-2">
                <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                  Findings
                </span>
                {report.findings.map((finding) => (
                  <div
                    key={finding.finding_number}
                    className="rounded-md border border-slate-100 bg-white p-2.5 text-[10px]"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-slate-700">
                        Finding #{finding.finding_number}
                      </span>
                      {finding.risk ? (
                        <Badge
                          variant={
                            finding.risk.level === "Low" ? "teal" :
                            finding.risk.level === "Moderate" ? "amber" : "red"
                          }
                        >
                          {finding.risk.level === "High" || finding.risk.level === "Refer" ? (
                            <AlertTriangle className="w-2.5 h-2.5" />
                          ) : (
                            <CheckCircle className="w-2.5 h-2.5" />
                          )}
                          {finding.risk.level}
                        </Badge>
                      ) : null}
                    </div>
                    <div className="text-slate-500">
                      Confidence: {(finding.confidence * 100).toFixed(1)}%
                    </div>
                    {finding.abcde && (
                      <div className="text-slate-500 mt-0.5">
                        ABCDE: {finding.abcde.total_score.toFixed(1)}/9
                        {" · "}
                        {finding.abcde.diameter_mm.toFixed(1)}mm
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Download button */}
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              className="w-full"
              id="btn-download-report"
            >
              <Download className="w-3.5 h-3.5" />
              Download JSON
            </Button>

            {/* Disclaimer */}
            <p className="text-[9px] text-slate-400 leading-relaxed text-center">
              AI-assisted screening only. Does not constitute a medical diagnosis.
            </p>
          </div>
        ) : (
          <p className="text-xs text-slate-400 text-center py-2">
            Run an inference, then generate a report.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
