import { Badge } from "@/components/ui/badge";
import type { SystemStatus } from "@/types/api";
import { Activity, Cpu, Zap } from "lucide-react";

interface HeaderProps {
  systemStatus: SystemStatus;
  sessionId: string;
}

export function Header({ systemStatus, sessionId }: HeaderProps) {
  const isActive = systemStatus.status === "active";

  return (
    <header className="h-14 border-b border-slate-200/80 bg-white/80 backdrop-blur-sm flex items-center justify-between px-5 shrink-0">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-teal-600 flex items-center justify-center">
            <Activity className="w-4 h-4 text-white" strokeWidth={2.5} />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold text-slate-900 leading-tight tracking-tight">
              MedSight
            </span>
            <span className="text-[10px] text-slate-400 font-medium leading-tight -mt-0.5">
              AI Lesion Detection
            </span>
          </div>
        </div>

        <div className="w-px h-6 bg-slate-200 mx-1" />

        {/* Session ID */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-slate-400 font-medium">Session</span>
          <Badge variant="outline" className="font-mono text-[11px]">
            {sessionId}
          </Badge>
        </div>
      </div>

      {/* System Status */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-slate-400" />
            <span className="font-medium text-slate-600">
              {systemStatus.model_name}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-500">{systemStatus.device}</span>
          </div>
        </div>

        <Badge variant={isActive ? "success" : "default"}>
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isActive ? "bg-emerald-500 pulse-dot" : "bg-slate-400"
            }`}
          />
          {isActive ? "Active" : systemStatus.status}
        </Badge>
      </div>
    </header>
  );
}
