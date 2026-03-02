import { useState } from "react";
import {
  CheckCircle,
  ChevronDown,
  ChevronUp,
  ShieldAlert,
  XCircle,
} from "lucide-react";

import type { GroundingDetail, ViolationInfo } from "@/types";

interface MessageBubbleProps {
  role: string;
  content: string;
  blocked: boolean;
  violationInfo: ViolationInfo | null;
  groundingDetails: GroundingDetail[] | null;
  monitorState: Record<string, boolean> | null;
}

export default function MessageBubble({
  role,
  content,
  blocked,
  violationInfo,
  groundingDetails,
  monitorState,
}: MessageBubbleProps) {
  const [expanded, setExpanded] = useState(false);
  const isUser = role === "user";

  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
      data-testid={`message-${role}`}
    >
      <div
        className={`max-w-[75%] rounded-xl px-4 py-3 ${
          blocked
            ? "border border-red-200 bg-red-50"
            : isUser
              ? "bg-blue-500 text-white"
              : "border border-slate-200 bg-white"
        }`}
        data-testid={blocked ? "message-blocked" : "message-content"}
      >
        {blocked && (
          <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-red-600">
            <ShieldAlert size={14} />
            BLOCKED
          </div>
        )}

        <p
          className={`text-sm whitespace-pre-wrap ${
            blocked
              ? "text-red-400 line-through"
              : isUser
                ? "text-white"
                : "text-slate-700"
          }`}
        >
          {content}
        </p>

        {/* Monitor verdict tag */}
        <div className="mt-2 flex items-center justify-between">
          <div className="flex items-center gap-1 text-xs">
            {blocked ? (
              <>
                <XCircle size={12} className="text-red-500" />
                <span className="text-red-500">Blocked</span>
              </>
            ) : (
              <>
                <CheckCircle
                  size={12}
                  className={isUser ? "text-blue-200" : "text-emerald-500"}
                />
                <span className={isUser ? "text-blue-200" : "text-emerald-500"}>
                  Passed
                </span>
              </>
            )}
          </div>

          {(groundingDetails?.length || violationInfo || monitorState) && (
            <button
              onClick={() => setExpanded(!expanded)}
              className={`flex items-center gap-0.5 text-xs ${
                isUser && !blocked
                  ? "text-blue-200 hover:text-white"
                  : "text-slate-400 hover:text-slate-600"
              }`}
              aria-label={expanded ? "Hide details" : "Show details"}
              data-testid="toggle-details"
            >
              {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              Details
            </button>
          )}
        </div>

        {expanded && (
          <div
            className={`mt-2 border-t pt-2 text-xs ${
              blocked
                ? "border-red-200"
                : isUser
                  ? "border-blue-400"
                  : "border-slate-200"
            }`}
            data-testid="message-details"
          >
            {violationInfo && (
              <div className="mb-2">
                <p className="font-medium text-red-600">
                  Violation: {violationInfo.policy_name}
                </p>
                <p className="font-mono text-red-500">
                  {violationInfo.formula_str}
                </p>
              </div>
            )}

            {groundingDetails && groundingDetails.length > 0 && (
              <div className="space-y-1">
                <p
                  className={`font-medium ${isUser && !blocked ? "text-blue-200" : "text-slate-500"}`}
                >
                  Grounding:
                </p>
                {groundingDetails
                  .filter((g) => g.method !== "monitor_note")
                  .map((g, i) => (
                    <div
                      key={i}
                      className={`rounded-md p-1.5 ${
                        blocked
                          ? "bg-red-100"
                          : isUser
                            ? "bg-blue-600"
                            : "bg-slate-50"
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono">{g.prop_id}</span>
                        <span
                          className={`font-medium ${g.match ? "text-amber-500" : blocked ? "text-red-400" : isUser ? "text-blue-200" : "text-emerald-500"}`}
                        >
                          {g.match ? "Match" : "No match"}
                        </span>
                        {g.method !== "carried_forward" && (
                          <span
                            className={
                              isUser && !blocked
                                ? "text-blue-300"
                                : "text-slate-400"
                            }
                          >
                            ({(g.confidence * 100).toFixed(0)}%)
                          </span>
                        )}
                        {g.method === "carried_forward" && (
                          <span
                            className={
                              isUser && !blocked
                                ? "text-blue-300"
                                : "text-slate-400"
                            }
                          >
                            (carried)
                          </span>
                        )}
                      </div>
                      <p
                        className={
                          isUser && !blocked
                            ? "text-blue-200"
                            : "text-slate-400"
                        }
                      >
                        {g.reasoning}
                      </p>
                    </div>
                  ))}
              </div>
            )}

            {monitorState && Object.keys(monitorState).length > 0 && (
              <div className="mt-1">
                <p
                  className={`font-medium ${isUser && !blocked ? "text-blue-200" : "text-slate-500"}`}
                >
                  Monitor:
                </p>
                <div className="flex flex-wrap gap-1.5 mt-0.5">
                  {Object.entries(monitorState).map(([pid, passing]) => (
                    <span
                      key={pid}
                      className={`rounded-full px-2 py-0.5 ${
                        passing
                          ? "bg-emerald-100 text-emerald-600"
                          : "bg-red-100 text-red-600"
                      }`}
                    >
                      {pid}: {passing ? "Pass" : "Fail"}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
