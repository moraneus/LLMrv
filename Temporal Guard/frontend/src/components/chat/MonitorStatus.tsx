interface MonitorStatusProps {
  monitorState: Record<string, boolean> | null;
}

export default function MonitorStatus({ monitorState }: MonitorStatusProps) {
  if (!monitorState || Object.keys(monitorState).length === 0) {
    return null;
  }

  const allPassing = Object.values(monitorState).every(Boolean);

  return (
    <div
      className="flex items-center gap-2 text-sm"
      data-testid="chat-monitor-status"
    >
      <span
        className={`h-2 w-2 rounded-full ${allPassing ? "bg-emerald-500" : "bg-red-500"}`}
      />
      <span className={allPassing ? "text-emerald-600" : "text-red-600"}>
        {allPassing ? "All policies passing" : "Violation detected"}
      </span>
    </div>
  );
}
