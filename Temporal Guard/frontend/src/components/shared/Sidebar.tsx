import { MessageSquare, ScrollText, Settings, Shield } from "lucide-react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/rules", label: "Rules", icon: ScrollText },
  { to: "/settings", label: "Settings", icon: Settings },
] as const;

interface SidebarProps {
  monitorStatus?: Record<string, boolean> | null;
}

export default function Sidebar({ monitorStatus }: SidebarProps) {
  const allPassing = monitorStatus
    ? Object.values(monitorStatus).every(Boolean)
    : true;

  return (
    <aside
      className="flex w-64 flex-col border-r border-slate-200 bg-slate-50"
      data-testid="sidebar"
    >
      <div className="flex items-center gap-2 px-5 py-5">
        <Shield className="h-6 w-6 text-blue-500" />
        <span className="text-lg font-semibold text-slate-800">
          TemporalGuard
        </span>
      </div>

      <nav className="flex-1 px-3" data-testid="sidebar-nav">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-blue-50 text-blue-600"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-800"
              }`
            }
            data-testid={`nav-${label.toLowerCase()}`}
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      {monitorStatus && Object.keys(monitorStatus).length > 0 && (
        <div
          className="border-t border-slate-200 px-4 py-4"
          data-testid="monitor-status"
        >
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-400">
            Monitor Status
          </p>
          <div className="flex items-center gap-2 text-sm">
            <span
              className={`h-2 w-2 rounded-full ${allPassing ? "bg-emerald-500" : "bg-red-500"}`}
            />
            <span className={allPassing ? "text-emerald-600" : "text-red-600"}>
              {allPassing ? "All policies passing" : "Violation detected"}
            </span>
          </div>
          <div className="mt-2 space-y-1">
            {Object.entries(monitorStatus).map(([policyId, passing]) => (
              <div
                key={policyId}
                className="flex items-center gap-2 text-xs text-slate-500"
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${passing ? "bg-emerald-400" : "bg-red-400"}`}
                />
                {policyId}: {passing ? "Pass" : "Fail"}
              </div>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}
