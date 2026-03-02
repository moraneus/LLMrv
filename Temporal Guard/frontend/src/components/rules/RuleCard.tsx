import { Trash2 } from "lucide-react";

import Badge from "@/components/shared/Badge";
import type { Policy } from "@/types";

interface RuleCardProps {
  policy: Policy;
  onToggle: (policyId: string, enabled: boolean) => void;
  onDelete: (policyId: string) => void;
}

export default function RuleCard({
  policy,
  onToggle,
  onDelete,
}: RuleCardProps) {
  return (
    <div
      className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
      data-testid={`policy-card-${policy.policy_id}`}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-800">
            {policy.name}
          </span>
          <Badge variant={policy.enabled ? "success" : "neutral"}>
            {policy.enabled ? "Active" : "Disabled"}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <label className="relative inline-flex cursor-pointer items-center">
            <input
              type="checkbox"
              checked={policy.enabled}
              onChange={(e) => onToggle(policy.policy_id, e.target.checked)}
              className="peer sr-only"
              aria-label={`Toggle policy ${policy.name}`}
              data-testid={`toggle-policy-${policy.policy_id}`}
            />
            <div className="h-5 w-9 rounded-full bg-slate-200 after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:bg-white after:transition-all after:content-[''] peer-checked:bg-blue-500 peer-checked:after:translate-x-full peer-focus:ring-2 peer-focus:ring-blue-300" />
          </label>
          <button
            onClick={() => onDelete(policy.policy_id)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-red-50 hover:text-red-500"
            aria-label={`Delete policy ${policy.name}`}
            data-testid={`delete-policy-${policy.policy_id}`}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      <div className="mb-2 rounded-lg bg-slate-50 px-3 py-2">
        <code
          className="font-mono text-sm text-slate-700"
          data-testid="policy-formula"
        >
          {policy.formula_str}
        </code>
      </div>

      {policy.propositions.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {policy.propositions.map((propId) => (
            <span
              key={propId}
              className="rounded-full bg-blue-50 px-2 py-0.5 font-mono text-xs text-blue-600"
            >
              {propId}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
