import { FileText, Pencil, Trash2 } from "lucide-react";

import Badge from "@/components/shared/Badge";
import type { Proposition } from "@/types";

interface PropositionCardProps {
  proposition: Proposition;
  onEdit: (proposition: Proposition) => void;
  onDelete: (propId: string) => void;
  onViewPrompt: (proposition: Proposition) => void;
}

export default function PropositionCard({
  proposition,
  onEdit,
  onDelete,
  onViewPrompt,
}: PropositionCardProps) {
  return (
    <div
      className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
      data-testid={`proposition-card-${proposition.prop_id}`}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-semibold text-blue-600">
            {proposition.prop_id}
          </span>
          <Badge variant={proposition.role === "user" ? "info" : "neutral"}>
            {proposition.role}
          </Badge>
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => onViewPrompt(proposition)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
            aria-label={`View grounding prompt for ${proposition.prop_id}`}
            data-testid={`view-prompt-${proposition.prop_id}`}
          >
            <FileText size={14} />
          </button>
          <button
            onClick={() => onEdit(proposition)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
            aria-label={`Edit proposition ${proposition.prop_id}`}
            data-testid={`edit-proposition-${proposition.prop_id}`}
          >
            <Pencil size={14} />
          </button>
          <button
            onClick={() => onDelete(proposition.prop_id)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-red-50 hover:text-red-500"
            aria-label={`Delete proposition ${proposition.prop_id}`}
            data-testid={`delete-proposition-${proposition.prop_id}`}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>
      <p className="text-sm text-slate-600">{proposition.description}</p>
      {proposition.few_shot_generated_at && (
        <p className="mt-2 text-xs text-slate-400">
          Few-shots generated: {new Date(proposition.few_shot_generated_at).toLocaleString()}
        </p>
      )}
    </div>
  );
}
