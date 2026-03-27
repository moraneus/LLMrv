import { useState } from "react";
import { Monitor, User } from "lucide-react";

import type { Proposition } from "@/types";

interface PropositionEditorProps {
  initial?: Proposition;
  onSave: (data: {
    prop_id: string;
    description: string;
    role: string;
  }) => Promise<void> | void;
  onCancel: () => void;
}

export default function PropositionEditor({
  initial,
  onSave,
  onCancel,
}: PropositionEditorProps) {
  const [propId, setPropId] = useState(initial?.prop_id ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [role, setRole] = useState<"user" | "assistant">(
    initial?.role ?? "user",
  );
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const isEdit = !!initial;
  const isValid = propId.trim().length > 0 && description.trim().length > 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValid) return;
    setSaving(true);
    setSaveError(null);
    try {
      await onSave({
        prop_id: propId.trim(),
        description: description.trim(),
        role,
      });
    } catch (err) {
      setSaveError(
        err instanceof Error ? err.message : "Failed to save proposition",
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} data-testid="proposition-editor">
      <div className="space-y-4">
        <div>
          <label
            className="mb-1 block text-terminal-text font-mono text-sm"
            htmlFor="prop-id"
          >
            Proposition ID
          </label>
          <input
            id="prop-id"
            name="prop_id"
            type="text"
            value={propId}
            onChange={(e) => setPropId(e.target.value)}
            disabled={isEdit}
            placeholder="p_fraud"
            className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 font-mono text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20 disabled:opacity-50"
            data-testid="prop-id-input"
          />
        </div>

        <div>
          <label className="mb-2 block text-terminal-text font-mono text-sm">
            Role
          </label>
          <div className="flex gap-4" data-testid="prop-role-select">
            <label className="flex items-center gap-2 text-sm text-terminal-text">
              <input
                type="radio"
                name="role"
                value="user"
                checked={role === "user"}
                onChange={() => setRole("user")}
                className="accent-accent"
                data-testid="prop-role-user"
              />
              <User size={14} className="text-terminal-cyan" />
              User
            </label>
            <label className="flex items-center gap-2 text-sm text-terminal-text">
              <input
                type="radio"
                name="role"
                value="assistant"
                checked={role === "assistant"}
                onChange={() => setRole("assistant")}
                className="accent-accent"
                data-testid="prop-role-assistant"
              />
              <Monitor size={14} className="text-terminal-amber" />
              Assistant
            </label>
          </div>
        </div>

        <div>
          <label
            className="mb-1 block text-terminal-text font-mono text-sm"
            htmlFor="description"
          >
            Description
          </label>
          <textarea
            id="description"
            name="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="The user requests methods for committing financial fraud"
            rows={3}
            className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
            data-testid="prop-description-input"
          />
        </div>

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-none border border-border px-4 py-2 text-sm font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
            data-testid="prop-cancel"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!isValid || saving}
            className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
            data-testid="prop-save"
          >
            {saving
              ? isEdit
                ? "Updating..."
                : "Generating Few-Shots..."
              : isEdit
                ? "Update Proposition"
                : "Save Proposition"}
          </button>
        </div>
        {saving && !isEdit && (
          <p className="text-xs text-terminal-dim" data-testid="prop-generating">
            Generating few-shot examples with the chat model and saving to DB...
          </p>
        )}
        {saveError && (
          <p className="text-sm text-terminal-red" data-testid="prop-save-error">
            {saveError}
          </p>
        )}
      </div>
    </form>
  );
}
