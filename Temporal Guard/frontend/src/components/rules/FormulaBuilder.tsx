import { useCallback, useEffect, useRef, useState } from "react";
import { CheckCircle, XCircle } from "lucide-react";

import type { FormulaValidation, Proposition } from "@/types";

interface FormulaBuilderProps {
  propositions: Proposition[];
  onSave: (data: { name: string; formula_str: string }) => void;
  onCancel: () => void;
  onValidate: (name: string, formulaStr: string) => Promise<FormulaValidation>;
}

const operators = [
  { label: "H()", insert: "H()", desc: "Historically" },
  { label: "P()", insert: "P()", desc: "Previously" },
  { label: "Y()", insert: "Y()", desc: "Yesterday" },
  { label: "S", insert: " S ", desc: "Since" },
  { label: "!", insert: "!", desc: "Not" },
  { label: "&", insert: " & ", desc: "And" },
  { label: "|", insert: " | ", desc: "Or" },
  { label: "->", insert: " -> ", desc: "Implies" },
  { label: "(", insert: "(", desc: "Open paren" },
  { label: ")", insert: ")", desc: "Close paren" },
] as const;

const builtInPropositions = [
  {
    prop_id: "user_turn",
    role: "builtin",
    description: "True when the current message is from the user, false otherwise.",
  },
] as const;

export default function FormulaBuilder({
  propositions,
  onSave,
  onCancel,
  onValidate,
}: FormulaBuilderProps) {
  const [name, setName] = useState("");
  const [formula, setFormula] = useState("");
  const [validation, setValidation] = useState<FormulaValidation | null>(null);
  const [validating, setValidating] = useState(false);
  const [saving, setSaving] = useState(false);
  const formulaRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const validate = useCallback(
    async (formulaStr: string) => {
      if (!formulaStr.trim()) {
        setValidation(null);
        return;
      }
      setValidating(true);
      try {
        const result = await onValidate(name || "untitled", formulaStr);
        setValidation(result);
      } catch {
        setValidation({
          valid: false,
          error: "Validation request failed",
          propositions: [],
        });
      } finally {
        setValidating(false);
      }
    },
    [name, onValidate],
  );

  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => validate(formula), 400);
    return () => clearTimeout(debounceRef.current);
  }, [formula, validate]);

  const insertAtCursor = (text: string) => {
    const input = formulaRef.current;
    if (!input) {
      setFormula((prev) => prev + text);
      return;
    }
    const start = input.selectionStart ?? formula.length;
    const end = input.selectionEnd ?? formula.length;
    const next = formula.slice(0, start) + text + formula.slice(end);
    setFormula(next);
    // Move cursor after inserted text
    requestAnimationFrame(() => {
      const pos = start + text.length;
      input.setSelectionRange(pos, pos);
      input.focus();
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validation?.valid || !name.trim()) return;
    setSaving(true);
    try {
      onSave({ name: name.trim(), formula_str: formula.trim() });
    } finally {
      setSaving(false);
    }
  };

  const propositionChips = [
    ...propositions.map((p) => ({
      prop_id: p.prop_id,
      role: p.role,
      description: p.description,
    })),
    ...builtInPropositions,
  ];

  return (
    <form onSubmit={handleSubmit} data-testid="formula-builder">
      <div className="space-y-4">
        <div>
          <label
            className="mb-1 block text-terminal-text font-mono text-sm"
            htmlFor="policy-name"
          >
            Policy Name
          </label>
          <input
            id="policy-name"
            name="policy_name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Fraud Prevention Policy"
            className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
            data-testid="policy-name-input"
          />
        </div>

        {propositionChips.length > 0 && (
          <div>
            <p className="mb-2 text-terminal-text font-mono text-sm">
              Available Propositions
            </p>
            <div
              className="flex flex-wrap gap-1.5"
              data-testid="proposition-chips"
            >
              {propositionChips.map((p) => (
                <button
                  key={p.prop_id}
                  type="button"
                  onClick={() => insertAtCursor(p.prop_id)}
                  className="rounded-none bg-accent-muted border border-accent/20 text-accent font-mono text-xs px-2.5 py-1 hover:bg-accent/15 transition-colors"
                  title={`${p.role}: ${p.description}`}
                  data-testid={`chip-${p.prop_id}`}
                >
                  {p.prop_id}
                </button>
              ))}
            </div>
            <p className="mt-1 text-xs text-terminal-dim">
              Built-in proposition: <code className="font-mono text-accent">user_turn</code>{" "}
              is true on user messages and false otherwise.
            </p>
          </div>
        )}

        <div>
          <p className="mb-2 text-terminal-text font-mono text-sm">Operators</p>
          <div
            className="flex flex-wrap gap-1.5"
            data-testid="operator-buttons"
          >
            {operators.map(({ label, insert, desc }) => (
              <button
                key={label}
                type="button"
                onClick={() => insertAtCursor(insert)}
                className="rounded-none border border-border text-terminal-text font-mono px-2.5 py-1 text-sm hover:bg-dark-hover hover:text-accent transition-colors"
                title={desc}
                data-testid={`op-${label.replace(/[()]/g, "")}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label
            className="mb-1 block text-terminal-text font-mono text-sm"
            htmlFor="formula"
          >
            Formula
          </label>
          <input
            ref={formulaRef}
            id="formula"
            name="formula"
            type="text"
            value={formula}
            onChange={(e) => setFormula(e.target.value)}
            placeholder="H(p_fraud -> !q_comply)"
            className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 font-mono text-sm text-accent placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
            data-testid="formula-input"
          />
          {formula.trim() && !validating && validation && (
            <div
              className="mt-1 flex items-center gap-1.5 text-xs"
              data-testid="formula-validation"
            >
              {validation.valid ? (
                <>
                  <CheckCircle size={14} className="text-terminal-green" />
                  <span className="text-terminal-green">Formula is valid</span>
                </>
              ) : (
                <>
                  <XCircle size={14} className="text-terminal-red" />
                  <span className="text-terminal-red">
                    {validation.error || "Invalid formula"}
                  </span>
                </>
              )}
            </div>
          )}
        </div>

        <div className="rounded-none bg-dark-primary border border-border p-3">
          <p className="mb-1 text-xs font-medium text-terminal-dim">
            Temporal Operators Reference
          </p>
          <div className="space-y-0.5 text-xs text-terminal-dim">
            <p>
              <code className="font-mono text-accent/70">H(φ)</code> = φ held at every past
              step
            </p>
            <p>
              <code className="font-mono text-accent/70">P(φ)</code> = φ held at some past step
            </p>
            <p>
              <code className="font-mono text-accent/70">Y(φ)</code> = φ held at the previous
              step
            </p>
            <p>
              <code className="font-mono text-accent/70">φ S ψ</code> = ψ occurred and φ held
              continuously since
            </p>
          </div>
        </div>

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-none border border-border px-4 py-2 text-sm font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
            data-testid="policy-cancel"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!validation?.valid || !name.trim() || saving}
            className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
            data-testid="policy-save"
          >
            Save Policy
          </button>
        </div>
      </div>
    </form>
  );
}
