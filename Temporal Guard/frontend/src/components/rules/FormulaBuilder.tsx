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
            className="mb-1 block text-sm font-medium text-slate-600"
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
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
            data-testid="policy-name-input"
          />
        </div>

        {propositionChips.length > 0 && (
          <div>
            <p className="mb-2 text-sm font-medium text-slate-600">
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
                  className="rounded-full bg-blue-50 px-2.5 py-1 font-mono text-xs text-blue-600 hover:bg-blue-100"
                  title={`${p.role}: ${p.description}`}
                  data-testid={`chip-${p.prop_id}`}
                >
                  {p.prop_id}
                </button>
              ))}
            </div>
            <p className="mt-1 text-xs text-slate-400">
              Built-in proposition: <code className="font-mono">user_turn</code>{" "}
              is true on user messages and false otherwise.
            </p>
          </div>
        )}

        <div>
          <p className="mb-2 text-sm font-medium text-slate-600">Operators</p>
          <div
            className="flex flex-wrap gap-1.5"
            data-testid="operator-buttons"
          >
            {operators.map(({ label, insert, desc }) => (
              <button
                key={label}
                type="button"
                onClick={() => insertAtCursor(insert)}
                className="rounded-lg border border-slate-200 px-2.5 py-1 font-mono text-sm text-slate-700 hover:bg-slate-50"
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
            className="mb-1 block text-sm font-medium text-slate-600"
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
            className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
            data-testid="formula-input"
          />
          {formula.trim() && !validating && validation && (
            <div
              className="mt-1 flex items-center gap-1.5 text-xs"
              data-testid="formula-validation"
            >
              {validation.valid ? (
                <>
                  <CheckCircle size={14} className="text-emerald-500" />
                  <span className="text-emerald-600">Formula is valid</span>
                </>
              ) : (
                <>
                  <XCircle size={14} className="text-red-500" />
                  <span className="text-red-600">
                    {validation.error || "Invalid formula"}
                  </span>
                </>
              )}
            </div>
          )}
        </div>

        <div className="rounded-lg bg-slate-50 p-3">
          <p className="mb-1 text-xs font-medium text-slate-500">
            Temporal Operators Reference
          </p>
          <div className="space-y-0.5 text-xs text-slate-400">
            <p>
              <code className="font-mono">H(φ)</code> = φ held at every past
              step
            </p>
            <p>
              <code className="font-mono">P(φ)</code> = φ held at some past step
            </p>
            <p>
              <code className="font-mono">Y(φ)</code> = φ held at the previous
              step
            </p>
            <p>
              <code className="font-mono">φ S ψ</code> = ψ occurred and φ held
              continuously since
            </p>
          </div>
        </div>

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50"
            data-testid="policy-cancel"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!validation?.valid || !name.trim() || saving}
            className="rounded-lg bg-blue-500 px-4 py-2 text-sm font-medium text-white hover:bg-blue-600 disabled:opacity-50"
            data-testid="policy-save"
          >
            Save Policy
          </button>
        </div>
      </div>
    </form>
  );
}
