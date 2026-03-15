import { useState } from "react";
import { Loader2, Plus } from "lucide-react";

import { getPropositionGroundingPrompt } from "@/api/client";
import Modal from "@/components/shared/Modal";
import { usePolicies } from "@/hooks/usePolicies";
import type { GroundingPromptPreview, Proposition } from "@/types";
import FormulaBuilder from "./FormulaBuilder";
import PropositionCard from "./PropositionCard";
import PropositionEditor from "./PropositionEditor";
import RuleCard from "./RuleCard";

export default function RulesView() {
  const {
    propositions,
    policies,
    createProposition,
    updateProposition,
    deleteProposition,
    createPolicy,
    deletePolicy,
    togglePolicy,
    validateFormula,
  } = usePolicies();

  const [showPropEditor, setShowPropEditor] = useState(false);
  const [editingProp, setEditingProp] = useState<Proposition | null>(null);
  const [showFormulaBuilder, setShowFormulaBuilder] = useState(false);
  const [showPromptModal, setShowPromptModal] = useState(false);
  const [promptPreview, setPromptPreview] = useState<GroundingPromptPreview | null>(
    null,
  );
  const [promptLoading, setPromptLoading] = useState(false);
  const [promptError, setPromptError] = useState<string | null>(null);

  const handleSaveProp = async (data: {
    prop_id: string;
    description: string;
    role: string;
  }) => {
    if (editingProp) {
      await updateProposition(editingProp.prop_id, {
        description: data.description,
        role: data.role,
      });
    } else {
      await createProposition(data);
    }
    setShowPropEditor(false);
    setEditingProp(null);
  };

  const handleEditProp = (proposition: Proposition) => {
    setEditingProp(proposition);
    setShowPropEditor(true);
  };

  const handleDeleteProp = async (propId: string) => {
    await deleteProposition(propId);
  };

  const handleViewPrompt = async (proposition: Proposition) => {
    setShowPromptModal(true);
    setPromptLoading(true);
    setPromptError(null);
    setPromptPreview(null);
    try {
      const preview = await getPropositionGroundingPrompt(proposition.prop_id);
      setPromptPreview(preview);
    } catch (err) {
      setPromptError(
        err instanceof Error
          ? err.message
          : "Failed to load grounding prompt preview",
      );
    } finally {
      setPromptLoading(false);
    }
  };

  const handleSavePolicy = async (data: {
    name: string;
    formula_str: string;
  }) => {
    await createPolicy({ ...data, enabled: true });
    setShowFormulaBuilder(false);
  };

  const handleDeletePolicy = async (policyId: string) => {
    await deletePolicy(policyId);
  };

  const propsLoading =
    propositions.status === "loading" || propositions.status === "idle";
  const policiesLoading =
    policies.status === "loading" || policies.status === "idle";
  const propsList = propositions.status === "success" ? propositions.data : [];
  const policiesList = policies.status === "success" ? policies.data : [];

  if (propsLoading && policiesLoading) {
    return (
      <div
        className="flex h-full items-center justify-center"
        data-testid="rules-loading"
      >
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8 p-6" data-testid="rules-view">
      {/* Propositions Section */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-slate-800">Propositions</h2>
          <button
            onClick={() => {
              setEditingProp(null);
              setShowPropEditor(true);
            }}
            className="flex items-center gap-1.5 rounded-lg bg-blue-500 px-3 py-2 text-sm font-medium text-white hover:bg-blue-600"
            aria-label="Add proposition"
            data-testid="add-proposition"
          >
            <Plus size={16} />
            Add
          </button>
        </div>

        {propositions.status === "error" && (
          <p className="text-sm text-red-500" data-testid="propositions-error">
            {propositions.error}
          </p>
        )}

        {propsList.length === 0 && !propsLoading && (
          <p className="text-sm text-slate-400" data-testid="no-propositions">
            No propositions defined yet. Click "Add" to create one.
          </p>
        )}

        <div className="space-y-3">
          {propsList.map((p) => (
            <PropositionCard
              key={p.prop_id}
              proposition={p}
              onEdit={handleEditProp}
              onDelete={handleDeleteProp}
              onViewPrompt={handleViewPrompt}
            />
          ))}
        </div>
      </section>

      {/* Policies Section */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-slate-800">Policies</h2>
          <button
            onClick={() => setShowFormulaBuilder(true)}
            className="flex items-center gap-1.5 rounded-lg bg-blue-500 px-3 py-2 text-sm font-medium text-white hover:bg-blue-600"
            aria-label="Add policy"
            data-testid="add-policy"
          >
            <Plus size={16} />
            Add
          </button>
        </div>

        {policies.status === "error" && (
          <p className="text-sm text-red-500" data-testid="policies-error">
            {policies.error}
          </p>
        )}

        {policiesList.length === 0 && !policiesLoading && (
          <p className="text-sm text-slate-400" data-testid="no-policies">
            No policies defined yet.{" "}
            {propsList.length === 0
              ? 'You can still click "Add" and use the built-in proposition user_turn.'
              : 'Click "Add" to create one.'}
          </p>
        )}

        <div className="space-y-3">
          {policiesList.map((p) => (
            <RuleCard
              key={p.policy_id}
              policy={p}
              onToggle={togglePolicy}
              onDelete={handleDeletePolicy}
            />
          ))}
        </div>
      </section>

      {/* Proposition Editor Modal */}
      <Modal
        open={showPropEditor}
        onClose={() => {
          setShowPropEditor(false);
          setEditingProp(null);
        }}
        title={editingProp ? "Edit Proposition" : "New Proposition"}
      >
        <PropositionEditor
          initial={editingProp ?? undefined}
          onSave={handleSaveProp}
          onCancel={() => {
            setShowPropEditor(false);
            setEditingProp(null);
          }}
        />
      </Modal>

      {/* Formula Builder Modal */}
      <Modal
        open={showFormulaBuilder}
        onClose={() => setShowFormulaBuilder(false)}
        title="New Policy"
      >
        <FormulaBuilder
          propositions={propsList}
          onSave={handleSavePolicy}
          onCancel={() => setShowFormulaBuilder(false)}
          onValidate={validateFormula}
        />
      </Modal>

      <Modal
        open={showPromptModal}
        onClose={() => {
          setShowPromptModal(false);
          setPromptPreview(null);
          setPromptError(null);
          setPromptLoading(false);
        }}
        title="Grounding Prompt Preview"
      >
        <div className="space-y-4">
          {promptLoading && (
            <div className="flex items-center gap-2 text-sm text-slate-600">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading prompt...
            </div>
          )}
          {promptError && <p className="text-sm text-red-500">{promptError}</p>}
          {promptPreview && (
            <>
              <div>
                <p className="mb-1 text-xs font-semibold uppercase text-slate-500">
                  System Prompt
                </p>
                <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs">
                  {promptPreview.system_prompt}
                </pre>
              </div>
              <div>
                <p className="mb-1 text-xs font-semibold uppercase text-slate-500">
                  User Prompt
                </p>
                <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs">
                  {promptPreview.user_prompt}
                </pre>
              </div>
            </>
          )}
        </div>
      </Modal>
    </div>
  );
}
