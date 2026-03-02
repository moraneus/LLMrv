import { useEffect, useState } from "react";
import { CheckCircle, Eye, EyeOff, Loader2, XCircle } from "lucide-react";

import ModelCombobox from "@/components/shared/ModelCombobox";
import type { AppSettings, OpenRouterModel } from "@/types";

interface OpenRouterConfigProps {
  settings: AppSettings;
  openRouterModels: OpenRouterModel[];
  modelsError: string | null;
  onUpdate: (settings: AppSettings) => void;
  onFetchModels: () => void;
}

export default function OpenRouterConfig({
  settings,
  openRouterModels,
  modelsError,
  onUpdate,
  onFetchModels,
}: OpenRouterConfigProps) {
  const [showKey, setShowKey] = useState(false);
  const [apiKey, setApiKey] = useState(settings.openrouter_api_key);
  const [model, setModel] = useState(settings.openrouter_model);
  const [useCustomModel, setUseCustomModel] = useState(
    !!settings.openrouter_model_custom,
  );
  const [customModel, setCustomModel] = useState(
    settings.openrouter_model_custom,
  );
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<boolean | null>(null);

  useEffect(() => {
    setApiKey(settings.openrouter_api_key);
    setModel(settings.openrouter_model);
    setUseCustomModel(!!settings.openrouter_model_custom);
    setCustomModel(settings.openrouter_model_custom);
  }, [settings]);

  useEffect(() => {
    if (settings.openrouter_api_key) {
      onFetchModels();
    }
  }, [settings.openrouter_api_key, onFetchModels]);

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      onUpdate({ ...settings, openrouter_api_key: apiKey });
      // Give time for settings to propagate then fetch models
      await new Promise((r) => setTimeout(r, 500));
      onFetchModels();
      setTestResult(true);
    } catch {
      setTestResult(false);
    } finally {
      setTesting(false);
    }
  };

  const handleSave = () => {
    onUpdate({
      ...settings,
      openrouter_api_key: apiKey,
      openrouter_model: model,
      openrouter_model_custom: useCustomModel ? customModel : "",
    });
  };

  return (
    <div
      className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="openrouter-config"
    >
      <h3 className="mb-4 text-base font-semibold text-slate-800">
        Chat Model (OpenRouter)
      </h3>

      <div className="space-y-4">
        <div>
          <label
            className="mb-1 block text-sm font-medium text-slate-600"
            htmlFor="api-key"
          >
            API Key
          </label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <input
                id="api-key"
                type={showKey ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-or-v1-..."
                className="w-full rounded-lg border border-slate-200 px-3 py-2 pr-10 text-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
                data-testid="openrouter-api-key"
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                aria-label={showKey ? "Hide API key" : "Show API key"}
                data-testid="toggle-api-key-visibility"
              >
                {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            <button
              onClick={handleTest}
              disabled={testing || !apiKey}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
              data-testid="test-openrouter"
            >
              {testing ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                "Test"
              )}
            </button>
          </div>
          {testResult !== null && (
            <div className="mt-1 flex items-center gap-1 text-xs">
              {testResult ? (
                <>
                  <CheckCircle size={14} className="text-emerald-500" />
                  <span className="text-emerald-600">Connected</span>
                </>
              ) : (
                <>
                  <XCircle size={14} className="text-red-500" />
                  <span className="text-red-600">Connection failed</span>
                </>
              )}
            </div>
          )}
        </div>

        <div>
          <label
            className="mb-1 block text-sm font-medium text-slate-600"
            htmlFor="or-model"
          >
            Model
          </label>
          <ModelCombobox
            models={openRouterModels}
            value={model}
            onChange={setModel}
            disabled={useCustomModel}
            placeholder="Enter API key to load models"
            data-testid="openrouter-model-select"
          />
          {modelsError && openRouterModels.length === 0 && (
            <p
              className="mt-1 text-xs text-amber-600"
              data-testid="openrouter-models-error"
            >
              {modelsError}
            </p>
          )}
        </div>

        <div>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            <input
              type="checkbox"
              checked={useCustomModel}
              onChange={(e) => {
                setUseCustomModel(e.target.checked);
                if (!e.target.checked) {
                  setCustomModel("");
                }
              }}
              className="rounded border-slate-300"
              data-testid="custom-model-checkbox"
            />
            Use custom model ID
          </label>
          {useCustomModel && (
            <input
              type="text"
              value={customModel}
              onChange={(e) => setCustomModel(e.target.value)}
              placeholder="e.g. anthropic/claude-3-opus"
              className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
              data-testid="custom-model-input"
            />
          )}
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleSave}
            className="rounded-lg bg-blue-500 px-4 py-2 text-sm font-medium text-white hover:bg-blue-600"
            data-testid="save-openrouter"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
