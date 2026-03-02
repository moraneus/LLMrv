import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Search } from "lucide-react";

import type { OpenRouterModel } from "@/types";

interface ModelComboboxProps {
  models: OpenRouterModel[];
  value: string;
  onChange: (modelId: string) => void;
  disabled?: boolean;
  placeholder?: string;
  "data-testid"?: string;
}

function formatContextLength(length: number | undefined): string | null {
  if (!length) return null;
  if (length >= 1_000_000) return `${(length / 1_000_000).toFixed(1)}M`;
  if (length >= 1_000) return `${Math.round(length / 1_000)}K`;
  return `${length}`;
}

function formatPrice(price: string | undefined): string | null {
  if (!price) return null;
  const num = parseFloat(price);
  if (isNaN(num)) return null;
  // Price is per token, convert to per 1M tokens
  const perMillion = num * 1_000_000;
  if (perMillion < 0.01) return "<$0.01";
  return `$${perMillion.toFixed(2)}`;
}

export default function ModelCombobox({
  models,
  value,
  onChange,
  disabled = false,
  placeholder = "Select a model",
  "data-testid": testId,
}: ModelComboboxProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [highlightIndex, setHighlightIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const selectedModel = models.find((m) => m.id === value);
  const displayText = selectedModel?.name || value || placeholder;

  const filtered = search.trim()
    ? models.filter((m) => {
        const q = search.toLowerCase();
        return (
          m.id.toLowerCase().includes(q) || m.name.toLowerCase().includes(q)
        );
      })
    : models;

  // Reset highlight when filter changes
  useEffect(() => {
    setHighlightIndex(0);
  }, [search]);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (open) {
      requestAnimationFrame(() => searchRef.current?.focus());
    } else {
      setSearch("");
    }
  }, [open]);

  // Click outside to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Scroll highlighted item into view
  useEffect(() => {
    if (!open || !listRef.current) return;
    const item = listRef.current.children[highlightIndex] as
      | HTMLElement
      | undefined;
    item?.scrollIntoView({ block: "nearest" });
  }, [highlightIndex, open]);

  const handleSelect = useCallback(
    (modelId: string) => {
      onChange(modelId);
      setOpen(false);
    },
    [onChange],
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setOpen(false);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIndex((prev) => Math.min(prev + 1, filtered.length - 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIndex((prev) => Math.max(prev - 1, 0));
      return;
    }
    if (e.key === "Enter" && filtered.length > 0) {
      e.preventDefault();
      handleSelect(filtered[highlightIndex].id);
    }
  };

  const safeTestId = (modelId: string) => modelId.replace(/\//g, "-");

  return (
    <div ref={containerRef} className="relative" data-testid={testId}>
      <button
        type="button"
        onClick={() => !disabled && setOpen(!open)}
        aria-disabled={disabled || undefined}
        className={`flex w-full items-center justify-between rounded-lg border border-slate-200 px-3 py-2 text-left text-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400 ${
          disabled
            ? "cursor-not-allowed opacity-50"
            : "cursor-pointer hover:bg-slate-50"
        }`}
        data-testid={testId ? `${testId}-trigger` : undefined}
      >
        <span className={value ? "text-slate-800" : "text-slate-400"}>
          {displayText}
        </span>
        <ChevronDown size={16} className="text-slate-400" />
      </button>

      {open && !disabled && (
        <div
          className="absolute z-50 mt-1 w-full rounded-lg border border-slate-200 bg-white shadow-lg"
          data-testid={testId ? `${testId}-dropdown` : undefined}
        >
          <div className="border-b border-slate-100 p-2">
            <div className="relative">
              <Search
                size={14}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400"
              />
              <input
                ref={searchRef}
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Search models..."
                className="w-full rounded-md border border-slate-200 py-1.5 pl-8 pr-3 text-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
                data-testid={testId ? `${testId}-search` : undefined}
              />
            </div>
          </div>
          <div ref={listRef} className="max-h-64 overflow-y-auto">
            {filtered.length === 0 ? (
              <div className="px-3 py-6 text-center text-sm text-slate-400">
                No models found
              </div>
            ) : (
              filtered.map((m, i) => {
                const ctx = formatContextLength(m.context_length);
                const promptPrice = formatPrice(m.pricing?.prompt);
                const completionPrice = formatPrice(m.pricing?.completion);
                const priceStr =
                  promptPrice && completionPrice
                    ? `${promptPrice}/${completionPrice}`
                    : null;

                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => handleSelect(m.id)}
                    onMouseEnter={() => setHighlightIndex(i)}
                    className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm ${
                      i === highlightIndex
                        ? "bg-blue-50 text-blue-700"
                        : "text-slate-700 hover:bg-slate-50"
                    } ${m.id === value ? "font-medium" : ""}`}
                    data-testid={
                      testId
                        ? `${testId}-option-${safeTestId(m.id)}`
                        : undefined
                    }
                  >
                    <span className="min-w-0 truncate">{m.name}</span>
                    <span className="ml-2 flex shrink-0 items-center gap-2 text-xs text-slate-400">
                      {ctx && (
                        <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono">
                          {ctx}
                        </span>
                      )}
                      {priceStr && <span>{priceStr}</span>}
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
