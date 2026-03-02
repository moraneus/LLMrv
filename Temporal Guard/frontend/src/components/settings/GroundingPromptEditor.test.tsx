import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createSettings } from "../../test/mocks";
import GroundingPromptEditor from "./GroundingPromptEditor";
import type { AppSettings } from "../../types";

function renderEditor(
  overrides: {
    settings?: AppSettings;
    onUpdate?: () => void;
  } = {},
) {
  const props = {
    settings: overrides.settings ?? createSettings(),
    onUpdate: overrides.onUpdate ?? vi.fn(),
  };
  return { ...render(<GroundingPromptEditor {...props} />), props };
}

describe("GroundingPromptEditor", () => {
  // --- Rendering tests ---

  it("renders the editor container", () => {
    renderEditor();
    expect(screen.getByTestId("grounding-prompt-editor")).toBeInTheDocument();
  });

  it('renders heading "Grounding Prompt"', () => {
    renderEditor();
    expect(screen.getByText("Grounding Prompt")).toBeInTheDocument();
  });

  it("renders system prompt textarea with current value", () => {
    const settings = createSettings({
      grounding: {
        provider: "ollama",
        base_url: "http://localhost:11434",
        model: "mistral",
        system_prompt: "Custom system prompt text",
        user_prompt_template: "Template text",
        api_key: "",
      },
    });
    renderEditor({ settings });
    const textarea = screen.getByTestId("system-prompt-textarea");
    expect(textarea).toHaveValue("Custom system prompt text");
  });

  it("renders user prompt textarea with current value", () => {
    const settings = createSettings({
      grounding: {
        provider: "ollama",
        base_url: "http://localhost:11434",
        model: "mistral",
        system_prompt: "System text",
        user_prompt_template: "Custom user template",
        api_key: "",
      },
    });
    renderEditor({ settings });
    const textarea = screen.getByTestId("user-prompt-textarea");
    expect(textarea).toHaveValue("Custom user template");
  });

  it("renders template variables help text", () => {
    renderEditor();
    expect(screen.getByText(/Template variables:/)).toBeInTheDocument();
  });

  // --- Interaction tests ---

  it('clicking "Reset to Default" resets both textareas to default values', async () => {
    const user = userEvent.setup();
    renderEditor();

    // Modify both textareas
    const systemTextarea = screen.getByTestId("system-prompt-textarea");
    const userTextarea = screen.getByTestId("user-prompt-textarea");
    await user.clear(systemTextarea);
    await user.type(systemTextarea, "Modified system prompt");
    await user.clear(userTextarea);
    await user.type(userTextarea, "Modified user prompt");

    // Click reset
    await user.click(screen.getByTestId("reset-prompts"));

    // Verify reset to defaults (the default system prompt starts with "You are a precise content classifier")
    const systemValue = (systemTextarea as HTMLTextAreaElement).value;
    const userValue = (userTextarea as HTMLTextAreaElement).value;
    expect(systemValue).toContain("You are a precise content classifier");
    expect(userValue).toContain("PROPOSITION:");
  });

  it('clicking "Save Changes" calls onUpdate with updated prompts', async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    renderEditor({ onUpdate });

    const systemTextarea = screen.getByTestId("system-prompt-textarea");
    await user.clear(systemTextarea);
    await user.type(systemTextarea, "New system prompt");

    await user.click(screen.getByTestId("save-prompts"));

    expect(onUpdate).toHaveBeenCalledTimes(1);
    const calledWith = onUpdate.mock.calls[0][0] as AppSettings;
    expect(calledWith.grounding.system_prompt).toBe("New system prompt");
  });

  it("editing textareas does not call onUpdate until save is clicked", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    renderEditor({ onUpdate });

    const systemTextarea = screen.getByTestId("system-prompt-textarea");
    await user.clear(systemTextarea);
    await user.type(systemTextarea, "Changed text");

    expect(onUpdate).not.toHaveBeenCalled();
  });
});
