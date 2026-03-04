import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PropositionCard from "./PropositionCard";
import { createProposition } from "../../test/mocks";

describe("PropositionCard", () => {
  it("renders prop_id with monospace styling", () => {
    render(
      <PropositionCard
        proposition={createProposition()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(screen.getByText("p_weapon")).toBeInTheDocument();
    expect(screen.getByText("p_weapon")).toHaveClass("font-mono");
  });

  it("renders role badge for user proposition", () => {
    render(
      <PropositionCard
        proposition={createProposition({ role: "user" })}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(screen.getByText("user")).toBeInTheDocument();
  });

  it("renders role badge for assistant proposition", () => {
    render(
      <PropositionCard
        proposition={createProposition({
          role: "assistant",
          prop_id: "q_comply",
        })}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(screen.getByText("assistant")).toBeInTheDocument();
  });

  it("renders description text", () => {
    render(
      <PropositionCard
        proposition={createProposition({
          description: "Requests weapon instructions",
        })}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(
      screen.getByText("Requests weapon instructions"),
    ).toBeInTheDocument();
  });

  it("has data-testid with prop_id", () => {
    render(
      <PropositionCard
        proposition={createProposition({ prop_id: "p_weapon" })}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(screen.getByTestId("proposition-card-p_weapon")).toBeInTheDocument();
  });

  it("clicking edit button calls onEdit with proposition", async () => {
    const user = userEvent.setup();
    const onEdit = vi.fn();
    const prop = createProposition();
    render(
      <PropositionCard
        proposition={prop}
        onEdit={onEdit}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("edit-proposition-p_weapon"));
    expect(onEdit).toHaveBeenCalledWith(prop);
  });

  it("clicking delete button calls onDelete with prop_id", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    render(
      <PropositionCard
        proposition={createProposition()}
        onEdit={vi.fn()}
        onDelete={onDelete}
        onViewPrompt={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("delete-proposition-p_weapon"));
    expect(onDelete).toHaveBeenCalledWith("p_weapon");
  });

  it("edit button has accessible label", () => {
    render(
      <PropositionCard
        proposition={createProposition()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(
      screen.getByLabelText("Edit proposition p_weapon"),
    ).toBeInTheDocument();
  });

  it("delete button has accessible label", () => {
    render(
      <PropositionCard
        proposition={createProposition()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(
      screen.getByLabelText("Delete proposition p_weapon"),
    ).toBeInTheDocument();
  });
});
