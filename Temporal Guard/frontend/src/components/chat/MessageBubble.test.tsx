import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MessageBubble from "./MessageBubble";
import type { ViolationInfo, GroundingDetail } from "../../types";

function createViolation(
  overrides: Partial<ViolationInfo> = {},
): ViolationInfo {
  return {
    policy_id: "pol_weapons",
    policy_name: "Weapons Prohibition",
    formula_str: "H(p_weapon -> !q_comply)",
    violated_at_index: 2,
    labeling: { p_weapon: true, q_comply: true },
    grounding_details: [],
    ...overrides,
  };
}

function createGroundingDetail(
  overrides: Partial<GroundingDetail> = {},
): GroundingDetail {
  return {
    prop_id: "p_weapon",
    match: true,
    confidence: 0.95,
    reasoning: "User explicitly requested weapon instructions",
    method: "llm",
    ...overrides,
  };
}

describe("MessageBubble", () => {
  it("renders user message aligned to the right", () => {
    render(
      <MessageBubble
        role="user"
        content="Hello world"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    const bubble = screen.getByTestId("message-user");
    expect(bubble).toBeInTheDocument();
    expect(bubble.className).toContain("justify-end");
  });

  it("renders assistant message aligned to the left", () => {
    render(
      <MessageBubble
        role="assistant"
        content="Hi there"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    const bubble = screen.getByTestId("message-assistant");
    expect(bubble).toBeInTheDocument();
    expect(bubble.className).toContain("justify-start");
  });

  it("displays message content text", () => {
    render(
      <MessageBubble
        role="user"
        content="Test message content"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    expect(screen.getByText("Test message content")).toBeInTheDocument();
  });

  it('shows "Passed" tag when message is not blocked', () => {
    render(
      <MessageBubble
        role="assistant"
        content="Safe response"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    expect(screen.getByText("Passed")).toBeInTheDocument();
  });

  it('shows "BLOCKED" label and "Blocked" tag when message is blocked', () => {
    render(
      <MessageBubble
        role="assistant"
        content="Dangerous response"
        blocked={true}
        violationInfo={createViolation()}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    expect(screen.getByText("BLOCKED")).toBeInTheDocument();
    expect(screen.getByText("Blocked")).toBeInTheDocument();
    expect(screen.getByTestId("message-blocked")).toBeInTheDocument();
  });

  it("applies line-through style to blocked message content", () => {
    render(
      <MessageBubble
        role="assistant"
        content="Blocked content"
        blocked={true}
        violationInfo={createViolation()}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    const textEl = screen.getByText("Blocked content");
    expect(textEl.className).toContain("line-through");
  });

  it("shows details panel with violation info when toggle is clicked", async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        role="assistant"
        content="Bad response"
        blocked={true}
        violationInfo={createViolation({ policy_name: "Weapons Prohibition" })}
        groundingDetails={null}
        monitorState={null}
      />,
    );

    expect(screen.queryByTestId("message-details")).not.toBeInTheDocument();

    const toggle = screen.getByTestId("toggle-details");
    await user.click(toggle);

    expect(screen.getByTestId("message-details")).toBeInTheDocument();
    expect(
      screen.getByText("Violation: Weapons Prohibition"),
    ).toBeInTheDocument();
    expect(screen.getByText("H(p_weapon -> !q_comply)")).toBeInTheDocument();
  });

  it("shows grounding details in expanded panel", async () => {
    const user = userEvent.setup();
    const details = [
      createGroundingDetail({
        prop_id: "p_weapon",
        match: true,
        confidence: 0.95,
        reasoning: "Weapon request detected",
      }),
      createGroundingDetail({
        prop_id: "q_comply",
        match: false,
        confidence: 0.1,
        reasoning: "Refusal detected",
      }),
    ];

    render(
      <MessageBubble
        role="assistant"
        content="Refusal"
        blocked={false}
        violationInfo={null}
        groundingDetails={details}
        monitorState={null}
      />,
    );

    await user.click(screen.getByTestId("toggle-details"));

    expect(screen.getByText("Grounding:")).toBeInTheDocument();
    expect(screen.getByText("p_weapon")).toBeInTheDocument();
    expect(screen.getByText("Match")).toBeInTheDocument();
    expect(screen.getByText("(95%)")).toBeInTheDocument();
    expect(screen.getByText("Weapon request detected")).toBeInTheDocument();
    expect(screen.getByText("q_comply")).toBeInTheDocument();
    expect(screen.getByText("No match")).toBeInTheDocument();
  });

  it("shows monitor state in expanded panel", async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        role="user"
        content="Hello"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={{ pol_weapons: true, pol_sensitive: false }}
      />,
    );

    await user.click(screen.getByTestId("toggle-details"));

    expect(screen.getByText("Monitor:")).toBeInTheDocument();
    expect(screen.getByText("pol_weapons: Pass")).toBeInTheDocument();
    expect(screen.getByText("pol_sensitive: Fail")).toBeInTheDocument();
  });

  it("hides details panel when toggle is clicked again", async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        role="user"
        content="Hello"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={{ pol_weapons: true }}
      />,
    );

    const toggle = screen.getByTestId("toggle-details");
    await user.click(toggle);
    expect(screen.getByTestId("message-details")).toBeInTheDocument();

    await user.click(toggle);
    expect(screen.queryByTestId("message-details")).not.toBeInTheDocument();
  });
});
