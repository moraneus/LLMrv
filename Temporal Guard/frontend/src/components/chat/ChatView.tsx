import { useEffect, useRef, useState } from "react";
import { Loader2, MessageSquarePlus, Pencil, Plus, Trash2 } from "lucide-react";

import { useChat } from "@/hooks/useChat";
import type { ChatResponse } from "@/types";
import MessageBubble from "./MessageBubble";
import MessageInput from "./MessageInput";
import MonitorStatus from "./MonitorStatus";
import ViolationAlert from "./ViolationAlert";

export default function ChatView() {
  const {
    sessions,
    activeSessionId,
    messages,
    sendState,
    lastResponse,
    createSession,
    switchSession,
    deleteSession,
    renameSession,
    sendMessage,
  } = useChat();

  const [violation, setViolation] = useState<ChatResponse | null>(null);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editingHeader, setEditingHeader] = useState(false);
  const [headerEditName, setHeaderEditName] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const sidebarInputRef = useRef<HTMLInputElement>(null);
  const headerInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Show violation alert from last response
  useEffect(() => {
    if (lastResponse?.blocked && lastResponse.violation) {
      setViolation(lastResponse);
    }
  }, [lastResponse]);

  const handleSend = async (message: string) => {
    try {
      await sendMessage(message);
    } catch {
      // Error handled by hook
    }
  };

  // Focus sidebar input when editing starts
  useEffect(() => {
    if (editingSessionId && sidebarInputRef.current) {
      sidebarInputRef.current.focus();
      sidebarInputRef.current.select();
    }
  }, [editingSessionId]);

  // Focus header input when editing starts
  useEffect(() => {
    if (editingHeader && headerInputRef.current) {
      headerInputRef.current.focus();
      headerInputRef.current.select();
    }
  }, [editingHeader]);

  const startSidebarRename = (sessionId: string, currentName: string) => {
    setEditingSessionId(sessionId);
    setEditName(currentName);
  };

  const commitSidebarRename = async () => {
    if (editingSessionId && editName.trim()) {
      await renameSession(editingSessionId, editName.trim());
    }
    setEditingSessionId(null);
  };

  const startHeaderRename = (currentName: string) => {
    setEditingHeader(true);
    setHeaderEditName(currentName);
  };

  const commitHeaderRename = async () => {
    if (activeSessionId && headerEditName.trim()) {
      await renameSession(activeSessionId, headerEditName.trim());
    }
    setEditingHeader(false);
  };

  const sessionsList = sessions.status === "success" ? sessions.data : [];
  const messagesList = messages.status === "success" ? messages.data : [];

  // Get latest monitor state from the most recent message
  const latestMonitorState =
    messagesList.length > 0
      ? messagesList[messagesList.length - 1].monitor_state
      : null;

  return (
    <div className="flex h-full" data-testid="chat-view">
      {/* Session Sidebar */}
      <div
        className="flex w-56 flex-col border-r border-slate-200 bg-slate-50"
        data-testid="session-list"
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-3 py-3">
          <span className="text-sm font-medium text-slate-600">Sessions</span>
          <button
            onClick={createSession}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-blue-500"
            aria-label="New session"
            data-testid="new-session"
          >
            <Plus size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {sessions.status === "loading" && (
            <div className="flex justify-center p-4">
              <Loader2 size={18} className="animate-spin text-blue-500" />
            </div>
          )}
          {sessionsList.length === 0 && sessions.status === "success" && (
            <div className="p-4 text-center">
              <p className="text-xs text-slate-400">No sessions yet</p>
              <button
                onClick={createSession}
                className="mt-2 flex items-center gap-1 mx-auto text-xs text-blue-500 hover:text-blue-600"
                data-testid="create-first-session"
              >
                <MessageSquarePlus size={14} />
                Start chatting
              </button>
            </div>
          )}
          {sessionsList.map((s) => (
            <div
              key={s.session_id}
              className={`group flex items-center justify-between gap-1 px-3 py-2 text-sm cursor-pointer ${
                activeSessionId === s.session_id
                  ? "bg-blue-50 text-blue-600"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
              onClick={() => {
                if (editingSessionId !== s.session_id)
                  switchSession(s.session_id);
              }}
              data-testid={`session-${s.session_id}`}
            >
              {editingSessionId === s.session_id ? (
                <input
                  ref={sidebarInputRef}
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onBlur={commitSidebarRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitSidebarRename();
                    if (e.key === "Escape") setEditingSessionId(null);
                  }}
                  onClick={(e) => e.stopPropagation()}
                  className="min-w-0 flex-1 rounded border border-blue-300 bg-white px-1.5 py-0.5 text-sm text-slate-700 outline-none focus:ring-1 focus:ring-blue-400"
                  data-testid={`rename-input-${s.session_id}`}
                />
              ) : (
                <span className="min-w-0 flex-1 truncate">
                  {s.name || `Session ${s.session_id.slice(0, 8)}`}
                </span>
              )}
              <div className="flex shrink-0 items-center gap-0.5">
                {editingSessionId !== s.session_id && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      startSidebarRename(
                        s.session_id,
                        s.name || `Session ${s.session_id.slice(0, 8)}`,
                      );
                    }}
                    className="hidden rounded p-0.5 text-slate-400 hover:bg-blue-50 hover:text-blue-500 group-hover:block"
                    aria-label={`Rename session ${s.session_id}`}
                    data-testid={`rename-session-${s.session_id}`}
                  >
                    <Pencil size={13} />
                  </button>
                )}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(s.session_id);
                  }}
                  className="hidden rounded p-0.5 text-slate-400 hover:bg-red-50 hover:text-red-500 group-hover:block"
                  aria-label={`Delete session ${s.session_id}`}
                  data-testid={`delete-session-${s.session_id}`}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex flex-1 flex-col">
        {!activeSessionId ? (
          <div className="flex flex-1 items-center justify-center">
            <div className="text-center">
              <MessageSquarePlus className="mx-auto h-12 w-12 text-slate-300" />
              <p className="mt-3 text-sm text-slate-400">
                Select a session or create a new one to start chatting
              </p>
              <button
                onClick={createSession}
                className="mt-3 rounded-lg bg-blue-500 px-4 py-2 text-sm font-medium text-white hover:bg-blue-600"
                data-testid="create-session-cta"
              >
                New Session
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
              {editingHeader ? (
                <input
                  ref={headerInputRef}
                  type="text"
                  value={headerEditName}
                  onChange={(e) => setHeaderEditName(e.target.value)}
                  onBlur={commitHeaderRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitHeaderRename();
                    if (e.key === "Escape") setEditingHeader(false);
                  }}
                  className="rounded border border-blue-300 px-2 py-0.5 text-sm font-medium text-slate-700 outline-none focus:ring-1 focus:ring-blue-400"
                  data-testid="header-rename-input"
                />
              ) : (
                <button
                  onClick={() =>
                    startHeaderRename(
                      sessionsList.find((s) => s.session_id === activeSessionId)
                        ?.name || `Session ${activeSessionId.slice(0, 8)}`,
                    )
                  }
                  className="group flex items-center gap-1.5 rounded px-1 py-0.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
                  title="Click to rename"
                  data-testid="header-session-name"
                >
                  <span>
                    {sessionsList.find((s) => s.session_id === activeSessionId)
                      ?.name || `Session ${activeSessionId.slice(0, 8)}`}
                  </span>
                  <Pencil
                    size={13}
                    className="text-slate-400 opacity-0 group-hover:opacity-100"
                  />
                </button>
              )}
              <MonitorStatus monitorState={latestMonitorState} />
            </div>

            {/* Messages */}
            <div
              ref={scrollRef}
              className="flex-1 space-y-4 overflow-y-auto px-4 py-4"
              data-testid="message-list"
            >
              {messages.status === "loading" && (
                <div className="flex justify-center">
                  <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
                </div>
              )}
              {messagesList.length === 0 && messages.status === "success" && (
                <div className="flex h-full items-center justify-center">
                  <p className="text-sm text-slate-400">
                    Send a message to start the conversation
                  </p>
                </div>
              )}
              {messagesList.map((msg) => (
                <MessageBubble
                  key={msg.id}
                  role={msg.role}
                  content={msg.content}
                  blocked={msg.blocked}
                  violationInfo={msg.violation_info}
                  groundingDetails={msg.grounding_details}
                  monitorState={msg.monitor_state}
                />
              ))}
            </div>

            {/* Violation Alert */}
            {violation && violation.violation && (
              <ViolationAlert
                violation={violation.violation}
                blockedResponse={violation.blocked_response}
                onDismiss={() => setViolation(null)}
              />
            )}

            {/* Input */}
            <MessageInput
              onSend={handleSend}
              disabled={!activeSessionId}
              sending={sendState === "sending"}
            />
          </>
        )}
      </div>
    </div>
  );
}
