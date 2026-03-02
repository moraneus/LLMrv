import { useRef, useState } from "react";
import { Loader2, Send } from "lucide-react";

interface MessageInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
  sending: boolean;
}

export default function MessageInput({
  onSend,
  disabled,
  sending,
}: MessageInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || disabled || sending) return;
    onSend(text.trim());
    setText("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInput = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t border-slate-200 bg-white px-4 py-3"
      data-testid="message-input-form"
    >
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Type a message..."
          rows={1}
          disabled={disabled || sending}
          className="flex-1 resize-none rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400 disabled:bg-slate-50"
          data-testid="message-input"
        />
        <button
          type="submit"
          disabled={!text.trim() || disabled || sending}
          className="rounded-lg bg-blue-500 p-2.5 text-white hover:bg-blue-600 disabled:opacity-50"
          aria-label="Send"
          data-testid="send-button"
        >
          {sending ? (
            <Loader2 size={18} className="animate-spin" />
          ) : (
            <Send size={18} />
          )}
        </button>
      </div>
    </form>
  );
}
