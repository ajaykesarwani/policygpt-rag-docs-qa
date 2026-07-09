"use client";

import React, { useState, useRef, useEffect } from "react";

type ContextChunk = {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, any>;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  context?: ContextChunk[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    cost_usd: number;
  };
};

export default function Chat() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userQuery = query.trim();
    setQuery("");
    setLoading(true);
    setError(null);

    const userMessage: Message = { role: "user", content: userQuery };
    const chatHistoryPayload = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    setMessages((prev) => [...prev, userMessage]);

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userQuery,
          history: chatHistoryPayload,
          top_k: 5,
        }),
      });

      if (!resp.ok) {
        const errorText = await resp.text();
        let errorData: any = {};
        try {
          errorData = JSON.parse(errorText);
        } catch {
          errorData = { message: errorText };
        }
        throw new Error(errorData.detail || errorData.message || "Query failed");
      }

      if (!resp.body) {
        throw new Error("No response body available for streaming");
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      // Add placeholder assistant message
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.type === "context") {
              setMessages((prev) => {
                if (prev.length === 0) return prev;
                const last = prev[prev.length - 1];
                if (last && last.role === "assistant") {
                  const updatedLast = { ...last, context: data.context };
                  return [...prev.slice(0, -1), updatedLast];
                }
                return prev;
              });
            } else if (data.type === "token") {
              setMessages((prev) => {
                if (prev.length === 0) return prev;
                const last = prev[prev.length - 1];
                if (last && last.role === "assistant") {
                  const updatedLast = { ...last, content: last.content + data.content };
                  return [...prev.slice(0, -1), updatedLast];
                }
                return prev;
              });
            } else if (data.type === "usage") {
              setMessages((prev) => {
                if (prev.length === 0) return prev;
                const last = prev[prev.length - 1];
                if (last && last.role === "assistant") {
                  const updatedLast = { ...last, usage: data };
                  return [...prev.slice(0, -1), updatedLast];
                }
                return prev;
              });
            } else if (data.type === "error") {
              setError(data.message);
            }
          } catch (err) {
            console.error("Error parsing NDJSON stream chunk:", err);
          }
        }
      }
    } catch (err: any) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setMessages([]);
    setError(null);
  }

  return (
    <section className="panel">
      <div className="panel__header" style={{ borderBottom: "1px solid var(--border-subtle)", paddingBottom: "10px", marginBottom: "14px" }}>
        <div>
          <h2 className="panel__title">Ask a question</h2>
          <p className="panel__hint">
            The assistant retains history and answers queries using your uploaded documents.
          </p>
        </div>
        {messages.length > 0 && (
          <button
            type="button"
            onClick={handleClear}
            className="button button--secondary"
            style={{ padding: "4px 10px", borderRadius: "6px" }}
          >
            Clear conversation
          </button>
        )}
      </div>

      <div className="panel__body">
        {/* Chat Thread */}
        <div className="chat-thread">
          {messages.length === 0 ? (
            <div className="chat-thread__empty">
              No messages yet. Ask a question about your documents to start the conversation!
            </div>
          ) : (
            messages.map((m, idx) => (
              <div key={idx} className={`chat-bubble-container ${m.role}`}>
                <div className="chat-bubble__avatar">
                  {m.role === "user" ? "U" : "AI"}
                </div>
                <div className="chat-bubble__content">
                  <div className="chat-bubble__message">
                    {m.content || (loading && idx === messages.length - 1 ? (
                      <span className="chat-bubble__loading">Thinking...</span>
                    ) : "")}
                  </div>

                  {m.context && m.context.length > 0 && (
                    <div className="chat-context">
                      <div className="chat-context__label">Context used:</div>
                      <div className="chat-context__list">
                        {m.context.map((c) => (
                          <div className="chat-context__item" key={c.id}>
                            <div className="chat-context__meta">
                              score={c.score.toFixed(3)}
                              {c.metadata?.source ? ` · source=${c.metadata.source}` : ""}
                              {c.metadata?.filename ? ` · file=${c.metadata.filename}` : ""}
                            </div>
                            <div>{c.text}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {m.usage && (
                    <div className="chat-bubble__usage">
                      Tokens: {m.usage.total_tokens} (prompt: {m.usage.prompt_tokens}, completion: {m.usage.completion_tokens}) · Cost: ${m.usage.cost_usd.toFixed(5)}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {error && <div className="status status--error" style={{ marginBottom: "10px" }}>Error: {error}</div>}

        {/* Input Form */}
        <form onSubmit={handleAsk}>
          <div className="field-group">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="field-group__textarea"
              rows={3}
              placeholder="Ask a question (e.g., What is our remote work policy?)..."
              disabled={loading}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleAsk(e);
                }
              }}
            />
          </div>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="button button--primary"
            style={{ width: "100%", borderRadius: "8px" }}
          >
            {loading ? "Thinking..." : "Send Message"}
          </button>
        </form>
      </div>
    </section>
  );
}