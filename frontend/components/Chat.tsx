﻿"use client";

import React, { useState } from "react";

type ContextChunk = {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, any>;
};

type QueryResponse = {
  answer: string;
  context: ContextChunk[];
};

export default function Chat() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });

      const text = await resp.text();
      let data: any = {};
      try {
        data = JSON.parse(text);
      } catch {
        data = { message: text };
      }

      if (!resp.ok) {
        setError(data.detail || data.message || "Query failed");
      } else {
        setResponse(data as QueryResponse);
      }
    } catch (err: any) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <h2 className="panel__title">Ask a question</h2>
          <p className="panel__hint">
            Queries are answered using only your ingested documents. Check the context section to see which snippets were used.
          </p>
        </div>
      </div>

      <div className="panel__body">
        <form onSubmit={handleAsk} style={{ marginBottom: 10 }}>
          <div className="field-group">
            <span className="field-group__label">Question</span>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="field-group__textarea"
              rows={3}
              placeholder="Example: What does our vacation policy say about carry-over days?"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="button button--primary"
          >
            {loading ? "Thinking..." : "Ask"}
          </button>
        </form>

        {error && <div className="status status--error">Error: {error}</div>}

        {response && (
          <>
            <div style={{ marginTop: 12 }}>
              <div className="field-group__label">Answer</div>
              <div className="chat-answer">{response.answer}</div>
            </div>

            <div className="chat-context">
              <div className="field-group__label">Context used</div>
              {response.context.map((c) => (
                <div className="chat-context__item" key={c.id}>
                  <div className="chat-context__meta">
                    score={c.score.toFixed(3)}
                    {c.metadata?.source ? ` · source=${c.metadata.source}` : ""}
                  </div>
                  <div>{c.text}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}