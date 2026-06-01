import React from "react";

type Props = {
  children: React.ReactNode;
};

export default function Layout({ children }: Props) {
  return (
    <>
      <header className="app-header">
        <div className="app-header__title-block">
          <div className="app-header__badge">
            <span className="app-header__badge-name">PolicyGPT</span>
            <span className="app-header__badge-tagline">
              An AI assistant that answers company policy questions using only your internal documents.
            </span>
          </div>

          <h1 className="app-header__title">RAG Document Q&amp;A workspace</h1>

          <p className="app-header__subtitle">
            Upload internal PDFs or text files, then ask grounded questions. The assistant answers strictly from
            retrieved context and shows you the source snippets it used.
          </p>
        </div>

        <div className="app-header__pill">
          <span className="tech-pill">RAG pipeline</span>
          <span className="tech-pill">Groq</span>
          <span className="tech-pill">local embeddings</span>
          <span className="tech-pill">FastAPI</span>
          <span className="tech-pill">Next.js</span>
        </div>
      </header>

      <main className="app-main-grid">{children}</main>

      <footer style={{ marginTop: 24, fontSize: 11, color: "#6b7280" }}>
        Built by Ajay Kesarwani · Powered by Groq, FastAPI &amp; Next.js
      </footer>
    </>
  );
}