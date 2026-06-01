"use client";

import React, { useState } from "react";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [sourceName, setSourceName] = useState("default");
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    setLoading(true);
    setError(null);
    setStatus(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const resp = await fetch(
        `/api/upload?source_name=${encodeURIComponent(sourceName)}`,
        {
          method: "POST",
          body: formData,
        }
      );

      const text = await resp.text();
      let data: any = {};
      try {
        data = JSON.parse(text);
      } catch {
        data = { message: text };
      }

      if (!resp.ok) {
        setError(data.detail || data.message || "Upload failed");
      } else {
        setStatus(
          `Uploaded successfully. Ingested chunks: ${
            data.ingested_chunks ?? "unknown"
          }`
        );
        // Clear the file after success
        setFile(null);
      }
    } catch (err: any) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] || null;
    setFile(selected);
    setError(null);
    setStatus(null);
  }

  return (
    <section className="panel panel--upload">
      <div className="panel__header">
        <div>
          <h2 className="panel__title">Upload documents</h2>
          <p className="panel__hint">
            Add PDFs or text files to your knowledge base. Each upload becomes searchable context for Q&amp;A.
          </p>
        </div>
      </div>

      <div className="panel__body">
        <form onSubmit={handleSubmit}>
          <div className="field-group">
            <span className="field-group__label">Source name</span>
            <input
              type="text"
              value={sourceName}
              onChange={(e) => setSourceName(e.target.value)}
              className="field-group__input"
              placeholder="e.g. policies, handbook, benefits"
            />
          </div>

          <div className="field-group">
            <span className="field-group__label">File (PDF or TXT)</span>
            {/* Drag-and-drop style zone wrapping the native input */}
            <label className="upload-dropzone">
              <span>
                {file
                  ? `Selected: ${file.name}`
                  : "Drop a PDF or TXT here, or click to browse"}
              </span>
              <input
                type="file"
                accept=".pdf,.txt"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
            </label>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="button button--primary"
          >
            {loading ? "Uploading..." : "Upload & ingest"}
          </button>
        </form>

        {status && <div className="status status--success">{status}</div>}
        {error && <div className="status status--error">Error: {error}</div>}
      </div>
    </section>
  );
}