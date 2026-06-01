"use client";

import { useState } from "react";
import Layout from "../components/Layout";
import Chat from "../components/Chat";
import UploadForm from "../components/UploadForm";

export default function HomePage() {
  const [isResetting, setIsResetting] = useState(false);

  async function handleResetKnowledge() {
    if (!confirm("This will delete all uploaded documents. Continue?")) return;

    try {
      setIsResetting(true);
      const res = await fetch("/api/reset", {
        method: "POST",
      });

      if (!res.ok) {
        alert("Failed to reset knowledge");
        return;
      }

      alert("Knowledge reset successfully");
    } catch (err) {
      console.error(err);
      alert("Failed to reset knowledge");
    } finally {
      setIsResetting(false);
    }
  }

  return (
    <Layout>
      <UploadForm />
      <div>
        <Chat />
        <div style={{ marginTop: 16 }}>
          <button
            type="button"
            onClick={handleResetKnowledge}
            disabled={isResetting}
            className="button button--danger"
          >
            {isResetting ? "Resetting..." : "Reset knowledge"}
          </button>
        </div>
      </div>
    </Layout>
  );
}