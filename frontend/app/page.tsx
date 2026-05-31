"use client";

import Layout from "../components/Layout";
import Chat from "../components/Chat";
import UploadForm from "../components/UploadForm";

export default function HomePage() {
  return (
    <Layout>
      <UploadForm />
      <Chat />
    </Layout>
  );
}