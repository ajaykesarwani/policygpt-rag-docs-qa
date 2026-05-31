import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const sourceName = searchParams.get("source_name") || "default";

    const formData = await req.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json(
        { detail: "Missing file field" },
        { status: 400 }
      );
    }

    const backendForm = new FormData();
    backendForm.append("file", file);

    const resp = await fetch(
      `${BACKEND_URL}/upload?source_name=${encodeURIComponent(sourceName)}`,
      {
        method: "POST",
        body: backendForm,
      }
    );

    const contentType = resp.headers.get("content-type") || "";
    const rawText = await resp.text();

    let body: any;
    if (contentType.includes("application/json")) {
      try {
        body = JSON.parse(rawText);
      } catch {
        body = { message: rawText };
      }
    } else {
      body = { message: rawText };
    }

    return NextResponse.json(body, { status: resp.status });
  } catch (err: any) {
    return NextResponse.json(
      { detail: err?.message || "Unexpected error in /api/upload" },
      { status: 500 }
    );
  }
}