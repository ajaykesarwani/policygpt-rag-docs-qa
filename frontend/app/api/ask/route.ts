import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const resp = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const rawText = await resp.text();
      return new Response(rawText, { status: resp.status, headers: { "Content-Type": "application/json" } });
    }

    return new Response(resp.body, {
      status: resp.status,
      headers: {
        "Content-Type": "application/x-ndjson",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  } catch (err: any) {
    return NextResponse.json(
      { detail: err?.message || "Unexpected error in /api/ask" },
      { status: 500 }
    );
  }
}