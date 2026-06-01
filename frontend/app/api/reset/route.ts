import { NextRequest, NextResponse } from "next/server";

export async function POST(_req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return NextResponse.json(
      { error: "BACKEND_URL is not configured" },
      { status: 500 }
    );
  }

  const res = await fetch(`${backendUrl}/admin/reset`, {
    method: "POST",
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}