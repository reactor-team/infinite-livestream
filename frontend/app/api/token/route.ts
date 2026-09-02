import { NextResponse } from "next/server";

import { readConfig } from "@/lib/config";

const TOKEN_LIFETIME_SECONDS = 60 * 60;
const MAX_SESSIONS = 10;
const DEFAULT_API_URL = "https://api.reactor.inc";

/** Mint a model-scoped browser token without exposing the account API key. */
export async function GET() {
  const apiKey = process.env.REACTOR_API_KEY?.trim();
  if (!apiKey) {
    return NextResponse.json(
      { error: "REACTOR_API_KEY is not configured." },
      { status: 500 },
    );
  }

  const { modelName, apiUrl } = readConfig();
  const response = await fetch(`${apiUrl ?? DEFAULT_API_URL}/tokens`, {
    method: "POST",
    headers: {
      "Reactor-API-Key": apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      expires_after: TOKEN_LIFETIME_SECONDS,
      authorization_details: [
        {
          type: "session",
          resources: { models: { match: [modelName] } },
          constraints: { max_sessions: MAX_SESSIONS },
        },
      ],
    }),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    return NextResponse.json(
      { error: `Token request failed (${response.status}). ${detail}`.trim() },
      { status: 502 },
    );
  }

  const { jwt, expires_at } = (await response.json()) as {
    jwt: string;
    expires_at: number;
  };
  return NextResponse.json(
    { jwt, expires_at },
    { headers: { "Cache-Control": "private, no-store" } },
  );
}
