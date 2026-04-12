import type { TranscribeResponse, StreamToken, BenchmarkResponse, HealthResponse } from "@/types/api";

const API_URL = import.meta.env.VITE_API_URL || "";

/**
 * POST /transcribe — upload audio file, get full pipeline result
 */
export async function transcribeAudio(file: File): Promise<TranscribeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/transcribe`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error(`Transcription failed: ${res.statusText}`);
  return res.json();
}

/**
 * GET /stream/token — get single-use token for WebSocket auth
 */
export async function getStreamToken(): Promise<StreamToken> {
  const res = await fetch(`${API_URL}/stream/token`);
  if (!res.ok) throw new Error(`Failed to get stream token: ${res.statusText}`);
  return res.json();
}

/**
 * GET /benchmark — run benchmark and get WER results
 */
export async function fetchBenchmark(filter: "all" | "adversarial" | "standard" = "all"): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_URL}/benchmark?clips=${filter}`);
  if (!res.ok) throw new Error(`Benchmark fetch failed: ${res.statusText}`);
  return res.json();
}

/**
 * GET /health — check backend health
 */
export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

/**
 * Create WebSocket connection for live streaming
 */
export function createStreamSocket(token: string): WebSocket {
  const wsUrl = API_URL.replace(/^http/, "ws");
  return new WebSocket(`${wsUrl}/stream?token=${token}`);
}
