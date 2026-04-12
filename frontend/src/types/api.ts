// Types matching the FastAPI backend contract

export type ConfidenceLevel = "LOW" | "MEDIUM" | "HIGH";

export interface RawWord {
  word: string;
  start_ms: number;
  end_ms: number;
  speaker: "Doctor" | "Patient";
  confidence: ConfidenceLevel;
  uncertainty_signals?: string[];
}

export interface CorrectedWord {
  word: string;
  changed: boolean;
  tavily_verified: boolean;
  unverified: boolean;
  speaker: "Doctor" | "Patient";
}

export interface Medication {
  name: string;
  dosage: string;
  frequency: string;
  route: string;
  tavily_verified: boolean;
}

export interface ClinicalSummary {
  medications: Medication[];
  symptoms: string[];
  allergies: string[];
  follow_up_actions: string[];
  appointment_needed: boolean;
}

export interface PipelineLatency {
  preprocessing: number;
  scribe: number;
  uncertainty: number;
  tavily: number;
  claude: number;
  total: number;
}

export interface TranscribeResponse {
  raw_transcript: RawWord[];
  corrected_transcript: CorrectedWord[];
  clinical_summary: ClinicalSummary;
  pipeline_latency_ms: PipelineLatency;
}

export interface StreamToken {
  token: string;
  expires_in: number;
}

export interface StreamFrame {
  type: "partial" | "committed" | "correction" | "error";
  text?: string;
  words?: { word: string; start_ms: number; end_ms: number }[];
  payload?: TranscribeResponse;
  stage?: string;
  message?: string;
}

export interface AblationRow {
  stage: string;
  wer: number;
  delta: number;
  description: string;
}

export interface BenchmarkClipResult {
  clip_id: string;
  category: string;
  difficulty: "Standard" | "Adversarial";
  raw_wer: number;
  corrected_wer: number;
  raw_cer?: number | null;
  corrected_cer?: number | null;
  raw_digit_accuracy?: number | null;
  corrected_digit_accuracy?: number | null;
  raw_medical_keyword_accuracy?: number | null;
  corrected_medical_keyword_accuracy?: number | null;
  improvement_pct: number;
}

export interface BenchmarkMetrics {
  verification_rate: number;
  unsafe_guess_rate: number;
  uncertainty_coverage: number;
  phonetic_hit_rate: number;
  digit_accuracy_coverage?: number | null;
  medical_keyword_accuracy_coverage?: number | null;
}

export interface BenchmarkAggregate {
  avg_raw_wer: number;
  avg_corrected_wer: number;
  avg_raw_cer?: number | null;
  avg_corrected_cer?: number | null;
  avg_raw_digit_accuracy?: number | null;
  avg_corrected_digit_accuracy?: number | null;
  avg_raw_medical_keyword_accuracy?: number | null;
  avg_corrected_medical_keyword_accuracy?: number | null;
  avg_improvement_pct: number;
  keyterm_impact_pct: number;
}

export interface BenchmarkResponse {
  results: BenchmarkClipResult[];
  ablation: AblationRow[];
  metrics: BenchmarkMetrics;
  aggregate: BenchmarkAggregate;
}

export interface HealthResponse {
  status: string;
  redis: string;
  scribe: string;
  tavily: string;
  claude: string;
  learning_loop?: {
    keyterm_count: number;
    phonetic_map_size: number;
  };
  realtime?: string;
}

export type ProcessingStage = "idle" | "uploading" | "preprocessing" | "scribe" | "uncertainty" | "tavily" | "claude" | "done" | "error";
