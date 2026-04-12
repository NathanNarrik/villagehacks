import { useState, useCallback } from "react";
import { Stethoscope, Pill, Syringe, AlertTriangle, Activity, CheckCircle, HelpCircle, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { MOCK_TRANSCRIBE } from "@/services/mockData";
import type { TranscribeResponse, ProcessingStage, RawWord, CorrectedWord } from "@/types/api";

const PIPELINE_STAGES: { key: ProcessingStage; label: string }[] = [
  { key: "preprocessing", label: "Preprocessing" },
  { key: "scribe", label: "Scribe v2" },
  { key: "uncertainty", label: "Uncertainty" },
  { key: "tavily", label: "Tavily Verify" },
  { key: "claude", label: "Correcting" },
];

const SCENARIOS = [
  { id: "med-refill", label: "Medication Refill", description: "Patient calling to refill metformin and lisinopril prescriptions", icon: Pill, category: "Standard" },
  { id: "post-op", label: "Post-Op Follow-up", description: "Surgeon reviewing recovery progress after knee replacement", icon: Syringe, category: "Standard" },
  { id: "symptom-check", label: "New Symptom Report", description: "Patient describing new headaches and dizziness symptoms", icon: Stethoscope, category: "Standard" },
  { id: "allergy-review", label: "Allergy Review", description: "Nurse confirming drug allergies before administering new prescription", icon: Stethoscope, category: "Standard" },
  { id: "adversarial-accent", label: "Heavy Accent + Noise", description: "Thick accent over speakerphone with background TV audio", icon: AlertTriangle, category: "Adversarial" },
  { id: "rapid-meds", label: "Rapid Med List", description: "Doctor rattling off 6 medications in under 15 seconds", icon: Activity, category: "Adversarial" },
];

const DemoPage = () => {
  const [result, setResult] = useState<TranscribeResponse | null>(null);
  const [stage, setStage] = useState<ProcessingStage>("idle");
  const [activeScenario, setActiveScenario] = useState<string | null>(null);

  const simulateProcess = useCallback((scenarioId: string) => {
    setActiveScenario(scenarioId);
    setResult(null);
    setStage("preprocessing");
    setTimeout(() => setStage("scribe"), 500);
    setTimeout(() => setStage("uncertainty"), 1200);
    setTimeout(() => setStage("tavily"), 1700);
    setTimeout(() => setStage("claude"), 2400);
    setTimeout(() => {
      setResult(MOCK_TRANSCRIBE);
      setStage("done");
    }, 3200);
  }, []);

  const isProcessing = stage !== "idle" && stage !== "done" && stage !== "error";

  return (
    <div className="min-h-screen bg-secondary">
      <Navbar />

      <div className="bg-primary text-primary-foreground pt-20">
        <div className="container mx-auto px-6 max-w-[1400px] py-4">
          <h1 className="text-lg font-semibold text-primary-foreground">Interactive Demo</h1>
          <p className="text-sm text-primary-foreground/60 mt-1">Select a clinical scenario to see the pipeline in action</p>
        </div>
      </div>

      <div className="container mx-auto px-6 max-w-[1400px] py-8">
        {/* Scenario Buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 mb-8">
          {SCENARIOS.map((s) => {
            const Icon = s.icon;
            const isActive = activeScenario === s.id;
            return (
              <button
                key={s.id}
                onClick={() => !isProcessing && simulateProcess(s.id)}
                disabled={isProcessing}
                className={`text-left rounded-lg border p-4 transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
                  isActive
                    ? "border-accent bg-accent/5 shadow-card"
                    : "border-border bg-card hover:border-accent/50 hover:shadow-card"
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Icon className={`h-4 w-4 shrink-0 ${isActive ? "text-accent" : "text-muted-foreground"}`} />
                  <span className="text-sm font-medium text-foreground">{s.label}</span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{s.description}</p>
                {s.category === "Adversarial" && (
                  <Badge className="mt-2 text-[10px] bg-signal-red/10 text-signal-red border-0">Adversarial</Badge>
                )}
              </button>
            );
          })}
        </div>

        {/* Pipeline Status Bar */}
        {isProcessing && (
          <div className="mb-8">
            <div className="bg-card rounded-lg shadow-card p-4">
              <div className="flex items-center gap-2">
                {PIPELINE_STAGES.map((ps, i) => {
                  const stageIndex = PIPELINE_STAGES.findIndex(s => s.key === stage);
                  const isActive = i === stageIndex;
                  const isDone = i < stageIndex;
                  return (
                    <div key={ps.key} className="flex items-center gap-2 flex-1">
                      <div className={`flex items-center gap-2 px-3 py-2 rounded-md text-xs font-medium transition-all w-full ${
                        isActive ? "bg-accent/15 text-accent" : isDone ? "bg-success/10 text-success" : "bg-muted text-muted-foreground"
                      }`}>
                        {isActive && <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin shrink-0" />}
                        {isDone && <CheckCircle className="h-3 w-3 shrink-0" />}
                        <span className="truncate">{ps.label}</span>
                      </div>
                      {i < PIPELINE_STAGES.length - 1 && <span className="text-muted-foreground text-xs">→</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Three Panel Output */}
        {(isProcessing || stage === "done") && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <TranscriptPanel
              title="Raw Transcript"
              loading={stage !== "done"}
              words={result?.raw_transcript}
            />
            <CorrectedPanel
              title="Corrected Transcript"
              loading={stage !== "done"}
              words={result?.corrected_transcript}
              latency={result?.pipeline_latency_ms}
            />
            <SummaryPanel
              loading={stage !== "done"}
              summary={result?.clinical_summary}
            />
          </div>
        )}

        {/* Empty state */}
        {stage === "idle" && (
          <div className="text-center py-16 text-muted-foreground">
            <Stethoscope className="h-10 w-10 mx-auto mb-3 opacity-40" />
            <p className="text-sm">Pick a scenario above to run the pipeline</p>
          </div>
        )}
      </div>

      <Footer />
    </div>
  );
};

/* Sub-components */

const TranscriptPanel = ({ title, loading, words }: { title: string; loading: boolean; words?: RawWord[] }) => (
  <div className="bg-card rounded-lg shadow-card overflow-hidden">
    <div className="bg-primary px-4 py-3 flex items-center justify-between">
      <h3 className="text-sm font-bold text-primary-foreground">{title}</h3>
      {words && <Badge variant="secondary" className="text-xs">{words.length} words</Badge>}
    </div>
    <div className="p-4 max-h-[400px] overflow-auto">
      {loading ? (
        <div className="space-y-3">
          {[...Array(6)].map((_, i) => <div key={i} className="skeleton h-4 w-full" />)}
        </div>
      ) : words ? (
        <div className="leading-relaxed">
          {words.map((w, i) => {
            const isSpeakerLabel = w.word === "Doctor:" || w.word === "Patient:";
            if (isSpeakerLabel) {
              return (
                <span key={i}>
                  {i > 0 && <br />}
                  <Badge className={`mr-2 mt-2 text-xs ${w.speaker === "Doctor" ? "bg-primary/20 text-primary" : "bg-success/20 text-success"}`}>
                    {w.speaker === "Doctor" ? "Dr." : "Patient"}
                  </Badge>
                </span>
              );
            }

            let wordCls = "text-sm text-foreground";
            if (w.confidence === "LOW") wordCls = "text-sm bg-signal-red/20 text-signal-red rounded px-0.5 font-medium";
            else if (w.confidence === "MEDIUM") wordCls = "text-sm bg-warning/20 text-warning rounded px-0.5";

            return (
              <Tooltip key={i}>
                <TooltipTrigger asChild>
                  <span className={`${wordCls} cursor-default`}>
                    {w.word}{" "}
                  </span>
                </TooltipTrigger>
                {w.confidence !== "HIGH" && w.uncertainty_signals && (
                  <TooltipContent>
                    <p className="text-xs font-semibold mb-1">{w.confidence} confidence</p>
                    {w.uncertainty_signals.map((s, si) => (
                      <p key={si} className="text-xs text-muted-foreground">• {s}</p>
                    ))}
                  </TooltipContent>
                )}
              </Tooltip>
            );
          })}
        </div>
      ) : null}
    </div>
  </div>
);

const CorrectedPanel = ({ title, loading, words, latency }: {
  title: string; loading: boolean; words?: CorrectedWord[];
  latency?: TranscribeResponse["pipeline_latency_ms"];
}) => (
  <div className="bg-card rounded-lg shadow-card overflow-hidden">
    <div className="bg-primary px-4 py-3">
      <h3 className="text-sm font-bold text-primary-foreground">{title}</h3>
      {latency && (
        <p className="text-xs text-primary-foreground/60 mt-1">
          Scribe {latency.scribe}ms + Tavily {latency.tavily}ms + Claude {latency.claude}ms = {latency.total}ms
        </p>
      )}
    </div>
    <div className="p-4 max-h-[400px] overflow-auto">
      {loading ? (
        <div className="space-y-3">
          {[...Array(6)].map((_, i) => <div key={i} className="skeleton h-4 w-full" />)}
        </div>
      ) : words ? (
        <div className="leading-relaxed">
          {words.map((w, i) => {
            const isSpeakerLabel = w.word === "Doctor:" || w.word === "Patient:";
            if (isSpeakerLabel) {
              return (
                <span key={i}>
                  {i > 0 && <br />}
                  <Badge className={`mr-2 mt-2 text-xs ${w.speaker === "Doctor" ? "bg-primary/20 text-primary" : "bg-success/20 text-success"}`}>
                    {w.speaker === "Doctor" ? "Dr." : "Patient"}
                  </Badge>
                </span>
              );
            }
            let cls = "text-sm text-foreground";
            if (w.changed && w.tavily_verified) cls = "text-sm bg-accent/15 text-accent underline decoration-accent rounded px-0.5 font-medium";
            else if (w.changed) cls = "text-sm bg-warning/15 text-warning rounded px-0.5";
            else if (w.unverified) cls = "text-sm text-muted-foreground border-b border-dashed border-muted-foreground";

            return (
              <span key={i} className={cls}>
                {w.word}
                {w.unverified && <HelpCircle className="inline h-3 w-3 ml-0.5 text-muted-foreground" />}
                {" "}
              </span>
            );
          })}
        </div>
      ) : null}
    </div>
  </div>
);

const SummaryPanel = ({ loading, summary }: { loading: boolean; summary?: TranscribeResponse["clinical_summary"] }) => (
  <div className="bg-card rounded-lg shadow-card overflow-hidden">
    <div className="bg-primary px-4 py-3 flex items-center justify-between">
      <h3 className="text-sm font-bold text-primary-foreground">Clinical Summary</h3>
      {summary && <Badge className="bg-success text-success-foreground text-xs rounded-pill">EHR-Ready</Badge>}
    </div>
    <div className="p-4 max-h-[400px] overflow-auto">
      {loading ? (
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => <div key={i} className="skeleton h-4 w-full" />)}
        </div>
      ) : summary ? (
        <div className="space-y-4">
          {summary.appointment_needed && (
            <div className="bg-warning/10 border border-warning/30 rounded-lg p-3 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-warning shrink-0" />
              <p className="text-sm font-medium text-warning">Follow-up appointment recommended</p>
            </div>
          )}

          <div>
            <h4 className="text-xs font-bold uppercase text-muted-foreground mb-2">Medications</h4>
            <div className="space-y-2">
              {summary.medications.map((med, i) => (
                <div key={i} className="bg-secondary rounded-lg p-3">
                  <div className="flex items-center justify-between">
                    <p className="font-semibold text-sm text-foreground">{med.name}</p>
                    {med.tavily_verified
                      ? <CheckCircle className="h-4 w-4 text-success" />
                      : <HelpCircle className="h-4 w-4 text-muted-foreground" />}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {med.dosage} · {med.frequency} · {med.route}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {summary.symptoms.length > 0 && (
            <div>
              <h4 className="text-xs font-bold uppercase text-muted-foreground mb-2">Symptoms</h4>
              <div className="flex flex-wrap gap-2">
                {summary.symptoms.map((s, i) => (
                  <Badge key={i} variant="secondary" className="rounded-pill">{s}</Badge>
                ))}
              </div>
            </div>
          )}

          {summary.follow_up_actions.length > 0 && (
            <div>
              <h4 className="text-xs font-bold uppercase text-muted-foreground mb-2">Follow-up Actions</h4>
              <ul className="space-y-1">
                {summary.follow_up_actions.map((a, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-foreground">
                    <input type="checkbox" className="mt-1 rounded border-border" readOnly />
                    {a}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <Button variant="outline" className="w-full rounded-pill text-sm gap-2"
            onClick={() => {
              const blob = new Blob([JSON.stringify(summary, null, 2)], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url; a.download = "clinical_summary.json"; a.click();
              URL.revokeObjectURL(url);
            }}>
            <Download className="h-4 w-4" /> Export JSON
          </Button>
        </div>
      ) : null}
    </div>
  </div>
);

export default DemoPage;
