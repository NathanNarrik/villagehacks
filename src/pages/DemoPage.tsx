import { useState, useCallback, useRef } from "react";
import { Upload, Mic, Square, FileAudio, CheckCircle, AlertTriangle, HelpCircle, Download, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { MOCK_TRANSCRIBE, DEMO_CLIPS } from "@/services/mockData";
import type { TranscribeResponse, ProcessingStage, RawWord, CorrectedWord } from "@/types/api";

const DemoPage = () => {
  const [result, setResult] = useState<TranscribeResponse | null>(null);
  const [stage, setStage] = useState<ProcessingStage>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordTime, setRecordTime] = useState(0);
  const [showClipsModal, setShowClipsModal] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval>>();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const simulateProcess = useCallback(() => {
    setStage("scribe");
    setTimeout(() => setStage("tavily"), 800);
    setTimeout(() => setStage("claude"), 1600);
    setTimeout(() => {
      setResult(MOCK_TRANSCRIBE);
      setStage("done");
    }, 2400);
  }, []);

  const handleFileSelect = (file: File) => {
    const validTypes = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/x-m4a", "audio/mp3"];
    if (!validTypes.some(t => file.type.includes(t.split("/")[1]))) {
      alert("Please upload WAV, MP3, or M4A files only.");
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setStage("idle");
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
  };

  const handleRecord = () => {
    if (isRecording) {
      setIsRecording(false);
      clearInterval(timerRef.current);
      simulateProcess();
    } else {
      setIsRecording(true);
      setRecordTime(0);
      setResult(null);
      setStage("idle");
      timerRef.current = setInterval(() => setRecordTime(t => t + 1), 1000);
    }
  };

  const handleClipSelect = (clipId: string) => {
    setShowClipsModal(false);
    setSelectedFile(null);
    setStage("idle");
    simulateProcess();
  };

  const stageText: Record<ProcessingStage, string> = {
    idle: "",
    uploading: "Uploading audio...",
    scribe: "Running Scribe v2...",
    tavily: "Verifying with Tavily...",
    claude: "Extracting clinical data...",
    done: "Complete",
    error: "Pipeline failed",
  };

  return (
    <div className="min-h-screen bg-secondary">
      <Navbar />

      {/* Header bar */}
      <div className="bg-primary text-primary-foreground pt-20">
        <div className="container mx-auto px-6 max-w-[1400px] py-4 flex flex-wrap items-center justify-between gap-4">
          <h1 className="text-xl font-bold text-primary-foreground">CareCaller AI Demo</h1>
          <Badge className="bg-accent text-accent-foreground rounded-pill px-4 py-1 text-sm font-semibold">
            37% fewer medical term errors
          </Badge>
          <Button variant="outline" className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10 rounded-pill text-sm"
            onClick={() => setShowClipsModal(true)}>
            Demo Clips
          </Button>
        </div>
      </div>

      <div className="container mx-auto px-6 max-w-[1400px] py-8">
        {/* Input Area */}
        <Tabs defaultValue="upload" className="mb-8">
          <TabsList className="bg-card shadow-card">
            <TabsTrigger value="upload" className="gap-2"><Upload className="h-4 w-4" /> Upload File</TabsTrigger>
            <TabsTrigger value="record" className="gap-2"><Mic className="h-4 w-4" /> Record Live</TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer ${
                dragOver ? "border-accent bg-accent/5" : "border-border bg-card"
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input ref={fileInputRef} type="file" accept="audio/*" className="hidden"
                onChange={e => e.target.files?.[0] && handleFileSelect(e.target.files[0])} />
              <FileAudio className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              {selectedFile ? (
                <p className="text-foreground font-medium">{selectedFile.name}</p>
              ) : (
                <p className="text-muted-foreground">Drop a call recording here or click to browse<br />
                  <span className="text-sm">WAV, MP3, M4A</span>
                </p>
              )}
            </div>
            {selectedFile && (
              <Button className="mt-4 bg-accent text-accent-foreground hover:bg-accent/90 rounded-pill px-8"
                onClick={simulateProcess} disabled={stage !== "idle" && stage !== "done"}>
                Process Call
              </Button>
            )}
          </TabsContent>

          <TabsContent value="record">
            <div className="bg-card rounded-lg p-12 text-center shadow-card">
              <div className="relative inline-block">
                {isRecording && (
                  <div className="absolute inset-0 rounded-full bg-signal-red/30 animate-pulse-ring" />
                )}
                <button
                  onClick={handleRecord}
                  className={`relative w-20 h-20 rounded-full flex items-center justify-center transition-colors ${
                    isRecording ? "bg-signal-red" : "bg-accent hover:bg-accent/90"
                  }`}
                >
                  {isRecording ? <Square className="h-8 w-8 text-accent-foreground" /> : <Mic className="h-8 w-8 text-accent-foreground" />}
                </button>
              </div>
              <p className="mt-4 text-muted-foreground">
                {isRecording
                  ? `Recording... ${Math.floor(recordTime / 60)}:${(recordTime % 60).toString().padStart(2, "0")}`
                  : "Click to start recording"}
              </p>
              {isRecording && (
                <Button className="mt-4 bg-signal-red text-accent-foreground hover:bg-signal-red/90 rounded-pill"
                  onClick={handleRecord}>
                  Stop & Process
                </Button>
              )}
            </div>
          </TabsContent>
        </Tabs>

        {/* Processing status */}
        {stage !== "idle" && stage !== "done" && (
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-3 bg-card rounded-pill px-6 py-3 shadow-card">
              <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span className="text-sm font-medium text-foreground">{stageText[stage]}</span>
            </div>
          </div>
        )}

        {/* Three Panel Output */}
        {(stage === "done" || (stage !== "idle" && stage !== "done")) && (
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
      </div>

      {/* Demo Clips Modal */}
      {showClipsModal && (
        <div className="fixed inset-0 z-50 bg-foreground/50 flex items-center justify-center p-4"
          onClick={() => setShowClipsModal(false)}>
          <div className="bg-background rounded-lg max-w-4xl w-full max-h-[80vh] overflow-auto p-6 shadow-hover"
            onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-foreground">Demo Clips</h2>
              <button onClick={() => setShowClipsModal(false)} className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {DEMO_CLIPS.map(clip => (
                <button key={clip.id}
                  className="text-left bg-card rounded-lg p-4 border hover:border-accent hover:shadow-card transition-all"
                  onClick={() => handleClipSelect(clip.id)}>
                  <p className="font-medium text-sm text-foreground">{clip.name}</p>
                  <div className="flex gap-2 mt-2">
                    <Badge variant="secondary" className="text-xs">{clip.category}</Badge>
                    <Badge className={`text-xs ${clip.difficulty === "Adversarial" ? "bg-signal-red text-accent-foreground" : "bg-muted text-muted-foreground"}`}>
                      {clip.difficulty}
                    </Badge>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

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
            return (
              <span key={i}
                className={`text-sm ${w.uncertain ? "bg-signal-red/20 text-signal-red rounded px-0.5" : "text-foreground"}`}
                title={`${w.start_ms}ms - ${w.end_ms}ms`}
              >
                {w.word}{" "}
              </span>
            );
          })}
        </div>
      ) : null}
    </div>
  </div>
);

const CorrectedPanel = ({ title, loading, words, latency }: {
  title: string; loading: boolean; words?: CorrectedWord[];
  latency?: { scribe: number; tavily: number; claude: number; total: number };
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
