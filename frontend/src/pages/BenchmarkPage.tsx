import { useState, useMemo, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, ArrowUpDown, Info } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, LineChart, Line } from "recharts";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FadeInSection from "@/components/FadeInSection";
import { MOCK_BENCHMARK } from "@/services/mockData";
import { fetchBenchmark } from "@/services/api";
import type { BenchmarkClipResult, BenchmarkResponse } from "@/types/api";

type SortKey = keyof BenchmarkClipResult;
type Filter = "all" | "Standard" | "Adversarial";

// Simulated learning loop delta data
const learningLoopData = Array.from({ length: 20 }, (_, i) => ({
  call: i + 1,
  wer: +(21.4 - (i * 0.4) + (Math.random() * 0.5 - 0.25)).toFixed(1),
}));

const BenchmarkPage = () => {
  const [data, setData] = useState<BenchmarkResponse>(MOCK_BENCHMARK);
  const [dataSource, setDataSource] = useState<"api" | "mock">("mock");
  const [benchmarkNote, setBenchmarkNote] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("clip_id");
  const [sortAsc, setSortAsc] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const clips =
      filter === "all" ? ("all" as const) : filter === "Adversarial" ? ("adversarial" as const) : ("standard" as const);
    fetchBenchmark(clips)
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setDataSource("api");
          setBenchmarkNote(null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setData(MOCK_BENCHMARK);
          setDataSource("mock");
          setBenchmarkNote(
            "Live benchmark data is unavailable (backend off or results not generated). Showing embedded sample data.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [filter]);

  const filtered = useMemo(() => {
    let rows = filter === "all" ? data.results : data.results.filter(r => r.difficulty === filter);
    rows = [...rows].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") return sortAsc ? av - bv : bv - av;
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    return rows;
  }, [data.results, filter, sortKey, sortAsc]);

  const chartData = useMemo(() => {
    const cats: Record<string, number[]> = {};
    data.results.forEach(r => {
      if (!cats[r.category]) cats[r.category] = [];
      cats[r.category].push(r.improvement_pct);
    });
    return Object.entries(cats).map(([cat, vals]) => ({
      category: cat,
      improvement: +(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1),
    }));
  }, [data.results]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(true); }
  };

  const maxWer = data.ablation[0].wer;

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      {/* Header */}
      <section className="pt-28 pb-10">
        <div className="container mx-auto px-6 max-w-[900px] space-y-3">
          {benchmarkNote && (
            <div className="rounded-lg border border-warning/40 bg-warning/10 px-4 py-2 text-sm text-warning">
              {benchmarkNote}
            </div>
          )}
          {dataSource === "api" && !benchmarkNote && (
            <div className="rounded-lg border border-success/30 bg-success/10 px-4 py-2 text-xs text-success">
              Showing benchmark results from the API ({import.meta.env.VITE_API_URL || "VITE_API_URL"})
            </div>
          )}
          <h1 className="text-2xl font-bold text-foreground">Benchmark Results</h1>
          <p className="text-sm text-muted-foreground mt-2">
            {data.aggregate.avg_improvement_pct}% fewer medical term errors across 20 adversarial clips.
            Verification rate {data.metrics.verification_rate}%, unsafe guess rate {data.metrics.unsafe_guess_rate}%.
          </p>
        </div>
      </section>

      <div className="container mx-auto px-6 max-w-[900px] pb-20">

        {/* Ablation Study Table */}
        <FadeInSection>
          <h2 className="text-2xl font-bold mb-6">Ablation Study</h2>
          <div className="overflow-x-auto rounded-lg border shadow-card mb-12">
            <table className="w-full text-sm">
              <thead className="bg-primary text-primary-foreground">
                <tr>
                  <th className="px-4 py-3 text-left">Pipeline Stage</th>
                  <th className="px-4 py-3 text-left w-32">WER</th>
                  <th className="px-4 py-3 text-left w-20">Delta</th>
                  <th className="px-4 py-3 text-left">What This Proves</th>
                </tr>
              </thead>
              <tbody>
                {data.ablation.map((row, i) => (
                  <tr key={i} className={`border-t ${i % 2 === 0 ? "" : "bg-secondary/50"}`}>
                    <td className="px-4 py-3 font-medium text-foreground">{row.stage}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-foreground">{row.wer}%</span>
                        <div className="flex-1 bg-muted rounded-full h-2 max-w-[80px]">
                          <div className="bg-accent rounded-full h-2 transition-all" style={{ width: `${(row.wer / maxWer) * 100}%` }} />
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      {row.delta > 0 ? (
                        <span className="text-success font-semibold">-{row.delta}%</span>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground text-xs">{row.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </FadeInSection>

        {/* Key Metrics */}
        <div className="mb-12">
          <h2 className="text-lg font-semibold mb-4">Key Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: "Verification Rate", value: `${data.metrics.verification_rate}%`, tip: "Corrections backed by Tavily confirmation" },
              { label: "Unsafe Guess Rate", value: `${data.metrics.unsafe_guess_rate}%`, tip: "Corrections made without verification" },
              { label: "Uncertainty Coverage", value: `${data.metrics.uncertainty_coverage}%`, tip: "Low-confidence words correctly flagged" },
              { label: "Phonetic Hit Rate", value: `${data.metrics.phonetic_hit_rate}%`, tip: "Drug name misspellings caught" },
            ].map((m) => (
              <Tooltip key={m.label}>
                <TooltipTrigger asChild>
                  <div className="border rounded-md p-3 text-center cursor-default">
                    <p className="text-lg font-semibold text-foreground">{m.value}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{m.label}</p>
                  </div>
                </TooltipTrigger>
                <TooltipContent><p className="text-xs max-w-[200px]">{m.tip}</p></TooltipContent>
              </Tooltip>
            ))}
          </div>
        </div>

        {/* Per-clip results */}
        <FadeInSection>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">Per-Clip Results</h2>
            <div className="flex gap-2">
              {(["all", "Standard", "Adversarial"] as Filter[]).map(f => (
                <Button key={f} variant={filter === f ? "default" : "outline"} size="sm"
                  className={filter === f ? "bg-accent text-accent-foreground" : ""}
                  onClick={() => setFilter(f)}>
                  {f === "all" ? "All" : f}
                </Button>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto rounded-lg border shadow-card mb-12">
            <table className="w-full text-sm">
              <thead className="bg-primary text-primary-foreground">
                <tr>
                  {[
                    { key: "clip_id" as SortKey, label: "Clip" },
                    { key: "category" as SortKey, label: "Category" },
                    { key: "difficulty" as SortKey, label: "Difficulty" },
                    { key: "raw_wer" as SortKey, label: "Raw WER" },
                    { key: "corrected_wer" as SortKey, label: "Corrected WER" },
                    { key: "improvement_pct" as SortKey, label: "Delta %" },
                  ].map(col => (
                    <th key={col.key} className="px-4 py-3 text-left cursor-pointer hover:bg-primary-foreground/10 whitespace-nowrap"
                      onClick={() => toggleSort(col.key)}>
                      <span className="flex items-center gap-1">
                        {col.label}
                        <ArrowUpDown className="h-3 w-3" />
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={r.clip_id} className={`border-t hover:bg-accent/5 transition-colors ${i % 2 === 0 ? "" : "bg-secondary/50"}`}>
                    <td className="px-4 py-3 font-medium">{r.clip_id}</td>
                    <td className="px-4 py-3">{r.category}</td>
                    <td className="px-4 py-3">
                      <Badge className={r.difficulty === "Adversarial" ? "bg-signal-red text-accent-foreground" : "bg-muted text-muted-foreground"}>
                        {r.difficulty}
                      </Badge>
                    </td>
                    <td className="px-4 py-3">{r.raw_wer}%</td>
                    <td className="px-4 py-3">{r.corrected_wer}%</td>
                    <td className="px-4 py-3">
                      <span className={r.improvement_pct > 0 ? "text-success font-semibold" : "text-signal-red font-semibold"}>
                        {r.improvement_pct > 0 ? "+" : ""}{r.improvement_pct}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </FadeInSection>

        {/* Category Chart */}
        <FadeInSection>
          <h3 className="text-lg font-bold mb-4">Average Improvement by Category</h3>
          <div className="h-64 mb-12">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical">
                <XAxis type="number" domain={[0, 50]} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="category" width={140} tick={{ fontSize: 12 }} />
                <RechartsTooltip formatter={(v: number) => `${v}%`} />
                <Bar dataKey="improvement" radius={[0, 4, 4, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill="hsl(180 100% 33%)" />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </FadeInSection>

        {/* Learning Loop Delta */}
        <FadeInSection>
          <h3 className="text-lg font-bold mb-2">Learning Loop Delta</h3>
          <p className="text-sm text-muted-foreground mb-4">WER improvement as the adaptive vocabulary grows across 20 calls</p>
          <div className="h-48 mb-12">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={learningLoopData}>
                <XAxis dataKey="call" label={{ value: "Call #", position: "insideBottom", offset: -5, fontSize: 12 }} />
                <YAxis domain={[10, 25]} tickFormatter={v => `${v}%`} />
                <RechartsTooltip formatter={(v: number) => `${v}%`} labelFormatter={l => `Call ${l}`} />
                <Line type="monotone" dataKey="wer" stroke="hsl(180 100% 33%)" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </FadeInSection>

        {/* Methodology */}
        <FadeInSection>
          <div className="mb-12">
            <h2 className="text-2xl font-bold mb-6">Methodology</h2>
            <ol className="space-y-3 text-sm text-foreground">
              {[
                "ElevenLabs TTS renders 20 healthcare call scripts with diverse voices and accents",
                "ffmpeg degrades audio to simulate 8kHz telephony conditions (loudnorm → afftdn → resample)",
                "Scribe v2 transcribes each clip with dynamic keyterm prompting (up to 100 terms)",
                "jiwer computes raw WER against known ground truth scripts",
                "Multi-signal uncertainty detection scores each word (timing + phonetic + keyterm + history)",
                "Tavily verification fires on LOW-confidence medical-pattern words (capped at 5/transcript)",
                "Claude corrects only Tavily-confirmed terms; flags all else as [UNVERIFIED]",
                "jiwer computes corrected WER — improvement measured per stage (ablation)",
              ].map((step, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-accent text-accent-foreground text-xs flex items-center justify-center font-bold">{i + 1}</span>
                  {step}
                </li>
              ))}
            </ol>
          </div>
        </FadeInSection>

        {/* Download */}
        <Button variant="outline" className="rounded-pill gap-2"
          onClick={() => {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url; a.download = "benchmark_results.json"; a.click();
            URL.revokeObjectURL(url);
          }}>
          <Download className="h-4 w-4" /> Download Full Results JSON
        </Button>
      </div>

      <Footer />
    </div>
  );
};

export default BenchmarkPage;
