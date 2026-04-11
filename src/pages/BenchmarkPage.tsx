import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, ArrowUpDown } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FadeInSection from "@/components/FadeInSection";
import { MOCK_BENCHMARK } from "@/services/mockData";
import type { BenchmarkClipResult } from "@/types/api";

type SortKey = keyof BenchmarkClipResult;
type Filter = "all" | "Standard" | "Adversarial";

const BenchmarkPage = () => {
  const data = MOCK_BENCHMARK;
  const [filter, setFilter] = useState<Filter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("clip_id");
  const [sortAsc, setSortAsc] = useState(true);

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

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      {/* Hero stat */}
      <section className="pt-28 pb-16 text-center">
        <FadeInSection>
          <p className="text-8xl font-extrabold text-accent">{data.aggregate.avg_improvement_pct}%</p>
          <p className="text-xl text-foreground mt-4 font-semibold">Fewer medical term errors across 20 adversarial clips</p>
          <p className="text-sm text-muted-foreground mt-2">
            Raw Scribe v2 WER: {data.aggregate.avg_raw_wer}% → Corrected WER: {data.aggregate.avg_corrected_wer}%
          </p>
        </FadeInSection>
      </section>

      <div className="container mx-auto px-6 max-w-[900px] pb-20">
        {/* Methodology */}
        <FadeInSection>
          <div className="mb-12">
            <h2 className="text-2xl font-bold mb-6">Methodology</h2>
            <ol className="space-y-3 text-sm text-foreground">
              {[
                "ElevenLabs TTS renders 20 healthcare call scripts with diverse voices",
                "ffmpeg degrades audio to simulate 8kHz telephony conditions",
                "Scribe v2 transcribes each clip with 100 medical keyterms loaded",
                "jiwer computes raw WER against known ground truth scripts",
                "Tavily + Claude correction pipeline applied to each transcript",
                "jiwer computes corrected WER — improvement measured",
              ].map((step, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-accent text-accent-foreground text-xs flex items-center justify-center font-bold">{i + 1}</span>
                  {step}
                </li>
              ))}
            </ol>
          </div>
        </FadeInSection>

        {/* Keyterm callout */}
        <FadeInSection>
          <div className="bg-accent/10 border border-accent/20 rounded-lg p-6 mb-12">
            <p className="text-accent font-bold text-lg">{data.aggregate.keyterm_impact_pct}%</p>
            <p className="text-sm text-foreground mt-1">
              Scribe v2 Keyterm Prompting reduced drug name errors before any Claude correction.
            </p>
          </div>
        </FadeInSection>

        {/* Filter */}
        <div className="flex gap-2 mb-4">
          {(["all", "Standard", "Adversarial"] as Filter[]).map(f => (
            <Button key={f} variant={filter === f ? "default" : "outline"} size="sm"
              className={filter === f ? "bg-accent text-accent-foreground" : ""}
              onClick={() => setFilter(f)}>
              {f === "all" ? "All" : f}
            </Button>
          ))}
        </div>

        {/* Table */}
        <FadeInSection>
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
                    { key: "improvement_pct" as SortKey, label: "Improvement" },
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

        {/* Chart */}
        <FadeInSection>
          <h3 className="text-lg font-bold mb-4">Average Improvement by Category</h3>
          <div className="h-64 mb-12">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical">
                <XAxis type="number" domain={[0, 50]} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="category" width={140} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v: number) => `${v}%`} />
                <Bar dataKey="improvement" radius={[0, 4, 4, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill="hsl(180 100% 33%)" />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
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
