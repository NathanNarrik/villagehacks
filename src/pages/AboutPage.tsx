import { Badge } from "@/components/ui/badge";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FadeInSection from "@/components/FadeInSection";

const team = [
  { name: "Person A", role: "STT + Data Engineer", color: "bg-primary", bullets: ["ElevenLabs Scribe v2 integration", "20-clip adversarial benchmark", "ffmpeg preprocessing pipeline"] },
  { name: "Person B", role: "AI / NLP Engineer", color: "bg-accent", bullets: ["Tavily knowledge grounding", "Claude correction prompts", "Entity extraction pipeline"] },
  { name: "Person C", role: "Backend Engineer", color: "bg-destructive", bullets: ["FastAPI endpoints & WebSocket", "Redis caching layer", "Deployment & infrastructure"] },
  { name: "Person D", role: "Frontend + Demo", color: "bg-warning", bullets: ["React UI & design system", "Three-panel demo interface", "Benchmark visualization"] },
];

const stack = [
  { name: "ElevenLabs Scribe v2", desc: "Medical STT with keyterm prompting + streaming", sponsor: true },
  { name: "Tavily", desc: "Live drug name verification & knowledge grounding", sponsor: true },
  { name: "Claude", desc: "Transcript correction & clinical entity extraction", sponsor: false },
  { name: "FastAPI", desc: "Async Python backend with WebSocket streaming", sponsor: false },
  { name: "React + Vite", desc: "Frontend SPA with Tailwind CSS", sponsor: false },
  { name: "Redis", desc: "Tavily cache + session state (1hr TTL)", sponsor: false },
  { name: "ffmpeg", desc: "Audio preprocessing: loudnorm → noise reduction → resample", sponsor: false },
  { name: "jiwer", desc: "Word Error Rate computation against ground truth", sponsor: false },
];

const AboutPage = () => (
  <div className="min-h-screen bg-background">
    <Navbar />

    <div className="container mx-auto px-6 max-w-[800px] pt-28 pb-20">
      {/* Vision */}
      <FadeInSection>
        <blockquote className="text-2xl md:text-3xl font-bold italic text-accent leading-snug mb-8">
          "No patient should feel unsupported between appointments."
        </blockquote>
        <p className="text-foreground leading-relaxed mb-4">
          Every day, thousands of healthcare calls are transcribed by generic STT systems that weren't built for telephony audio,
          medical terminology, or real-world noise. The result: wrong drug names, garbled dosages, missed symptoms.
        </p>
        <p className="text-foreground leading-relaxed mb-12">
          CareCaller AI is a four-layer pipeline that combines ElevenLabs Scribe v2 for medical-grade transcription,
          Tavily for live drug verification, and Claude for intelligent correction — producing clinically accurate,
          EHR-ready summaries from any phone call.
        </p>
      </FadeInSection>

      {/* Team */}
      <FadeInSection>
        <h2 className="text-2xl font-bold mb-6">The Team</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
          {team.map(t => (
            <div key={t.name} className="bg-secondary rounded-lg p-5 hover:shadow-card transition-shadow">
              <div className="flex items-center gap-3 mb-3">
                <h3 className="font-bold text-foreground">{t.name}</h3>
                <Badge className={`${t.color} text-accent-foreground text-xs rounded-pill`}>{t.role}</Badge>
              </div>
              <ul className="space-y-1">
                {t.bullets.map((b, i) => (
                  <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                    <span className="text-accent mt-0.5">•</span> {b}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </FadeInSection>

      {/* Tech Stack */}
      <FadeInSection>
        <h2 className="text-2xl font-bold mb-6">Tech Stack</h2>
        <div className="rounded-lg border overflow-hidden shadow-card mb-12">
          {stack.map((s, i) => (
            <div key={s.name} className={`flex items-center justify-between px-5 py-3 ${i % 2 === 0 ? "bg-background" : "bg-secondary/50"}`}>
              <div className="flex items-center gap-3">
                <span className="font-semibold text-sm text-foreground">{s.name}</span>
                {s.sponsor && (
                  <Badge className="bg-warning/20 text-warning text-xs rounded-pill border border-warning/30 hover:shadow-[0_0_8px_hsl(var(--warning)/0.3)]">
                    Sponsor
                  </Badge>
                )}
              </div>
              <span className="text-sm text-muted-foreground">{s.desc}</span>
            </div>
          ))}
        </div>
      </FadeInSection>

      {/* Built at banner */}
      <div className="bg-primary text-primary-foreground rounded-lg p-6 text-center">
        <p className="font-bold">Built in 12 hours at Hackathon · April 2026</p>
      </div>
    </div>

    <Footer />
  </div>
);

export default AboutPage;
