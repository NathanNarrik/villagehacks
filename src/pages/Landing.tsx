import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Mic, Upload, ArrowRight, Activity, Shield, Search } from "lucide-react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FadeInSection from "@/components/FadeInSection";

const Landing = () => {
  return (
    <div className="min-h-screen">
      <Navbar />

      {/* Hero */}
      <section className="relative bg-primary text-primary-foreground pt-32 pb-20 overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{
          backgroundImage: "radial-gradient(circle at 1px 1px, currentColor 1px, transparent 0)",
          backgroundSize: "40px 40px",
        }} />
        <div className="container mx-auto px-6 max-w-[1100px] relative">
          <FadeInSection>
            <p className="text-accent font-semibold text-sm tracking-widest uppercase mb-4">Verification-Augmented Speech-to-Text</p>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold leading-tight max-w-3xl text-primary-foreground">
              Speech-to-text that knows when it might be wrong.
            </h1>
            <p className="text-lg md:text-xl text-primary-foreground/80 mt-6 max-w-2xl">
              Confidence-gated, Tavily-verified, zero hallucinated corrections.
              A reliability layer for STT in noisy, high-stakes environments.
            </p>
            <div className="flex flex-wrap gap-4 mt-8">
              <Button asChild className="bg-accent text-accent-foreground hover:bg-accent/90 rounded-pill px-8 py-3 text-base font-semibold">
                <Link to="/demo">Try the Demo <ArrowRight className="ml-2 h-4 w-4" /></Link>
              </Button>
              <Button asChild className="border border-primary-foreground/30 bg-transparent text-primary-foreground hover:bg-primary-foreground/10 rounded-pill px-8 py-3 text-base">
                <Link to="/benchmark">See Benchmark Results</Link>
              </Button>
            </div>
          </FadeInSection>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="py-12 bg-background border-b">
        <div className="container mx-auto px-6 max-w-[1100px]">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            {[
              { num: "21.4%", label: "Raw WER on telephony audio" },
              { num: "37%", label: "Fewer medical term errors" },
              { num: "0%", label: "Hallucinated corrections" },
            ].map((stat) => (
              <FadeInSection key={stat.num}>
                <p className="text-4xl font-extrabold text-accent">{stat.num}</p>
                <p className="text-muted-foreground mt-1">{stat.label}</p>
              </FadeInSection>
            ))}
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section className="py-20 bg-background">
        <div className="container mx-auto px-6 max-w-[1100px]">
          <FadeInSection>
            <h2 className="text-3xl font-bold text-center mb-4">Where Standard STT Fails</h2>
            <p className="text-center text-muted-foreground max-w-2xl mx-auto mb-12">
              Generic STT produces confident-looking transcripts even when the audio is degraded.
              The core problem isn't that STT is wrong — it's that it doesn't know it's wrong.
            </p>
          </FadeInSection>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { icon: <Activity className="h-8 w-8 text-signal-red" />, title: "Telephony Audio", desc: "8kHz bandwidth strips high-frequency phonemes. Models trained on studio audio degrade sharply — methotrexate vs metformin becomes a coin flip." },
              { icon: <AlertTriangle className="h-8 w-8 text-signal-red" />, title: "Medical Terminology", desc: "Drug names, dosages, symptoms don't appear in standard training corpora. Generic models pattern-match to similar common words." },
              { icon: <Mic className="h-8 w-8 text-signal-red" />, title: "Silent Failure", desc: "The transcript looks fine. No confidence flag. No uncertainty marker. The clinician reads it and assumes it's correct." },
            ].map((card) => (
              <FadeInSection key={card.title}>
                <div className="bg-card rounded-lg p-6 shadow-card hover:shadow-hover hover:-translate-y-1 transition-all duration-200">
                  {card.icon}
                  <h3 className="text-lg font-bold mt-4 text-foreground">{card.title}</h3>
                  <p className="text-muted-foreground text-sm mt-2">{card.desc}</p>
                </div>
              </FadeInSection>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works — 3-step from doc */}
      <section className="py-20 bg-secondary">
        <div className="container mx-auto px-6 max-w-[1100px]">
          <FadeInSection>
            <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          </FadeInSection>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { step: "1", icon: <Upload className="h-10 w-10 text-accent" />, title: "ElevenLabs Scribe v2 Transcribes", desc: "Scribe v2 with dynamic keyterm prompting transcribes the call with word-level timestamps and speaker diarization." },
              { step: "2", icon: <Shield className="h-10 w-10 text-accent" />, title: "Multi-Signal Uncertainty Detection", desc: "Every word gets a composite confidence score from timing, phonetic distance, keyterm mismatch, and correction history." },
              { step: "3", icon: <Search className="h-10 w-10 text-accent" />, title: "Tavily Verifies — Correct or Flag", desc: "Low-confidence words are verified with Tavily. Confirmed terms are corrected; unconfirmed terms are flagged [UNVERIFIED]." },
            ].map((s) => (
              <FadeInSection key={s.step}>
                <div className="text-center group">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent/10 mb-4 group-hover:bg-accent/20 transition-colors">
                    {s.icon}
                  </div>
                  <div className="text-xs font-bold text-accent mb-2">STEP {s.step}</div>
                  <h3 className="text-lg font-bold text-foreground">{s.title}</h3>
                  <p className="text-muted-foreground text-sm mt-2">{s.desc}</p>
                </div>
              </FadeInSection>
            ))}
          </div>
        </div>
      </section>

      {/* Benchmark Callout */}
      <section className="py-20 bg-background">
        <div className="container mx-auto px-6 max-w-[1100px] text-center">
          <FadeInSection>
            <p className="text-7xl font-extrabold text-accent">37%</p>
            <p className="text-xl text-foreground mt-4 font-semibold">Fewer medical term errors across 20 adversarial clips</p>
            <p className="text-sm text-muted-foreground mt-3">
              Verification Rate: <span className="text-success font-semibold">100%</span> · Unsafe Guess Rate: <span className="text-accent font-semibold">0%</span>
            </p>
            <Button asChild variant="link" className="text-accent mt-4 text-base">
              <Link to="/benchmark">View Methodology →</Link>
            </Button>
          </FadeInSection>
        </div>
      </section>

      {/* Sponsor Bar */}
      <section className="py-12 bg-secondary border-t border-b">
        <div className="container mx-auto px-6 max-w-[1100px] text-center">
          <p className="text-sm text-muted-foreground uppercase tracking-widest mb-6">Powered By</p>
          <div className="flex items-center justify-center gap-12 text-xl font-bold text-foreground/60">
            <span>ElevenLabs</span>
            <span className="text-border">·</span>
            <span>Tavily</span>
            <span className="text-border">·</span>
            <span>Claude</span>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Landing;
