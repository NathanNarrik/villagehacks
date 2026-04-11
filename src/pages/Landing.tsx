import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Mic, Upload, ArrowRight, Activity } from "lucide-react";
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
            <p className="text-accent font-semibold text-sm tracking-widest uppercase mb-4">Healthcare Speech Intelligence</p>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold leading-tight max-w-3xl text-primary-foreground">
              No patient should feel unsupported between appointments.
            </h1>
            <p className="text-lg md:text-xl text-primary-foreground/80 mt-6 max-w-2xl">
              AI-powered speech intelligence for healthcare calls. Accurate transcription, verified medications, clinical summaries — in seconds.
            </p>
            <div className="flex flex-wrap gap-4 mt-8">
              <Button asChild className="bg-accent text-accent-foreground hover:bg-accent/90 rounded-pill px-8 py-3 text-base font-semibold">
                <Link to="/demo">Try the Demo <ArrowRight className="ml-2 h-4 w-4" /></Link>
              </Button>
              <Button asChild variant="outline" className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10 rounded-pill px-8 py-3 text-base">
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
              { num: "8kHz", label: "Phone audio strips clarity" },
              { num: "37%", label: "Fewer medical term errors" },
              { num: "150ms", label: "Real-time transcription" },
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
            <h2 className="text-3xl font-bold text-center mb-12">Where Standard STT Fails</h2>
          </FadeInSection>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { icon: <Activity className="h-8 w-8 text-signal-red" />, title: "Telephony Audio", desc: "8kHz bandwidth strips high-frequency detail. Models trained on studio audio degrade sharply on real phone lines." },
              { icon: <AlertTriangle className="h-8 w-8 text-signal-red" />, title: "Medical Terminology", desc: "Drug names, dosages, symptoms — none appear in standard training data. Generic models guess them wrong." },
              { icon: <Mic className="h-8 w-8 text-signal-red" />, title: "Real-World Noise", desc: "Background noise, speakerphones, poor connections. Patients don't call from quiet rooms." },
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

      {/* How It Works */}
      <section className="py-20 bg-secondary">
        <div className="container mx-auto px-6 max-w-[1100px]">
          <FadeInSection>
            <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          </FadeInSection>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { step: "1", icon: <Upload className="h-10 w-10 text-accent" />, title: "Upload or Record", desc: "Upload a call recording or use live mic streaming." },
              { step: "2", icon: <Activity className="h-10 w-10 text-accent" />, title: "ElevenLabs + Tavily Analyze", desc: "Scribe v2 transcribes, Tavily verifies medical terms, Claude corrects." },
              { step: "3", icon: <ArrowRight className="h-10 w-10 text-accent" />, title: "Clinical Summary", desc: "Get structured medications, symptoms, and follow-up actions — EHR-ready." },
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
