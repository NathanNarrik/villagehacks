import { Link } from "react-router-dom";

const Footer = () => (
  <footer className="bg-primary text-primary-foreground py-12">
    <div className="container mx-auto px-6 max-w-[1100px]">
      <div className="flex flex-col md:flex-row items-center justify-between gap-6">
        <div>
          <p className="text-lg font-bold">
            CareCaller <span className="text-accent">AI</span>
          </p>
          <p className="text-sm text-primary-foreground/70 mt-1">
            Healthcare Speech Intelligence Platform
          </p>
        </div>
        <div className="flex gap-6 text-sm text-primary-foreground/70">
          <Link to="/" className="hover:text-accent transition-colors">Home</Link>
          <Link to="/demo" className="hover:text-accent transition-colors">Demo</Link>
          <Link to="/benchmark" className="hover:text-accent transition-colors">Benchmark</Link>
          <Link to="/about" className="hover:text-accent transition-colors">About</Link>
        </div>
      </div>
      <div className="border-t border-primary-foreground/20 mt-8 pt-6 text-center text-xs text-primary-foreground/50">
        Built at Hackathon · April 2026 · Powered by ElevenLabs + Tavily + Claude
      </div>
    </div>
  </footer>
);

export default Footer;
