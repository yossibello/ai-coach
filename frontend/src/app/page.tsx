import Link from "next/link";
import { ArrowRight, Zap, TrendingUp, Brain, Upload, BarChart3, Shield } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-surface text-slate-200">
      {/* Nav */}
      <nav className="fixed top-0 inset-x-0 z-50 border-b border-surface-border/50 bg-surface/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-6 h-6 text-brand-500" />
            <span className="font-bold text-lg text-white">AI Coach</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/login" className="text-sm text-slate-400 hover:text-white transition-colors">
              Sign in
            </Link>
            <Link
              href="/signup"
              className="px-4 py-2 bg-brand-500 hover:bg-brand-600 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Start free
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-24 px-6 relative overflow-hidden">
        <div className="absolute inset-0 bg-hero-glow pointer-events-none" />
        <div className="max-w-5xl mx-auto text-center relative">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-brand-500/30 bg-brand-500/10 text-brand-400 text-xs font-medium mb-8">
            <Brain className="w-3.5 h-3.5" />
            Powered by a Cycling-Native Transformer Model
          </div>

          <h1 className="text-5xl md:text-7xl font-extrabold text-white leading-tight mb-6">
            Your AI coach knows
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-emerald-300">
              exactly how you train
            </span>
          </h1>

          <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Upload rides from Strava, Garmin, or GPX files. Our transformer model learns your unique
            physiological patterns — power, HRV, HR drift, environment — and builds the optimal plan
            to hit your goal, whether that&apos;s peak FTP or race day.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/signup"
              className="flex items-center gap-2 px-8 py-4 bg-brand-500 hover:bg-brand-600 text-white font-semibold rounded-xl transition-all hover:scale-105 shadow-lg shadow-brand-500/25"
            >
              Connect Strava & Start <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/login"
              className="px-8 py-4 border border-surface-border hover:border-slate-500 text-slate-300 font-semibold rounded-xl transition-colors"
            >
              Already have an account
            </Link>
          </div>
        </div>
      </section>

      {/* Feature grid */}
      <section className="py-20 px-6 border-t border-surface-border">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-white text-center mb-4">
            Every training variable. One brain.
          </h2>
          <p className="text-slate-400 text-center mb-14 max-w-2xl mx-auto">
            Trained on millions of rides across power, HR, HRV, sleep, environment and blood
            markers — the model sees what human coaches miss at scale.
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map((f) => (
              <div
                key={f.title}
                className="p-6 rounded-2xl bg-surface-card border border-surface-border hover:border-brand-500/40 transition-colors group"
              >
                <div className="w-10 h-10 rounded-lg bg-brand-500/10 flex items-center justify-center mb-4 group-hover:bg-brand-500/20 transition-colors">
                  <f.icon className="w-5 h-5 text-brand-400" />
                </div>
                <h3 className="font-semibold text-white mb-2">{f.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{f.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-20 px-6 bg-surface-muted border-y border-surface-border">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-white text-center mb-14">How it works</h2>
          <div className="space-y-10">
            {STEPS.map((step, i) => (
              <div key={i} className="flex gap-6 items-start">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-brand-500 flex items-center justify-center text-white font-bold text-sm">
                  {i + 1}
                </div>
                <div>
                  <h3 className="font-semibold text-white mb-1">{step.title}</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">{step.body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Metrics banner */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto grid sm:grid-cols-3 gap-8 text-center">
          {METRICS.map((m) => (
            <div key={m.label}>
              <div className="text-4xl font-extrabold text-brand-400 mb-1">{m.value}</div>
              <div className="text-sm text-slate-400">{m.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 border-t border-surface-border text-center">
        <h2 className="text-3xl font-bold text-white mb-4">Ready to train smarter?</h2>
        <p className="text-slate-400 mb-8 max-w-lg mx-auto">
          Import your full Strava history in one click. Start getting AI coaching today — no credit card required.
        </p>
        <Link
          href="/signup"
          className="inline-flex items-center gap-2 px-8 py-4 bg-brand-500 hover:bg-brand-600 text-white font-semibold rounded-xl transition-all hover:scale-105"
        >
          Get started free <ArrowRight className="w-4 h-4" />
        </Link>
      </section>

      <footer className="py-10 px-6 border-t border-surface-border text-center text-sm text-slate-500">
        © {new Date().getFullYear()} AI Coach — Built with ❤️ for cyclists
      </footer>
    </div>
  );
}

const FEATURES = [
  {
    icon: Brain,
    title: "Temporal Transformer Model",
    description:
      "A sequence model trained on training data ordered by time — it understands the cumulative effect of periodization, overreach, and recovery.",
  },
  {
    icon: TrendingUp,
    title: "FTP & Performance Prediction",
    description:
      "Forecasts FTP gains 4–12 weeks out based on your load, adaptation rate, and historical response to different stimulus types.",
  },
  {
    icon: Upload,
    title: "Import Everything",
    description:
      "Connect Strava or Garmin, bulk-import your full history, or drag-drop GPX / FIT files. The model uses all of it.",
  },
  {
    icon: BarChart3,
    title: "PMC & Zone Analysis",
    description:
      "Full Performance Management Chart (CTL, ATL, TSB), power zone distribution, HR drift, and aerobic efficiency trends.",
  },
  {
    icon: Shield,
    title: "Overtraining Risk Score",
    description:
      "Monitors monotony, ramp rate, HRV trends, and HR drift to flag overtraining and injury risk before it becomes a problem.",
  },
  {
    icon: Zap,
    title: "Event-Ready Planning",
    description:
      "Set a race or gran fondo goal date. The AI builds a periodized plan that peaks you at exactly the right moment.",
  },
];

const STEPS = [
  {
    title: "Connect your data sources",
    body: "Link Strava or Garmin OAuth, or upload GPX/FIT files directly. We parse every field: power, HR, cadence, GPS, temperature, and more.",
  },
  {
    title: "The model learns your patterns",
    body: "Your activity sequence is encoded as a time-series and fed through our cycling transformer. It identifies your unique response to load, rest, and different workout types.",
  },
  {
    title: "Get a personalized coaching plan",
    body: "Every day you receive a recommended workout with precise targets (power zones, HR ceiling, duration) and the reasoning behind it.",
  },
  {
    title: "Track fitness, adapt automatically",
    body: "As you complete workouts, the model re-calibrates. New blood test result? Add it. Slept badly? Log it. The plan adapts in real time.",
  },
];

const METRICS = [
  { value: "50+", label: "Training variables per ride" },
  { value: "42-day", label: "Fitness (CTL) horizon tracked" },
  { value: "1-click", label: "Full Strava history import" },
];
