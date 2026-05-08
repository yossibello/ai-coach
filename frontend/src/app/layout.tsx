import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "AI Coach — Intelligent Cycling Coaching",
  description:
    "Transformer-powered AI coach that learns your unique patterns to optimize training, predict FTP gains, and build the perfect plan for your goals.",
  keywords: ["cycling coach", "AI training", "power meter", "FTP", "Strava"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
