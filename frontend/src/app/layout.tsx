import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export const metadata: Metadata = {
  title: "AI Coach — Intelligent Cycling Coaching",
  description:
    "Transformer-powered AI coach that learns your unique patterns to optimize training, predict FTP gains, and build the perfect plan for your goals.",
  keywords: ["cycling coach", "AI training", "power meter", "FTP", "Strava"],
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "AI Coach",
  },
  other: {
    "mobile-web-app-capable": "yes",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <meta name="theme-color" content="#10b981" />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
