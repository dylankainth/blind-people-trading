import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import "@rainbow-me/rainbowkit/styles.css";
import { useAccount } from "wagmi";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "next-themes";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* You can include the metadata here if needed */}
      </head>
      <body>
        {/* Providers to wrap around app context */}
        <Providers>
          {/* Theme provider for controlling light/dark mode */}
          <ThemeProvider attribute="class" defaultTheme="light">
            <div className="min-h-screen flex flex-col">
              {/* Navbar */}
              <Navbar />

              {/* Main content area */}
              <main className="">{children}</main>
            </div>
          </ThemeProvider>
        </Providers>
      </body>
    </html>
  );
}
