import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from './providers';
import '@rainbow-me/rainbowkit/styles.css';
import { useAccount } from "wagmi";
import Navbar from "@/components/navbar";
import LoginScreen from "@/components/login-screen";

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {


    return (
        <html lang="en">
            <body>
                <Providers>
                    <div>
                        <Navbar />
                        <main className="flex flex-col min-h-screen mx-20 py-10">
                            {children}
                        </main>
                    </div>
                </Providers>
            </body>
        </html>
    );
}
