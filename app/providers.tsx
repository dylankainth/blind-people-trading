'use client';

import * as React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WagmiProvider, useAccount } from 'wagmi';
import { RainbowKitProvider, ConnectButton } from '@rainbow-me/rainbowkit';
import { useRouter } from 'next/navigation';

import { config } from '../wagmi';

const queryClient = new QueryClient();

function AuthWrapper({ children }: { children: React.ReactNode }) {
    const { isConnected } = useAccount();
    const router = useRouter();

    React.useEffect(() => {
        if (!isConnected) {
            router.prefetch('/'); // Prefetch the homepage or login page for better UX
        }
    }, [isConnected, router]);

    if (!isConnected) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
                <ConnectButton />
            </div>
        );
    }

    return <>{children}</>;
}

export function Providers({ children }: { children: React.ReactNode }) {
    return (
        <WagmiProvider config={config}>
            <QueryClientProvider client={queryClient}>
                <RainbowKitProvider>
                    <AuthWrapper>{children}</AuthWrapper>
                </RainbowKitProvider>
            </QueryClientProvider>
        </WagmiProvider>
    );
}