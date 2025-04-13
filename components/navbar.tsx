"use client";
import React from "react";
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuLink,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import Link from "next/link";
import { ConnectButton } from "@rainbow-me/rainbowkit";
import ThemeToggle from "./ui/Theme-Toggle";
import { useTheme } from "next-themes";

const Navbar: React.FC = () => {
  const { theme } = useTheme(); // Hook to get and set the theme
  return (
    <nav
      className={`relative z-10 bg-background text-foreground flex items-center justify-between p-4 ${
        theme === "dark"
          ? "shadow-[0px_4px_10px_rgba(255,255,255,0.1)] bg-gradient-to-r from-[#14F195] to-[#6345EB]" // Dark mode with Solana gradient
          : "shadow-lg bg-gradient-to-r from-[#14F195] to-[#6345EB]" // Light mode with Solana gradient
      }`}
    >
      <Link href="/">
        <div className="flex items-center gap-2 text-lg font-bold hover:scale-107 transition-transform duration-300 ">
          <img
            src={
              "/trading-icon.svg"
            }
            alt="Logo"
            className="h-8 w-8"
          />
          <div className="text-lg font-bold text-black">TradingBlind</div>
          <img
            src="/solana.png"
            alt="Solana Logo"
            className="h-10 w-10 transition-transform duration-300 ml-1"
          ></img>
        </div>
      </Link>

      <NavigationMenu>
        <NavigationMenuList className="flex items-center gap-4">
          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-4">
            <ThemeToggle />
          </div>

          {/* ConnectButton */}
          <NavigationMenuLink>
            <div className="flex items-center">
              <ConnectButton />
            </div>
          </NavigationMenuLink>
        </NavigationMenuList>
      </NavigationMenu>
    </nav>
  );
};

export default Navbar;
