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
  const { theme, setTheme } = useTheme(); // Hook to get and set the theme
  return (
    <nav
      className={`flex items-center justify-between p-4 ${
        theme === "dark"
          ? "shadow-[0px_4px_10px_rgba(255,255,255,0.1)]"
          : " shadow-lg"
      }`}
    >
      <Link href="/">
        <div className="flex items-center gap-2 text-lg font-bold hover:scale-110 transition-transform duration-300 ">
          <img
            src={
              theme === "dark" ? "/trading-icon-white.svg" : "/trading-icon.svg"
            }
            alt="Logo"
            className="h-8 w-8"
          />
          <div className="text-lg font-bold">MyApp</div>
        </div>
      </Link>

      <NavigationMenu>
        <NavigationMenuList className="flex items-center gap-4">
            <img
              src="/solana.png"
              alt="Solana Logo"
              className="h-10 w-10 hover:scale-110 transition-transform duration-300 "
            ></img>
          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-4">
            <ThemeToggle />
            {/* <Link
              href="/docs"
              className={`${navigationMenuTriggerStyle()} text-lg font-medium hover:text-blue-500`}
            >
              Docs
            </Link> */}
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
