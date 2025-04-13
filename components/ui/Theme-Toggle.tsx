"use client";

import { useTheme } from "next-themes";
import { useEffect, useState, useRef } from "react";

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        tooltipRef.current &&
        !tooltipRef.current.contains(event.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(event.target as Node)
      ) {
        setTooltipVisible(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!mounted) return null;

  const currentTheme = theme ?? "light";
  const nextTheme = currentTheme === "light" ? "dark" : "light";
  const tooltipText = `You're in ${
    currentTheme === "light" ? "Light Mode" : "Dark Mode"
  }. Click to switch!!`;

  return (
    <div
      className="relative inline-block"
      ref={triggerRef}
      onMouseEnter={() => setTooltipVisible(true)}
      onMouseLeave={() => setTooltipVisible(false)}
    >
      <button
        onClick={() => setTheme(nextTheme)}
        className={`p-2 rounded-full transition-all duration-300 transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-offset-2 ${
          currentTheme === "dark"
            ? "bg-gray-700 text-white focus:ring-gray-500"
            : "bg-[#f2e2ba] text-black focus:ring-yellow-500"
        }`}
        aria-label={`Switch to ${nextTheme} mode`}
      >
        {currentTheme === "light" ? "🌙" : "🌞"}
      </button>

      {tooltipVisible && (
        <div
          className={`absolute top-full left-1/2 -translate-x-1 mt-2 max-w-md whitespace-normal text-center text-xs py-1 px-3 rounded z-50 opacity-0 animate-fade-in pointer-events-none
          ${
            currentTheme === "dark"
              ? "bg-gray-700 text-white focus:ring-gray-500"
              : "bg-[#f2e2ba] text-gray-500 focus:ring-yellow-500"
          }
        `}
        >
          {tooltipText}
        </div>
      )}

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translate(-50%, 5px);
          }
          to {
            opacity: 1;
            transform: translate(-50%, 0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.2s ease-out forwards;
        }
      `}</style>
    </div>
  );
}
