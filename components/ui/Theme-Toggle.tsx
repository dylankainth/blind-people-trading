"use client"
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Wait until the component is mounted to avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  // If the component is not mounted yet, render a placeholder or nothing
  if (!mounted) {
    return <div>Loading...</div> // You can replace this with a loader or nothing
  }

  // Ensure theme is not null or undefined
  const currentTheme = theme ?? 'light'

  return (
    <button
      onClick={() => setTheme(currentTheme === 'light' ? 'dark' : 'light')}
      className={`p-2 rounded-full transition-colors hover:scale-110 transition-transform duration-300  ${currentTheme === 'dark' ? 'bg-gray-700 text-white' : 'bg-[#f2e2ba] text-black'}`}
      aria-label="Toggle theme"
    >
      {currentTheme === 'light' ? '🌙' : '🌞'}
    </button>
  )
}
