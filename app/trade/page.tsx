"use client"

import { useEffect, useRef, useState } from "react"

export default function SolanaPrice() {
  const [price, setPrice] = useState<string | null>(null)
  const [status, setStatus] = useState("Click to start")
  const [isAudioReady, setIsAudioReady] = useState(false)
  const [frequency, setFrequency] = useState(440)

  const prevPriceRef = useRef<number | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const oscillatorRef = useRef<OscillatorNode | null>(null)
  const gainNodeRef = useRef<GainNode | null>(null)

  const initAudio = () => {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
      const oscillator = audioCtx.createOscillator()
      const gainNode = audioCtx.createGain()

      oscillator.type = "sine"
      oscillator.frequency.setValueAtTime(440, audioCtx.currentTime)

      oscillator.connect(gainNode)
      gainNode.connect(audioCtx.destination)
      gainNode.gain.setValueAtTime(0.08, audioCtx.currentTime)

      oscillator.start()

      audioCtxRef.current = audioCtx
      oscillatorRef.current = oscillator
      gainNodeRef.current = gainNode
    }

    if (audioCtxRef.current?.state === "suspended") {
      audioCtxRef.current.resume()
    }

    setIsAudioReady(true)
    setStatus("Connecting...")
  }

  const updateFrequencyFromDelta = (newPrice: number) => {

    const prevPrice = prevPriceRef.current
    if (!oscillatorRef.current || !audioCtxRef.current || prevPrice === null) return

    const delta = newPrice - prevPrice

    // Map price delta to pitch delta (e.g. $1 = 10Hz pitch change)
    const pitchChange = delta * 150
    const newFreq = oscillatorRef.current.frequency.value + pitchChange

    // Clamp to sensible range
    const clampedFreq = Math.max(100, Math.min(1200, newFreq))

    oscillatorRef.current.frequency.linearRampToValueAtTime(
      clampedFreq,
      audioCtxRef.current.currentTime + 0.1
    )
  }

  useEffect(() => {
    if (!isAudioReady) return

    const socket = new WebSocket("wss://stream.binance.com:9443/ws/btcusdt@trade")

    socket.onopen = () => setStatus("Connected")

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (!data.p) return

      const numericPrice = parseFloat(data.p)
      setPrice(numericPrice.toFixed(2))

      if (prevPriceRef.current !== null) {
        updateFrequencyFromDelta(numericPrice)
      }

      prevPriceRef.current = numericPrice
    }

    socket.onerror = (error) => {
      console.error("WebSocket Error:", error)
      setStatus("Error")
    }

    socket.onclose = () => setStatus("Disconnected")

    return () => socket.close()
  }, [isAudioReady])

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 text-gray-800">
      <h1 className="text-3xl font-bold mb-4">Real-time Solana Price</h1>

      {isAudioReady ? (
        <>
          <div className="text-6xl font-mono text-green-600">
            {price ? `$${price}` : "Loading..."}
          </div>
          <p className="mt-4 text-sm text-gray-500">Status: {status}</p>
        </>
      ) : (
        <button
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          onClick={initAudio}
        >
          Start Audio + Connect
        </button>
      )}
    </div>
  )
}
