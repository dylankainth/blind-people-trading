"use client";
export default function Trade() {
  const playTone = (frequency: number, duration: number = 500) => {
    const audioCtx = new (window.AudioContext || window.AudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = "sine"; // Options: 'sine', 'square', 'sawtooth', 'triangle'
    oscillator.frequency.setValueAtTime(frequency, audioCtx.currentTime);

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + duration / 10000);
  };

  return (
    <div className="p-4">
      <button
        onClick={() => playTone(440)}
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        Play A4 (440 Hz)
      </button>
    </div>
  );
}
