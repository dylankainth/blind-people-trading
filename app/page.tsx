"use client";
import { Button } from "@/components/ui/button";
import { TypeAnimation } from "react-type-animation";
import Link from "next/link";
import { useTheme } from "next-themes";
import { motion } from "framer-motion";

export default function Home() {
  const { theme } = useTheme();

  return (
    <div
      className={`px-4 sm:px-8 md:px-16 lg:px-32 py-12 ${
        theme === "dark" ? "bg-[#0b0f19] text-[#e2e8f0]" : "bg-white"
      }`}
    >
      <div className="text-center">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <TypeAnimation
            key={theme} // Add key prop here
            sequence={["Welcome to MyApp", 1000]}
            wrapper="span"
            speed={50}
            className={`font-bold text-5xl sm:text-6xl ${
              theme === "dark" ? "text-gray-200" : "text-gray-600"
            }`}
            style={{ display: "inline-block" }}
            cursor={true}
          />
        </motion.div>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className={`text-xl sm:text-2xl mt-6 max-w-2xl mx-auto mt-4 ${
            theme === "dark" ? "text-gray-400" : "text-gray-400"
          }`}
        >
          This is a trading platform designed specifically for visually impaired
          individuals, aiming to transform the complex world of cryptocurrency
          into an accessible experience. Hear real-time price changes, get
          AI-powered predictions & news summaries.
        </motion.p>
      </div>

      <motion.div
        className="mt-12 text-center"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.7 }}
      >
        <h2
          className={`text-4xl font-semibold mb-10 ${
            theme === "dark" ? "text-gray-500" : "text-[#90cdf4]"
          }`}
        >
          How does it work?
        </h2>
      </motion.div>

      <motion.div
        className="grid gap-8 sm:grid-cols-1 md:grid-cols-2"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.1, duration: 0.6 }}
      >
        {/* Auto Monitoring Card */}
        <div
          className={`rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-[1.05] ${
            theme === "dark"
              ? "bg-[#1a202c] border border-gray-700"
              : "text-[#90cdf4] bg-[#fbfbfb]"
          }`}
        >
          <div className="p-8 text-center">
            <h3
              className={`text-2xl font-bold mb-4 ${
                theme === "dark" ? "text-green-400" : "text-gray-800"
              }`}
            >
              📈 Real-time predictions
            </h3>
            <p
              className={`mb-6 ${
                theme === "dark" ? "text-gray-400" : "text-gray-700"
              }`}
            >
              Hear Solana's price movements with intuitive audio cues. Get
              accurate hourly predictions powered by machine learning.
            </p>
            <Link href="/trade">
              <Button
                variant="default"
                size="lg"
                className={`font-semibold ${
                  theme === "dark"
                    ? "bg-green-500 hover:bg-green-400"
                    : "bg-green-600 hover:bg-green-300"
                } text-white hover:scale-[1.02]`}
              >
                Start Trading →
              </Button>
            </Link>
          </div>
        </div>

        {/* News Card */}
        <div
          className={`rounded-2xl overflow-hidden shadow-md transition-all duration-300 transform hover:scale-[1.05] ${
            theme === "dark"
              ? "bg-[#1a202c] border-gray-700"
              : "bg-[#fbfbfb] hover:shadow-xl"
          }`}
        >
          <div className="p-8 text-center">
            <h3
              className={`text-2xl font-bold mb-4 ${
                theme === "dark" ? "text-blue-300" : "text-gray-900"
              }`}
            >
              📰 AI summarised-news and insights
            </h3>
            <p
              className={`mb-6 ${
                theme === "dark" ? "text-gray-400" : "text-gray-700"
              }`}
            >
              Stay informed with AI-summarized cryptocurrency news. Receive
              intelligent trading suggestions based on market analysis.
            </p>
            <Link href="/news">
              <Button
                variant="default"
                size="lg"
                className={`font-semibold ${
                  theme === "dark"
                    ? "bg-blue-500 hover:bg-blue-400"
                    : "bg-blue-600 hover:bg-blue-500"
                } text-white hover:scale-[1.02]`}
              >
                Read News and Insights →
              </Button>
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
