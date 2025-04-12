"use client";
import { Button } from "@/components/ui/button";
import { TypeAnimation } from "react-type-animation";
import Link from "next/link";
import { useTheme } from "next-themes";

export default function Home() {
  const { theme, setTheme } = useTheme(); // Hook to get and set the theme

  return (
    <div
      className={`px-4 sm:px-8 md:px-16 lg:px-32 py-12 ${
        theme === "dark" ? "bg-[#0b0f19] text-[#e2e8f0]" : "bg-white"
      }`}
    >
      <div className="text-center">
        <TypeAnimation
          sequence={["Welcome to MyApp", 1000]}
          wrapper="span"
          speed={50}
          className={`font-bold text-5xl sm:text-6xl theme === "dark" ? "text-white" : "text-gray-800`}
          style={{ display: "inline-block" }}
          cursor={true}
        />
        <p
          className={`text-xl sm:text-2xl mt-6 max-w-2xl mx-auto mt-4 ${
            theme === "dark" ? "text-gray-400" : "text-gray-400"
          }`}
        >
          This is a website where ... [description]
        </p>
      </div>

      <div className="mt-12 text-center">
        <h2
          className={`text-4xl font-semibold mb-10 ${
            theme === "dark" ? "text-gray-500" : "text-[#90cdf4]"
          }`}
        >
          How does it work?
        </h2>

        <div className="grid gap-8 sm:grid-cols-1 md:grid-cols-2">
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
                📈 Auto Monitoring
              </h3>
              <p className="text-gray-700 mb-6 dark:text-gray-400">
                [Description of the auto monitoring feature.] <br />
                [Have graph to show real-time data and predicted data]
              </p>
              <Link href="/trade">
                <Button
                  variant="default"
                  size="lg"
                  className={`font-semibold ${
                    theme === "dark" ? "bg-green-500 hover:bg-green-400" : "bg-green-600 hover:bg-green-300"
                  } hover:${
                    theme === "dark" ? "bg-green-400 hover:bg-green-300" : "bg-green-500 hover:bg-green-400"
                  } text-white hover:scale-[1.02]`}
                >
                  Start Monitoring →
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
              <h3 className={`text-2xl font-bold mb-4 ${theme === "dark" ? "text-blue-300": "text-gray-900"}`}>
                📰 News
              </h3>
              <p
                className={`mb-6 ${
                  theme === "dark" ? "text-gray-400" : "text-gray-700"
                }`}
              >
                [Description of the news feature.]
              </p>
              <Link href="/news">
                <Button
                  variant="default"
                  size="lg"
                  className={`font-semibold ${
                    theme === "dark" ? "bg-blue-500 hover:bg-blue-400" : "bg-blue-600 hover:bg-blue-500"
                  } hover:${
                    theme === "dark" ? "bg-blue-400 hover:bg-blue-300" : "bg-blue-500 hover:bg-blue-400"
                  } text-white hover:scale-[1.02]`}
                >
                  Read News →
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
