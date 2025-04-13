"use client";
import GraphView from "@/components/graph-view";
import { motion } from "framer-motion";
import { useTheme } from "next-themes";
import { TypeAnimation } from "react-type-animation";

export default function TradePage() {
  const { theme } = useTheme();

  return (
    <div>
      <div className="pb-10">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <TypeAnimation
            key={theme} // Add key prop here
            sequence={["Trade", 1000]}
            wrapper="span"
            speed={10}
            className={`text-8xl font-bold ${
              theme === "dark" ? "text-gray-200" : "text-black"
            }`}
            style={{ display: "inline-block" }}
            cursor={true}
          />
        </motion.div>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className={`text-2xl pt-2 mt-6 px-2 py-2 ${
            theme === "dark" ? "text-gray-400" : "text-gray-600"
          }`}
        >
          Trade your assets here.
        </motion.p>
      </div>
      <GraphView />
    </div>
  );
}
