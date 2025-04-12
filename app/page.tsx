"use client";
import { Button } from "@/components/ui/button";
import { TypeAnimation } from "react-type-animation";
import Link from "next/link";

export default function Home() {
  return (
    <div className="px-4 sm:px-8 md:px-16 lg:px-32 py-12">
      <div className="text-center">
        <TypeAnimation
          sequence={["Welcome to MyApp", 1000]}
          wrapper="span"
          speed={50}
          className="font-bold text-5xl sm:text-6xl text-gray-800"
          style={{ display: "inline-block" }}
          cursor={true}
        />
        <p className="text-xl sm:text-2xl text-gray-600 mt-6 max-w-2xl mx-auto">
          This is a website where ... [description]
        </p>
      </div>

      <div className="mt-6 text-center">
        <h2 className="text-4xl font-semibold text-gray-800 mb-8">How does it work?</h2>
        <div className="grid gap-8 sm:grid-cols-1 md:grid-cols-2">
          {/* Auto Monitoring Card */}
          <div className="bg-white rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-[1.05]">
            <div className="p-8 text-center">
              <h3 className="text-2xl font-bold mb-4">📈 Auto Monitoring</h3>
              <p className="text-gray-700 mb-6">
                [Description of the auto monitoring feature.] <br />
                [Have graph to show real-time data and predicted data]
              </p>
              <Link href="/trade">
                <Button variant="default" size="lg" className="font-semibold hover:scale-[1.02]">
                  Start Monitoring →
                </Button>
              </Link>
            </div>
          </div>

          {/* News Card */}
          <div className="bg-white rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-[1.05]">
            <div className="p-8 text-center">
              <h3 className="text-2xl font-bold mb-4">📰 News</h3>
              <p className="text-gray-700 mb-6">
                [Description of the news feature.]
              </p>
              <Link href="/news">
                <Button variant="default" size="lg" className="font-semibold hover:scale-[1.02]">
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
