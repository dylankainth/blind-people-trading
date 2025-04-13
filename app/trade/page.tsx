'use client'
import GraphView from "@/components/graph-view";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Star } from "lucide-react";
import { useEffect, useState } from "react";

export default function Home() {

    // make a request to http://localhost:8000/price?symbol=SOL-USD&interval=1mo&range=max

    const [data, setData] = useState<any[]>([]);

    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch("http://localhost:8000/price?symbol=SOL-USD&interval=1mo&range=max");
            const result = await response.json();

            result.chart.result[0].parsed_quotes.map((item: any) => {
                item.date = new Date(item.TimestampUTC);
            })
            setData(result.chart.result[0].parsed_quotes);
        };

        fetchData();
    }
        , []);

    return (
        <main className="flex min-h-screen flex-col items-center justify-start sm:justify-start md:justify-center px-4 sm:px-8 md:px-12 lg:px-24 -my-25">

            <div className="flex justify-center w-full">
                <div className="flex flex-col items-center max-w-[1500px] w-full">
                    <div className="w-full h-[300px] sm:h-[400px] md:h-[500px] mb-12 lg:mb-0 lg:hidden">
                        <GraphView data={data} />

                    </div>
                    <div className="flex flex-col lg:flex-row items-center w-full">
                        <div className="w-full lg:w-1/3 lg:pr-8 mb-6 lg:mb-0">

                            <div className="pb-10">
                                <h1 className="text-8xl font-bold">Trade</h1>
                                <p className="text-gray-600 text-2xl pt-2">Trade your assets here.</p>


                                {/* add buttons and input to buy and sell solana; one input field, two buttons */}
                                <div className="flex items-center mt-6">
                                    <input type="text" placeholder="Amount" className="border border-gray-300 rounded-lg px-4 py-2 mr-4 w-full" />
                                    <Button
                                        variant="default"
                                        className="mr-4"
                                        onClick={() => {
                                            alert("Success: Solana bought!");
                                        }}
                                    >
                                        Buy
                                    </Button>
                                    <Button
                                        variant="destructive"
                                        onClick={() => {
                                            alert("Success: Solana sold!");
                                        }}
                                    >
                                        Sell
                                    </Button>
                                </div>


                            </div>

                        </div>
                        <div className="w-full lg:w-2/3 h-[300px] sm:h-[400px] md:h-[500px] hidden lg:block">
                            <GraphView data={data} />
                        </div>
                    </div>
                </div>
            </div>

        </main>
    );
}
