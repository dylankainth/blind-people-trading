import GraphView from "@/components/graph-view";

export default function TradePage() {


    return (
        <div>
            <div className="pb-10">
                <h1 className="text-8xl font-bold">Trade</h1>
                <p className="text-gray-600 text-2xl pt-2">Trade your assets here.</p>
            </div>
            <GraphView />
        </div>
    )
}