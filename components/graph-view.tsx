"use client"

import { Activity, TrendingUp } from "lucide-react"
import { Line, LineChart, CartesianGrid, XAxis } from "recharts"

import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card"
import {
    ChartConfig,
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart"

const chartData = [
    { month: "January", historical: 186 },
    { month: "February", historical: 305 },
    { month: "March", historical: 237 },
    { month: "April", historical: 73, predicted: 73 },
    { month: "May", predicted: 209 },
    { month: "June", predicted: 214 },
]

const chartConfig = {
    historical: {
        label: "Historical",
        color: "hsl(var(--sky-600))",
        icon: Activity,
    },
    predicted: {
        label: "Predicted",
        color: "hsl(var(--amber-400))",
        icon: Activity,
    },
} satisfies ChartConfig

const GraphView: React.FC = () => {
    return (
        <Card>
            <CardHeader>
                <CardTitle>Line Chart - Linear</CardTitle>
                <CardDescription>January - June 2024</CardDescription>
            </CardHeader>
            <CardContent>
                <ChartContainer config={chartConfig}>
                    <LineChart
                        accessibilityLayer
                        data={chartData}
                        margin={{
                            left: 12,
                            right: 12,
                        }}
                    >
                        <CartesianGrid vertical={false} />
                        <XAxis
                            dataKey="month"
                            tickLine={false}
                            axisLine={false}
                            tickMargin={8}
                            tickFormatter={(value) => value.slice(0, 3)}
                        />
                        <ChartTooltip
                            cursor={false}
                            content={<ChartTooltipContent hideLabel />}
                        />
                        <Line
                            dataKey="historical"
                            type="linear"
                            stroke="#0ea5e9"
                            strokeWidth={2}
                            dot={false}
                        />
                        <Line
                            dataKey="predicted"
                            type="linear"
                            stroke="#fbbf24"
                            strokeWidth={2}
                            dot={false}
                        />
                    </LineChart>
                </ChartContainer>
            </CardContent>
            <CardFooter className="flex-col items-start gap-2 text-sm">
                <div className="flex gap-2 font-medium leading-none">
                    Trending up by 5.2% this month <TrendingUp className="h-4 w-4" />
                </div>
                <div className="leading-none text-muted-foreground">
                    Showing total visitors for the last 6 months
                </div>
            </CardFooter>
        </Card>
    )
}

export default GraphView
