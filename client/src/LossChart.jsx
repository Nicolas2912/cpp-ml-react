import React, { useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const LossChart = ({ data }) => {
    const chartRef = useRef(null); // Ref for the chart instance

    const chartData = {
        labels: data.map(point => point.epoch), // Use epoch for labels
        datasets: [
            {
                label: 'Training Loss (MSE)',
                data: data.map(point => point.mse), // Use mse for data points
                fill: false,
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                pointRadius: 2, // Smaller points might look better with lots of data
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false, // Allow chart to fill container height
        animation: {
             duration: 50 // Minimal animation
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Epoch',
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Mean Squared Error (MSE)',
                },
                beginAtZero: true,
            },
        },
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Neural Network Training Loss',
            },
        },
    };

    // Effect to update the chart when data changes
    // Note: We don't need manual .update() here typically,
    // react-chartjs-2 handles updates when props change.
    // We rely on React's rendering based on the 'data' prop.

    // Ensure chart cleanup on unmount
    useEffect(() => {
        const chartInstance = chartRef.current;
        return () => {
            if (chartInstance) {
                // chartInstance.destroy(); // react-chartjs-2 might handle this
            }
        };
    }, []);


    return (
        <div style={{ position: 'relative', height: '400px', width: '100%' }}> {/* Ensure container has dimensions */}
             <Line ref={chartRef} options={options} data={chartData} />
        </div>
    );
};

export default LossChart;
