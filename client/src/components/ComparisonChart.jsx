import { useState } from "react";
import { Scatter, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

const colors = {
  lr: "#2866cf",
  nn: "#c76132",
  train: "#97a5b8",
  test: "#334155",
};
const options = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: {
    legend: { display: false },
    tooltip: { mode: "nearest", intersect: false },
  },
  scales: {
    x: {
      type: "linear",
      title: { display: true, text: "X · input" },
      grid: { color: "#edf0f4" },
      border: { display: false },
    },
    y: {
      title: { display: true, text: "Y · output" },
      grid: { color: "#edf0f4" },
      border: { display: false },
    },
  },
};

export function ComparisonChart({ data, result, prediction }) {
  const [visible, setVisible] = useState({ lr: true, nn: true });
  const train =
    result?.split.train || data.x.map((x, i) => ({ x, y: data.y[i] }));
  const datasets = [
    {
      label: result ? "Training data" : "Data",
      data: train,
      backgroundColor: colors.train,
      pointRadius: 3.5,
    },
  ];
  if (result) {
    datasets.push({
      label: "Test data (held out)",
      data: result.split.test,
      backgroundColor: colors.test,
      pointStyle: "rectRot",
      pointRadius: 5,
    });
    for (const [key, name] of [
      ["lr", "Linear regression"],
      ["nn", "Neural network"],
    ]) {
      if (!visible[key]) continue;
      datasets.push({
        label: name,
        data: result[key].curve,
        showLine: true,
        borderColor: colors[key],
        backgroundColor: colors[key],
        borderWidth: 2.5,
        pointRadius: 0,
      });
      if (prediction)
        datasets.push({
          label: `${name} prediction`,
          data: [{ x: prediction.x, y: prediction[key] }],
          backgroundColor: colors[key],
          pointRadius: 7,
          pointStyle: "crossRot",
          borderColor: colors[key],
          borderWidth: 3,
        });
    }
  }
  return (
    <>
      <div
        className="chart"
        role="img"
        aria-label="Dataset and model predictions. Training points are circles; held-out test points are diamonds."
      >
        <Scatter data={{ datasets }} options={options} />
      </div>
      <div className="legend">
        <span>
          <i className="dot train" />
          {result ? "Training data" : "Data"}
        </span>
        {result && (
          <>
            <span>
              <i className="dot test" />
              Test data
            </span>
            {[
              ["lr", "Linear regression"],
              ["nn", "Neural network"],
            ].map(([key, label]) => (
              <label key={key} className={key}>
                <input
                  type="checkbox"
                  checked={visible[key]}
                  onChange={(event) =>
                    setVisible({ ...visible, [key]: event.target.checked })
                  }
                />
                {label}
              </label>
            ))}
          </>
        )}
      </div>
    </>
  );
}

export function LossChart({ loss }) {
  return (
    <div
      className="loss-chart"
      role="img"
      aria-label="Neural network training MSE over epochs, in original Y units squared."
    >
      <Line
        data={{
          datasets: [
            {
              data: loss.map((p) => ({ x: p.epoch, y: p.mse })),
              borderColor: colors.nn,
              borderWidth: 2,
              pointRadius: 0,
            },
          ],
        }}
        options={{
          ...options,
          scales: {
            x: { ...options.scales.x, title: { display: true, text: "Epoch" } },
            y: {
              ...options.scales.y,
              title: { display: true, text: "Training MSE" },
              beginAtZero: true,
            },
          },
        }}
      />
    </div>
  );
}
