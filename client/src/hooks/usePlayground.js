import { useEffect, useMemo, useRef, useState } from "react";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LogarithmicScale,
} from "chart.js";
import useComparison from "./useComparison";
import { parseData, sampleData, format } from "../data";

ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LogarithmicScale,
);

// Seed of the "Noisy wave" sample: one where an unregularized network visibly overfits.
const NOISY_SEED = 12;
// Batch sizes at or above the number of training points use the full batch.
export const FULL_BATCH = 1000;
// An empty or partly typed number field is invalid, not zero.
const numberOrNaN = (value) => (String(value).trim() ? Number(value) : NaN);

export default function usePlayground() {
  const [inputs, setInputs] = useState({
    x: "1, 2, 3, 4, 5",
    y: "2, 4, 5, 4, 5",
  });
  const [activeTheme, setActiveTheme] = useState(() => {
    try {
      return localStorage.getItem("ml-theme") === "dark" ? "dark" : "light";
    } catch {
      return "light";
    }
  });
  const [numRandomPoints, setNumRandomPoints] = useState(20);
  const [linearityFactor, setLinearityFactor] = useState(0.7);
  const [nnLayerInput, setLayers] = useState("1-4-1");
  const [nnLearningRateInput, setRate] = useState("0.01");
  const [nnEpochsInput, setEpochs] = useState("1000");
  const [nnActivation, setActivation] = useState("sigmoid");
  const [nnOptimizer, setOptimizer] = useState("sgd");
  const [nnBatchSize, setBatchSize] = useState("1");
  const [nnWeightDecay, setWeightDecay] = useState("0");
  const [nnEarlyStopping, setEarlyStopping] = useState(false);
  const [nnPatience, setPatience] = useState("1000");
  const [lrModel, setLrModel] = useState(null);
  const [nnModel, setNnModel] = useState(null);
  const [trainingMode, setTrainingMode] = useState("both");
  const [lrError, setLrError] = useState(null);
  const [nnError, setNnError] = useState(null);
  const [wsError, setWsError] = useState(null);
  const [predictXInput, setPredictXInput] = useState("6");
  const [predictXInputNN, setPredictXInputNN] = useState("6");
  const [lrPoint, setLrPoint] = useState(null);
  const [nnPoint, setNnPoint] = useState(null);
  const [loadingLRPredict, setLoadingLRPredict] = useState(false);
  const [loadingNNPredict, setLoadingNNPredict] = useState(false);
  const [comparisonPrediction, setComparisonPrediction] = useState(null);
  const revision = useRef(0);
  const api = useComparison();
  const { run, connected, cancel } = api;
  const busy = run.status === "running";
  const isDark = activeTheme === "dark";
  const parsed = useMemo(() => {
    try {
      return { ...parseData(inputs), error: null };
    } catch (error) {
      return { x: [], y: [], error: error.message };
    }
  }, [inputs]);

  useEffect(() => {
    if (run.status !== "completed") return;
    if (run.result.lr)
      setLrModel({ id: run.id, ...run.result.lr, split: run.result.split });
    if (run.result.nn)
      setNnModel({
        id: run.id,
        ...run.result.nn,
        split: run.result.split,
        loss: run.loss,
      });
  }, [run]);
  useEffect(() => {
    if (connected) setWsError(null);
    else {
      revision.current++;
      setLrModel(null);
      setNnModel(null);
      setLrPoint(null);
      setNnPoint(null);
      setComparisonPrediction(null);
    }
  }, [connected]);
  useEffect(() => {
    if (run.error) {
      if (!connected) setWsError(run.error);
      else if (trainingMode === "lr") setLrError(run.error);
      else setNnError(run.error);
    }
  }, [run.error, connected, trainingMode]);

  const invalidate = (all = true) => {
    revision.current++;
    api.reset();
    if (all) {
      setLrModel(null);
      setLrPoint(null);
    }
    setNnModel(null);
    setNnPoint(null);
    setComparisonPrediction(null);
    setLrError(null);
    setNnError(null);
    setWsError(null);
  };
  const setXInput = (value) => {
    invalidate();
    setInputs((current) => ({ ...current, x: value }));
  };
  const setYInput = (value) => {
    invalidate();
    setInputs((current) => ({ ...current, y: value }));
  };
  const setNnLayerInput = (value) => {
    invalidate(false);
    setLayers(value);
  };
  const setNnLearningRateInput = (value) => {
    invalidate(false);
    setRate(value);
  };
  const setNnEpochsInput = (value) => {
    invalidate(false);
    setEpochs(value);
  };
  const setNnActivation = (value) => {
    invalidate(false);
    setActivation(value);
  };
  const setNnOptimizer = (value) => {
    invalidate(false);
    setOptimizer(value);
  };
  const setNnBatchSize = (value) => {
    invalidate(false);
    setBatchSize(value);
  };
  const setNnWeightDecay = (value) => {
    invalidate(false);
    setWeightDecay(value);
  };
  const setNnEarlyStopping = (value) => {
    invalidate(false);
    setEarlyStopping(value);
  };
  const setNnPatience = (value) => {
    invalidate(false);
    setPatience(value);
  };
  const toggleTheme = () => {
    const next = isDark ? "light" : "dark";
    setActiveTheme(next);
    try {
      localStorage.setItem("ml-theme", next);
    } catch {
      /* Theme still works if storage is unavailable. */
    }
  };
  const loadSample = (kind) => {
    invalidate();
    setInputs(
      kind === "noisy"
        ? sampleData("wave", 40, 0.5, NOISY_SEED)
        : sampleData(kind, 60),
    );
  };
  // A network big enough to memorize the noisy sample, trained long enough to do it.
  const loadOverfittingDemo = () => {
    loadSample("noisy");
    setLayers("1-32-32-1");
    setRate("0.01");
    setEpochs("5000");
    setActivation("tanh");
    setOptimizer("adam");
    setBatchSize(String(FULL_BATCH));
    setWeightDecay("0");
    setEarlyStopping(false);
    setPatience("1000");
  };
  const handleGenerateRandomData = () => {
    const n = Number(numRandomPoints);
    if (!Number.isInteger(n) || n < 2 || n > 200) {
      setLrError("Generate 2 to 200 points.");
      return;
    }
    const slope = (Math.random() - 0.5) * 5,
      intercept = Math.random() * 10;
    const noise =
      Math.max(1, Math.abs(slope * 50) * 0.4) * (1 - linearityFactor);
    const points = Array.from({ length: n }, () => {
      const x = Math.random() * 50;
      return {
        x: x.toFixed(2),
        y: (slope * x + intercept + (Math.random() - 0.5) * 2 * noise).toFixed(
          2,
        ),
      };
    });
    invalidate();
    setInputs({
      x: points.map((p) => p.x).join(", "),
      y: points.map((p) => p.y).join(", "),
    });
  };
  const train = (model) => {
    setLrError(null);
    setNnError(null);
    setWsError(null);
    setComparisonPrediction(null);
    if (parsed.error) {
      setLrError(parsed.error);
      return;
    }
    const patience = nnEarlyStopping ? numberOrNaN(nnPatience) : 0;
    if (model !== "lr" && nnEarlyStopping && !(patience >= 1)) {
      setNnError("Patience must be at least 1 epoch with early stopping on.");
      return;
    }
    revision.current++;
    setTrainingMode(model);
    if (model !== "nn") setLrPoint(null);
    if (model !== "lr") setNnPoint(null);
    api.start({
      x: parsed.x,
      y: parsed.y,
      model,
      layers: nnLayerInput,
      learningRate: numberOrNaN(nnLearningRateInput),
      epochs: numberOrNaN(nnEpochsInput),
      activation: nnActivation,
      optimizer: nnOptimizer,
      batchSize: numberOrNaN(nnBatchSize),
      weightDecay: numberOrNaN(nnWeightDecay),
      patience,
    });
  };
  const predictFor = async (model, value) => {
    const x = Number(value),
      cached = model === "nn" ? nnModel : lrModel;
    if (!String(value).trim() || !Number.isFinite(x) || Math.abs(x) > 1e6)
      throw new Error("Enter a finite X between −1,000,000 and 1,000,000.");
    if (!cached) throw new Error("Train this model first.");
    const current = revision.current;
    const point = await api.predict(x, cached.id, model);
    if (current !== revision.current)
      throw new Error("Model inputs changed. Predict again after training.");
    return point;
  };
  const predictOne = async (model) => {
    const setLoading =
      model === "lr" ? setLoadingLRPredict : setLoadingNNPredict;
    const setError = model === "lr" ? setLrError : setNnError;
    const setPoint = model === "lr" ? setLrPoint : setNnPoint;
    setLoading(true);
    setError(null);
    setPoint(null);
    try {
      setPoint(
        await predictFor(
          model,
          model === "lr" ? predictXInput : predictXInputNN,
        ),
      );
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };
  const predictBoth = async (x) => {
    let lr, nn;
    // A comparison run holds both models, so one request evaluates them together.
    if (lrModel && lrModel.id === nnModel?.id)
      lr = nn = await predictFor("both", x);
    else {
      lr = await predictFor("lr", x);
      nn = await predictFor("nn", x);
    }
    setLrPoint(lr);
    setNnPoint(nn);
    return {
      ...lr,
      nn: nn.nn,
      extrapolation: lr.extrapolation || nn.extrapolation,
    };
  };
  const lossHistory =
    busy && trainingMode !== "lr" ? run.loss : nnModel?.loss || [];
  const split = lrModel?.split || nnModel?.split;
  const comparisonResult = { split, lr: lrModel, nn: nnModel };
  const raw = parsed.x.map((x, i) => ({ x, y: parsed.y[i] }));
  const dataOnlyDatasets = [
    {
      label: "Original data",
      data: raw,
      backgroundColor: "#ef4444",
      pointRadius: 4,
    },
  ];
  const memoizedDatasets = [...dataOnlyDatasets];
  if (split?.test.length)
    memoizedDatasets.push({
      label: "Held-out test points",
      data: split.test,
      pointStyle: "rectRot",
      pointRadius: 6,
      backgroundColor: isDark ? "#cbd5e1" : "#475569",
    });
  if (nnModel?.split.validation?.length)
    memoizedDatasets.push({
      label: "Validation points (network)",
      data: nnModel.split.validation,
      pointRadius: 6,
      borderWidth: 2,
      borderColor: "#f59e0b",
      backgroundColor: "rgba(0, 0, 0, 0)",
    });
  if (lrModel)
    memoizedDatasets.push({
      label: "Linear regression",
      data: lrModel.curve,
      showLine: true,
      borderColor: "#3b82f6",
      backgroundColor: "#3b82f6",
      pointRadius: 0,
      borderWidth: 2,
    });
  if (nnModel)
    memoizedDatasets.push({
      label: "NN predictions",
      data: nnModel.curve,
      showLine: true,
      borderColor: "#f97316",
      backgroundColor: "#f97316",
      pointStyle: "triangle",
      pointRadius: 2,
      borderWidth: 2,
    });
  if (lrPoint)
    memoizedDatasets.push({
      label: "LR prediction",
      data: [{ x: lrPoint.x, y: lrPoint.lr }],
      pointStyle: "crossRot",
      pointRadius: 9,
      borderWidth: 3,
      borderColor: "#22c55e",
    });
  if (nnPoint)
    memoizedDatasets.push({
      label: "NN prediction",
      data: [{ x: nnPoint.x, y: nnPoint.nn }],
      pointStyle: "rect",
      pointRadius: 7,
      backgroundColor: "#a855f7",
    });
  const textColor = isDark ? "#cbd5e1" : "#475569",
    gridColor = isDark ? "#334155" : "#e2e8f0";
  const axis = (title) => ({
    type: "linear",
    title: { display: true, text: title, color: textColor },
    ticks: { color: textColor },
    grid: { color: gridColor },
  });
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { labels: { color: textColor } },
      tooltip: { mode: "nearest", intersect: false },
    },
    scales: { x: axis("X"), y: axis("Y") },
  };
  const hasValidation = lossHistory.some((p) => p.valMse !== undefined);
  const bestEpoch = !busy && nnModel?.bestEpoch ? nnModel.bestEpoch : null;
  const lossValues = lossHistory.flatMap((p) =>
    p.valMse === undefined ? [p.mse] : [p.mse, p.valMse],
  );
  const lossChartOptions = {
    ...baseOptions,
    plugins: {
      ...baseOptions.plugins,
      legend: {
        display: hasValidation,
        labels: { color: textColor, boxHeight: 2 },
      },
    },
    scales: {
      x: axis("Epoch"),
      y: {
        ...axis("MSE · original Y units²"),
        type: lossValues.some((value) => value === 0)
          ? "linear"
          : "logarithmic",
      },
    },
  };
  const lossChartData = {
    datasets: [
      {
        label: "Training MSE",
        data: lossHistory.map((p) => ({ x: p.epoch, y: p.mse })),
        borderColor: "#fb7185",
        backgroundColor: "#fb7185",
        pointRadius: 1,
        borderWidth: 1.5,
      },
      ...(hasValidation
        ? [
            {
              label: "Validation MSE",
              data: lossHistory
                .filter((p) => p.valMse !== undefined)
                .map((p) => ({ x: p.epoch, y: p.valMse })),
              borderColor: "#38bdf8",
              backgroundColor: "#38bdf8",
              pointRadius: 1,
              borderWidth: 1.5,
            },
          ]
        : []),
      ...(bestEpoch && lossValues.length
        ? [
            {
              label: nnModel.restoredBest
                ? `Restored best epoch (${bestEpoch.toLocaleString("en-US")})`
                : `Best validation epoch (${bestEpoch.toLocaleString("en-US")})`,
              data: [
                { x: bestEpoch, y: Math.min(...lossValues) },
                { x: bestEpoch, y: Math.max(...lossValues) },
              ],
              borderColor: "#f59e0b",
              backgroundColor: "#f59e0b",
              borderDash: [6, 4],
              pointRadius: 0,
              borderWidth: 1.5,
            },
          ]
        : []),
    ],
  };
  const formatMetric = (value, type) =>
    type === "time" && value != null ? `${format(value)} ms` : format(value);
  return {
    xInput: inputs.x,
    setXInput,
    yInput: inputs.y,
    setYInput,
    activeTheme,
    toggleTheme,
    isDark,
    numRandomPoints,
    setNumRandomPoints,
    linearityFactor,
    setLinearityFactor,
    nnLayerInput,
    setNnLayerInput,
    nnLearningRateInput,
    setNnLearningRateInput,
    nnEpochsInput,
    setNnEpochsInput,
    nnActivation,
    setNnActivation,
    nnOptimizer,
    setNnOptimizer,
    nnBatchSize,
    setNnBatchSize,
    nnWeightDecay,
    setNnWeightDecay,
    nnEarlyStopping,
    setNnEarlyStopping,
    nnPatience,
    setNnPatience,
    loadOverfittingDemo,
    lrModel,
    nnModel,
    lrSlope: lrModel?.slope,
    lrIntercept: lrModel?.intercept,
    lrMse: lrModel?.train.mse,
    lrRSquared: lrModel?.train.r2,
    lrTrainingTime: lrModel?.timeMs,
    isLrTrained: !!lrModel,
    lrPrediction: lrPoint?.lr ?? null,
    lastPredictedXLr: lrPoint?.x ?? null,
    nnPrediction: nnPoint?.nn ?? null,
    lastPredictedXNN: nnPoint?.x ?? null,
    nnResults: nnModel
      ? {
          finalMse: nnModel.train.mse,
          trainingTimeMs: nnModel.timeMs,
          validationMse: nnModel.validation?.mse ?? null,
          bestEpoch: nnModel.bestEpoch ?? null,
          epochsRun: nnModel.epochsRun ?? null,
          stoppedEarly: !!nnModel.stoppedEarly,
          restoredBest: !!nnModel.restoredBest,
          fitCount: nnModel.split.fit?.length ?? nnModel.split.train.length,
          validationCount: nnModel.split.validation?.length ?? 0,
        }
      : null,
    lrError,
    setLrError,
    nnError,
    setNnError,
    wsError,
    setWsError,
    loadingLRLinear: busy && trainingMode !== "nn",
    loadingNN: busy && trainingMode !== "lr",
    loadingLRPredict,
    loadingNNPredict,
    predictXInput,
    setPredictXInput,
    predictXInputNN,
    setPredictXInputNN,
    parsedX: parsed.x,
    parsedY: parsed.y,
    dataPointCount: parsed.x.length,
    datasetReady: !parsed.error,
    handleGenerateRandomData,
    handleTrainLR: () => train("lr"),
    handleTrainPredictNN: () => train("nn"),
    handleCompare: () => train("both"),
    handlePredictLR: () => predictOne("lr"),
    handlePredictNN: () => predictOne("nn"),
    formatMetric,
    lossHistory,
    dataOnlyDatasets,
    memoizedDatasets,
    dataOnlyChartOptions: {
      ...baseOptions,
      plugins: { ...baseOptions.plugins, legend: { display: false } },
    },
    overlayChartOptions: baseOptions,
    lossChartData,
    lossChartOptions,
    busy,
    connected,
    run,
    cancel,
    trainingMode,
    loadSample,
    comparisonResult,
    predictBoth,
    comparisonPrediction,
    setComparisonPrediction,
  };
}
