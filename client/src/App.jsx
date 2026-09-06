import { useState, lazy, Suspense } from "react";
import { Scatter, Line } from "react-chartjs-2";
const NNVisualizer = lazy(() => import("./NNVisualizer"));
import Results from "./components/Results";
import Predict from "./components/Predict";
import usePlayground from "./hooks/usePlayground";
import "./App.css";

function App() {
  const {
    xInput,
    setXInput,
    yInput,
    setYInput,
    predictXInput,
    setPredictXInput,
    predictXInputNN,
    setPredictXInputNN,
    activeTheme,
    toggleTheme,
    isDark,
    numRandomPoints,
    setNumRandomPoints,
    linearityFactor,
    setLinearityFactor,
    lrSlope,
    lrIntercept,
    lrPrediction,
    lastPredictedXLr,
    isLrTrained,
    lrTrainingTime,
    lrMse,
    lrRSquared,
    nnLayerInput,
    setNnLayerInput,
    nnLearningRateInput,
    setNnLearningRateInput,
    nnEpochsInput,
    setNnEpochsInput,
    nnResults,
    nnError,
    setNnError,
    loadingNN,
    lossHistory,
    wsError,
    setWsError,
    nnPrediction,
    lastPredictedXNN,
    loadingNNPredict,
    loadingLRLinear,
    loadingLRPredict,
    lrError,
    setLrError,
    parsedX,
    parsedY,
    dataPointCount,
    datasetReady,
    handleGenerateRandomData,
    handleTrainLR,
    handleTrainPredictNN,
    handlePredictLR,
    handlePredictNN,
    handleCompare,
    formatMetric,
    dataOnlyDatasets,
    memoizedDatasets,
    dataOnlyChartOptions,
    overlayChartOptions,
    lossChartData,
    lossChartOptions,
    busy,
    connected,
    run,
    cancel,
    trainingMode,
    loadSample,
    lrModel,
    nnModel,
    comparisonResult,
    predictBoth,
    comparisonPrediction,
    setComparisonPrediction,
  } = usePlayground();
  const [activeModelPanel, setActiveModelPanel] = useState("lr");
  const [blueprintVisibility, setBlueprintVisibility] = useState("hidden");
  const appBackgroundClass = isDark
    ? "bg-slate-950 text-slate-100"
    : "bg-slate-50 text-slate-900";
  const ambientGradientClass = isDark
    ? "from-indigo-900/80 via-slate-950 to-slate-950"
    : "from-slate-100 via-white to-slate-200";
  const panelSurfaceClass = isDark
    ? "bg-slate-900/80 border-slate-800/80 text-slate-100"
    : "bg-white border-slate-200 text-slate-900";
  const panelClass = `rounded-3xl border shadow-xl backdrop-blur-xl transition duration-300 ${panelSurfaceClass}`;
  const ribbonClass = isDark
    ? "bg-indigo-500/20 text-indigo-200 border border-indigo-400/40"
    : "bg-indigo-100 text-indigo-700 border border-indigo-300";
  const sectionLabelClass = isDark
    ? "text-[0.65rem] font-semibold uppercase tracking-[0.45em] text-indigo-300/80"
    : "text-[0.65rem] font-semibold uppercase tracking-[0.45em] text-indigo-600/70";
  const baseInputClasses =
    "min-w-0 w-full rounded-2xl border px-4 py-3 text-base shadow-sm transition focus:outline-none focus:ring-2 focus:ring-offset-0 disabled:opacity-60 disabled:cursor-not-allowed";
  const inputSurfaceClasses = isDark
    ? "bg-slate-900/60 border-slate-700 text-slate-100 placeholder-slate-500 focus:border-indigo-400 focus:ring-indigo-400/70"
    : "bg-white border-slate-200 text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:ring-indigo-500/40";
  const inputClassName = `${baseInputClasses} ${inputSurfaceClasses}`;
  const textareaClassName = `${inputClassName} min-h-[120px] resize-y leading-relaxed`;
  const selectClassName = `${inputClassName} appearance-none pr-12`;
  const selectArrowClass = isDark ? "text-slate-200" : "text-slate-500";
  const sliderTrackClass = isDark
    ? "accent-indigo-400 text-indigo-300"
    : "accent-indigo-500 text-indigo-500";
  const fieldLabelClass = isDark
    ? "text-[0.75rem] font-semibold uppercase tracking-[0.3em] text-slate-200"
    : "text-[0.75rem] font-semibold uppercase tracking-[0.3em] text-slate-600";
  const mutedTextClass = isDark ? "text-slate-400" : "text-slate-600";
  const primaryButtonClass = `inline-flex items-center justify-center gap-2 rounded-2xl px-5 py-3 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-offset-0 disabled:opacity-60 disabled:pointer-events-none ${isDark ? "bg-indigo-500/90 hover:bg-indigo-400 focus:ring-indigo-400/70 text-white shadow-lg shadow-indigo-900/40" : "bg-indigo-600 hover:bg-indigo-500 focus:ring-indigo-500/50 text-white shadow-lg shadow-indigo-300/40"}`;
  const secondaryButtonClass = `inline-flex items-center justify-center gap-2 rounded-2xl px-5 py-3 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-offset-0 disabled:opacity-60 disabled:pointer-events-none ${isDark ? "bg-emerald-500/90 hover:bg-emerald-400 focus:ring-emerald-400/70 text-white shadow-lg shadow-emerald-900/30" : "bg-emerald-500 hover:bg-emerald-400 focus:ring-emerald-500/40 text-white shadow-lg shadow-emerald-300/40"}`;
  const outlineButtonClass = `inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-2 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-offset-0 disabled:opacity-60 disabled:pointer-events-none ${isDark ? "border border-slate-700/80 bg-transparent text-slate-100 hover:border-slate-500 focus:ring-slate-500/40" : "border border-slate-200 bg-white text-slate-700 hover:border-slate-400 focus:ring-slate-400/30"}`;
  const accentButtonClass = `inline-flex items-center justify-center gap-2 rounded-2xl px-5 py-3 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-offset-0 disabled:opacity-60 disabled:pointer-events-none ${isDark ? "bg-fuchsia-500/80 hover:bg-fuchsia-400 focus:ring-fuchsia-400/70 text-white shadow-lg shadow-fuchsia-900/30" : "bg-fuchsia-500 hover:bg-fuchsia-400 focus:ring-fuchsia-500/40 text-white shadow-lg shadow-fuchsia-300/40"}`;
  const metricCardClass = isDark
    ? "rounded-2xl border border-slate-800/80 bg-slate-900/50 p-5 shadow-inner shadow-black/20"
    : "rounded-2xl border border-slate-200 bg-slate-50/80 p-5 shadow-inner shadow-slate-200/60";
  const pillClass = isDark
    ? "inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-200"
    : "inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600";

  // --- JSX Return ---
  return (
    <div
      data-theme={activeTheme}
      className={`min-h-screen ${appBackgroundClass}`}
    >
      <div className="relative isolate overflow-hidden">
        <div
          className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${ambientGradientClass}`}
        />

        <header className="relative z-10">
          <div className="max-w-screen-2xl 2xl:max-w-[1800px] mx-auto flex flex-col gap-8 px-4 py-12 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
            <div className="space-y-5 max-w-2xl">
              <span className={sectionLabelClass}>
                C++ Machine Learning Playground
              </span>
              <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">
                Shape your dataset, train two models, and compare their
                predictions in one view.
              </h1>
              <p
                className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}
              >
                Use the composer to sculpt input data, launch regression or
                neural experiments, and watch the visual canvas respond in real
                time.
              </p>
              <div className="flex flex-wrap gap-3">
                <span className={pillClass}>
                  <span
                    className={`h-2 w-2 rounded-full ${datasetReady ? "bg-emerald-400" : "bg-amber-400"}`}
                  />
                  {datasetReady
                    ? `${dataPointCount} data points ready`
                    : "Awaiting a valid dataset"}
                </span>
                <span className={pillClass}>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  {lossHistory.length > 0
                    ? `${lossHistory.length} loss samples streamed`
                    : "No training stream yet"}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3 self-start lg:self-center">
              <button
                type="button"
                onClick={toggleTheme}
                className={`rounded-full border border-transparent p-3 transition focus:outline-none focus:ring-2 focus:ring-offset-0 ${isDark ? "bg-slate-900/60 hover:bg-slate-900/40 focus:ring-indigo-400/60" : "bg-white/80 hover:bg-white focus:ring-indigo-500/40 shadow-lg shadow-indigo-200/30"}`}
                aria-label="Toggle theme"
              >
                {activeTheme === "light" ? (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                    />
                  </svg>
                ) : (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                    />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </header>

        <main className="relative z-10 max-w-screen-2xl 2xl:max-w-[1800px] mx-auto space-y-10 px-4 pb-16 sm:px-6 lg:px-8">
          {(lrError || nnError || wsError) && (
            <div className={`${panelClass} p-6 sm:p-7`} role="alert">
              <div
                className={`rounded-2xl border px-4 py-3 ${isDark ? "border-rose-500/30 bg-rose-500/10 text-rose-100" : "border-rose-200 bg-rose-50 text-rose-800"}`}
              >
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div className="flex items-start gap-3">
                    <span className="mt-1 shrink-0">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    </span>
                    <div className="space-y-1 text-sm leading-relaxed">
                      {lrError && (
                        <p>
                          <span className="font-semibold">
                            Linear Regression:
                          </span>{" "}
                          {lrError}
                        </p>
                      )}
                      {nnError && (
                        <p>
                          <span className="font-semibold">Neural Network:</span>{" "}
                          {nnError}
                        </p>
                      )}
                      {wsError && (
                        <p>
                          <span className="font-semibold">WebSocket:</span>{" "}
                          {wsError}
                        </p>
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setLrError(null);
                      setNnError(null);
                      setWsError(null);
                    }}
                    className={`${outlineButtonClass} whitespace-nowrap`}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="grid min-w-0 grid-cols-1 gap-8 items-start xl:grid-cols-[minmax(0,400px)_minmax(0,1fr)] 2xl:grid-cols-[minmax(0,480px)_minmax(0,1fr)]">
            <div className="space-y-8 xl:flex xl:flex-col xl:space-y-6">
              <section className={`${panelClass} p-7 space-y-6`}>
                <div className="space-y-4">
                  <header className="space-y-3">
                    <span className={sectionLabelClass}>Data Composer</span>
                    <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">
                      Handcraft or randomize points
                    </h2>
                    <p
                      className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}
                    >
                      Supply comma-separated X and Y values or spin up a
                      synthetic sample to explore model behaviour.
                    </p>
                  </header>

                  <div
                    className="flex flex-wrap gap-2"
                    aria-label="Dataset presets"
                  >
                    {[
                      ["line", "Line sample"],
                      ["curve", "Curve sample"],
                      ["wave", "Wave sample"],
                    ].map(([kind, label]) => (
                      <button
                        key={kind}
                        type="button"
                        className={outlineButtonClass}
                        disabled={busy}
                        onClick={() => loadSample(kind)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                      <label className={fieldLabelClass} htmlFor="x-values">
                        X series (comma separated)
                      </label>
                      <textarea
                        id="x-values"
                        className={textareaClassName}
                        value={xInput}
                        onChange={(e) => setXInput(e.target.value)}
                        disabled={busy}
                        placeholder="Ex: 1, 2, 3, 4, 5"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className={fieldLabelClass} htmlFor="y-values">
                        Y series (comma separated)
                      </label>
                      <textarea
                        id="y-values"
                        className={textareaClassName}
                        value={yInput}
                        onChange={(e) => setYInput(e.target.value)}
                        disabled={busy}
                        placeholder="Ex: 2, 4, 4.5, 5, 6"
                      />
                    </div>
                  </div>
                </div>

                <div
                  className={`grid gap-5 rounded-2xl border p-5 ${isDark ? "border-slate-800/80 bg-slate-900/50" : "border-slate-200 bg-slate-50/80"}`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className={`text-sm font-semibold ${mutedTextClass}`}>
                        Random data sculptor
                      </p>
                      <p className={`text-xs sm:text-sm ${mutedTextClass}`}>
                        Generate a sample with adjustable noise to kick-start
                        exploration.
                      </p>
                    </div>
                    <span className={pillClass}>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-4 w-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 19V6l-2 2m8-2v13l2-2"
                        />
                      </svg>
                      {linearityFactor.toFixed(2)} linearity
                    </span>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                      <label className={fieldLabelClass} htmlFor="num-points">
                        Number of points
                      </label>
                      <input
                        id="num-points"
                        type="number"
                        className={inputClassName}
                        value={numRandomPoints}
                        onChange={(e) => setNumRandomPoints(e.target.value)}
                        disabled={busy || !connected}
                        min="2"
                        max="200"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className={fieldLabelClass} htmlFor="linearity">
                        Linearity
                      </label>
                      <input
                        id="linearity"
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={linearityFactor}
                        onChange={(e) =>
                          setLinearityFactor(parseFloat(e.target.value))
                        }
                        disabled={busy || !connected}
                        className={`w-full ${sliderTrackClass}`}
                      />
                    </div>
                  </div>

                  <button
                    type="button"
                    className={`${secondaryButtonClass} w-full`}
                    onClick={handleGenerateRandomData}
                    disabled={busy || !connected}
                  >
                    Generate dataset
                  </button>
                </div>
              </section>
              <section className={`${panelClass} p-7 space-y-6`}>
                <div className="flex flex-col gap-4">
                  <div className="space-y-3">
                    <span className={sectionLabelClass}>Model Lab</span>
                    <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">
                      Configure, train, and inspect
                    </h2>
                    <p
                      className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}
                    >
                      Toggle between linear regression and neural workflows to
                      adjust inputs and view results in real time.
                    </p>
                  </div>
                  <div className={`${metricCardClass} space-y-3`}>
                    <p className={`text-sm ${mutedTextClass}`}>
                      Train separately below, or compare both on the same 80/20
                      split. Errors use original Y units.
                    </p>
                    <button
                      type="button"
                      className={`${secondaryButtonClass} w-full`}
                      disabled={busy || !connected}
                      onClick={handleCompare}
                    >
                      {busy && trainingMode === "both"
                        ? "Comparing models…"
                        : "Compare both models"}
                    </button>
                    <p className={`text-xs ${mutedTextClass}`} role="status">
                      {connected
                        ? busy
                          ? "Training in progress"
                          : "Connected · ready to train"
                        : "Connecting to the training server…"}
                    </p>
                    {busy && (
                      <button
                        type="button"
                        className={`${outlineButtonClass} w-full`}
                        disabled={!run.id}
                        onClick={cancel}
                      >
                        Cancel training
                      </button>
                    )}
                    {run.status === "cancelled" && (
                      <p role="status">
                        Training cancelled. Your completed models are still
                        available.
                      </p>
                    )}
                    {datasetReady && parsedX.length < 5 && (
                      <p className={`text-xs ${mutedTextClass}`}>
                        Fewer than 5 points: training only. Add points to get
                        held-out test scores.
                      </p>
                    )}
                  </div>
                  <div
                    className={`inline-flex w-fit rounded-full border p-1 text-xs sm:text-sm font-semibold ${isDark ? "border-slate-700 bg-slate-900/50" : "border-slate-200 bg-slate-100/80"}`}
                  >
                    <button
                      type="button"
                      onClick={() => setActiveModelPanel("lr")}
                      className={`rounded-full px-4 py-2 transition ${activeModelPanel === "lr" ? (isDark ? "bg-indigo-500/90 text-white shadow-lg shadow-indigo-900/40" : "bg-indigo-600 text-white shadow-lg shadow-indigo-300/40") : isDark ? "text-slate-300 hover:text-white" : "text-slate-500 hover:text-slate-700"}`}
                    >
                      Linear Regression
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveModelPanel("nn");
                        setBlueprintVisibility("visible");
                      }}
                      className={`rounded-full px-4 py-2 transition ${activeModelPanel === "nn" ? (isDark ? "bg-indigo-500/90 text-white shadow-lg shadow-indigo-900/40" : "bg-indigo-600 text-white shadow-lg shadow-indigo-300/40") : isDark ? "text-slate-300 hover:text-white" : "text-slate-500 hover:text-slate-700"}`}
                    >
                      Neural Network
                    </button>
                  </div>
                </div>

                {activeModelPanel === "lr" ? (
                  <div className="space-y-6">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div className="space-y-1">
                        <span className={sectionLabelClass}>
                          Stage 1 · Linear Regression
                        </span>
                        <h3 className="text-xl sm:text-2xl font-semibold tracking-tight">
                          Fit the baseline line
                        </h3>
                      </div>
                      <button
                        type="button"
                        className={`${primaryButtonClass} sm:w-52 ${loadingLRLinear ? "animate-pulse" : ""}`}
                        onClick={handleTrainLR}
                        disabled={busy || !connected}
                      >
                        {loadingLRLinear
                          ? "Training…"
                          : "Train Linear Regression"}
                      </button>
                    </div>

                    <div className="space-y-4">
                      <p
                        className={`text-xs font-semibold uppercase tracking-[0.3em] ${mutedTextClass}`}
                      >
                        Model readout
                      </p>
                      {isLrTrained ? (
                        <div className="grid gap-4 sm:grid-cols-2">
                          <div className={`${metricCardClass} space-y-1`}>
                            <p
                              className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}
                            >
                              Slope (m)
                            </p>
                            <p className="text-base sm:text-xl font-mono text-indigo-400">
                              {formatMetric(lrSlope)}
                            </p>
                          </div>
                          <div className={`${metricCardClass} space-y-1`}>
                            <p
                              className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}
                            >
                              Intercept (b)
                            </p>
                            <p className="text-base sm:text-xl font-mono text-indigo-400">
                              {formatMetric(lrIntercept)}
                            </p>
                          </div>
                          <div className={`${metricCardClass} space-y-1`}>
                            <p
                              className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}
                            >
                              Training MSE
                            </p>
                            <p className="text-base sm:text-xl font-mono text-emerald-400">
                              {formatMetric(lrMse, "mse")}
                            </p>
                          </div>
                          <div className={`${metricCardClass} space-y-1`}>
                            <p
                              className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}
                            >
                              R² &nbsp;|&nbsp; time
                            </p>
                            <p className="text-base sm:text-xl font-mono text-emerald-400">
                              {formatMetric(lrRSquared, "r2")} |{" "}
                              {formatMetric(lrTrainingTime, "time")}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div
                          className={`rounded-2xl border px-4 py-5 text-sm sm:text-base ${isDark ? "border-slate-800/70 bg-slate-900/40" : "border-slate-200 bg-slate-50/80"} ${mutedTextClass}`}
                        >
                          Train the model to populate slope, intercept, and
                          error metrics.
                        </div>
                      )}
                    </div>

                    <div className="space-y-3">
                      <label className={fieldLabelClass} htmlFor="predict-lr">
                        Predict Y for a chosen X
                      </label>
                      <div className="flex flex-col gap-3 sm:flex-row">
                        <input
                          id="predict-lr"
                          type="number"
                          className={`${inputClassName} sm:flex-1`}
                          value={predictXInput}
                          onChange={(e) => setPredictXInput(e.target.value)}
                          disabled={
                            loadingLRPredict || !isLrTrained || loadingNN
                          }
                          placeholder="Enter X"
                        />
                        <button
                          type="button"
                          className={`${accentButtonClass} sm:w-auto`}
                          onClick={handlePredictLR}
                          disabled={
                            loadingLRPredict || !isLrTrained || loadingNN
                          }
                        >
                          {loadingLRPredict ? "Predicting…" : "Predict"}
                        </button>
                      </div>
                      {lrPrediction !== null && lastPredictedXLr !== null && (
                        <p className="text-sm sm:text-base font-medium text-emerald-400">
                          At X = {formatMetric(lastPredictedXLr, "float", 2)},
                          LR predicts {formatMetric(lrPrediction)}
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                      <div className="space-y-1">
                        <span className={sectionLabelClass}>
                          Stage 2 · Neural Network
                        </span>
                        <h3 className="text-xl sm:text-2xl font-semibold tracking-tight">
                          Experiment with a neural stack
                        </h3>
                      </div>
                      <button
                        type="button"
                        className={`${primaryButtonClass} sm:w-56 ${loadingNN ? "animate-pulse" : ""}`}
                        onClick={handleTrainPredictNN}
                        disabled={busy || !connected}
                      >
                        {loadingNN
                          ? "Training neural network…"
                          : "Train NN & Predict"}
                      </button>
                    </div>

                    {(loadingNN || nnResults || lossHistory.length > 0) && (
                      <div
                        className={`rounded-2xl border p-5 ${isDark ? "border-slate-800/80 bg-slate-900/50" : "border-slate-200 bg-slate-50/80"}`}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <p
                            className={`text-xs font-semibold uppercase tracking-[0.3em] ${mutedTextClass}`}
                          >
                            Loss stream
                          </p>
                          <span className={pillClass}>
                            {lossHistory.length || 0} points
                          </span>
                        </div>
                        <div className="mt-4 h-52 sm:h-60">
                          <Line
                            data={lossChartData}
                            options={lossChartOptions}
                          />
                        </div>
                      </div>
                    )}

                    <div className="space-y-6">
                      <div className="grid gap-4 sm:grid-cols-2 sm:items-end">
                        <div className="space-y-2">
                          <label
                            className={fieldLabelClass}
                            htmlFor="layer-sizes"
                          >
                            Layer sizes (e.g., 1-4-1)
                          </label>
                          <input
                            id="layer-sizes"
                            type="text"
                            className={inputClassName}
                            value={nnLayerInput}
                            onChange={(e) => setNnLayerInput(e.target.value)}
                            disabled={busy || !connected}
                            placeholder="Input-Hidden-Output"
                          />
                        </div>
                        <div className="space-y-2">
                          <label
                            className={fieldLabelClass}
                            htmlFor="learning-rate"
                          >
                            Learning rate
                          </label>
                          <input
                            id="learning-rate"
                            type="number"
                            step="0.001"
                            className={inputClassName}
                            value={nnLearningRateInput}
                            onChange={(e) =>
                              setNnLearningRateInput(e.target.value)
                            }
                            disabled={busy || !connected}
                            placeholder="0.01"
                          />
                        </div>
                        <div className="space-y-2 sm:col-span-2">
                          <label className={fieldLabelClass} htmlFor="epochs">
                            Epochs
                          </label>
                          <input
                            id="epochs"
                            type="number"
                            min="1"
                            step="1"
                            className={inputClassName}
                            value={nnEpochsInput}
                            onChange={(e) => setNnEpochsInput(e.target.value)}
                            disabled={busy || !connected}
                            placeholder="1000"
                          />
                        </div>
                      </div>

                      {nnResults && !loadingNN && (
                        <div className="space-y-4">
                          <p
                            className={`text-xs font-semibold uppercase tracking-[0.3em] ${mutedTextClass}`}
                          >
                            Training results
                          </p>
                          <div className="grid gap-4 sm:grid-cols-2">
                            <div className={`${metricCardClass} space-y-1`}>
                              <p
                                className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}
                              >
                                Final training MSE
                              </p>
                              <p className="text-base sm:text-xl font-mono text-emerald-400">
                                {formatMetric(nnResults.finalMse, "mse")}
                              </p>
                            </div>
                            <div className={`${metricCardClass} space-y-1`}>
                              <p
                                className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}
                              >
                                Train time
                              </p>
                              <p className="text-base sm:text-xl font-mono text-emerald-400">
                                {formatMetric(nnResults.trainingTimeMs, "time")}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {nnResults && !loadingNN && (
                        <div className="space-y-3">
                          <label
                            className={fieldLabelClass}
                            htmlFor="predict-nn"
                          >
                            Predict Y for a chosen X
                          </label>
                          <div className="flex flex-col gap-3 sm:flex-row">
                            <input
                              id="predict-nn"
                              type="number"
                              className={`${inputClassName} sm:flex-1`}
                              value={predictXInputNN}
                              onChange={(e) =>
                                setPredictXInputNN(e.target.value)
                              }
                              disabled={loadingNNPredict}
                              placeholder="Enter X"
                            />
                            <button
                              type="button"
                              className={`${accentButtonClass} sm:w-auto`}
                              onClick={handlePredictNN}
                              disabled={loadingNNPredict || !nnResults}
                            >
                              {loadingNNPredict ? "Predicting…" : "Predict"}
                            </button>
                          </div>
                          {nnPrediction !== null &&
                            lastPredictedXNN !== null && (
                              <p className="text-sm sm:text-base font-medium text-emerald-400">
                                At X ={" "}
                                {formatMetric(lastPredictedXNN, "float", 2)}, NN
                                predicts {formatMetric(nnPrediction)}
                              </p>
                            )}
                        </div>
                      )}
                    </div>

                    <div className="space-y-3">
                      <label
                        className={fieldLabelClass}
                        htmlFor="network-blueprint-visibility"
                      >
                        Network blueprint
                      </label>
                      <div className="relative">
                        <select
                          id="network-blueprint-visibility"
                          className={selectClassName}
                          value={blueprintVisibility}
                          onChange={(e) =>
                            setBlueprintVisibility(e.target.value)
                          }
                        >
                          <option value="hidden">Hide blueprint</option>
                          <option value="visible">Show blueprint</option>
                        </select>
                        <span
                          className={`pointer-events-none absolute inset-y-0 right-4 flex items-center ${selectArrowClass}`}
                          aria-hidden="true"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={2}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M6 9l6 6 6-6"
                            />
                          </svg>
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </section>
            </div>
            <div className="space-y-8">
              <section className={`${panelClass} p-7 space-y-6`}>
                <header className="space-y-3">
                  <span className={sectionLabelClass}>Data View</span>
                  <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">
                    Raw distribution
                  </h2>
                  <p
                    className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}
                  >
                    Explore your dataset without model overlays to understand
                    its natural structure.
                  </p>
                </header>
                <div
                  className={`relative rounded-3xl border p-4 shadow-inner ${isDark ? "border-slate-800/70 bg-slate-950/40" : "border-slate-200 bg-white"}`}
                >
                  <div className="h-[24rem] sm:h-[26rem] lg:h-[28rem]">
                    <Scatter
                      data={{ datasets: dataOnlyDatasets }}
                      options={dataOnlyChartOptions}
                    />
                  </div>
                </div>
              </section>

              <section className={`${panelClass} p-7 space-y-6`}>
                <header className="space-y-3">
                  <span className={sectionLabelClass}>Model Overlay</span>
                  <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">
                    Compare predictions with truth
                  </h2>
                  <p
                    className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}
                  >
                    Visualize regression and neural outputs alongside your
                    original data.
                  </p>
                </header>
                <div
                  className={`relative rounded-3xl border p-4 shadow-inner ${isDark ? "border-slate-800/70 bg-slate-950/40" : "border-slate-200 bg-white"}`}
                >
                  <div className="h-[26rem] sm:h-[28rem] lg:h-[32rem]">
                    <Scatter
                      data={{ datasets: memoizedDatasets }}
                      options={overlayChartOptions}
                    />
                  </div>
                </div>
                <div
                  className={`grid gap-3 rounded-2xl border p-5 text-xs sm:text-sm leading-relaxed ${isDark ? "border-slate-800/70 bg-slate-900/50 text-slate-300" : "border-slate-200 bg-slate-50 text-slate-600"}`}
                >
                  <div
                    className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-[0.7rem] font-semibold uppercase tracking-[0.3em] ${isDark ? "bg-indigo-500/20 text-indigo-200" : "bg-indigo-100 text-indigo-700"}`}
                  >
                    Legend
                  </div>
                  <p>
                    • <span className="font-semibold">Red circles</span>{" "}
                    represent original data points.
                  </p>
                  <p>
                    • <span className="font-semibold">Azure line</span> is the
                    linear regression fit.
                  </p>
                  <p>
                    • <span className="font-semibold">Green cross</span> is the
                    latest LR prediction.
                  </p>
                  {nnResults && (
                    <p>
                      • <span className="font-semibold">Orange trail</span>{" "}
                      follows neural predictions across X.
                    </p>
                  )}
                  {nnPrediction !== null && (
                    <p>
                      • <span className="font-semibold">Purple square</span>{" "}
                      marks the NN prediction for your chosen X.
                    </p>
                  )}
                </div>
              </section>

              {(lrModel || nnModel) && (
                <Results
                  result={comparisonResult}
                  className={`${panelClass} p-7`}
                />
              )}
              {lrModel && nnModel && (
                <Predict
                  key={`${lrModel.id}-${nnModel.id}`}
                  predict={predictBoth}
                  prediction={comparisonPrediction}
                  onPrediction={setComparisonPrediction}
                  className={`${panelClass} p-7`}
                />
              )}

              {blueprintVisibility === "visible" && (
                <section className={`${panelClass} p-7 space-y-6`}>
                  <header className="space-y-3">
                    <span className={sectionLabelClass}>Network Blueprint</span>
                    <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">
                      Layer topology preview
                    </h2>
                    <p
                      className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}
                    >
                      Inspect how inputs, hidden units, and outputs connect for
                      the current architecture.
                    </p>
                  </header>
                  <div
                    className={`relative rounded-3xl border p-4 shadow-inner ${isDark ? "border-slate-800/70 bg-slate-950/40" : "border-slate-200 bg-white"}`}
                  >
                    <Suspense
                      fallback={<p role="status">Loading network blueprint…</p>}
                    >
                      <NNVisualizer
                        layerStructure={nnLayerInput}
                        onChange={setNnLayerInput}
                        disabled={busy}
                        isDark={isDark}
                        height={420}
                      />
                    </Suspense>
                  </div>
                </section>
              )}
            </div>
          </div>
        </main>

        <footer
          className={`${ribbonClass} relative z-10 mx-auto mt-6 w-full max-w-screen-2xl 2xl:max-w-[1800px] rounded-3xl px-6 py-4 text-center text-xs sm:text-sm`}
        >
          C++ Linear Regression &amp; Neural Network Playground · Built with
          React, Chart.js, and Node.js
        </footer>
      </div>
    </div>
  );
}

export default App;
