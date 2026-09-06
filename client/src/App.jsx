import { useMemo, useState } from "react";
import useComparison from "./hooks/useComparison";
import { sampleData, parseData, format } from "./data";
import { ComparisonChart, LossChart } from "./components/ComparisonChart";
import Results from "./components/Results";
import Predict from "./components/Predict";
import "./App.css";

export default function App() {
  const [preset, setPreset] = useState("curve");
  const [text, setText] = useState(() => sampleData());
  const [noise, setNoise] = useState(0.15);
  const [count, setCount] = useState(60);
  const [layers, setLayers] = useState("1-8-1");
  const [rate, setRate] = useState("0.3");
  const [epochs, setEpochs] = useState("3000");
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const { run, connected, start, cancel, reset, predict } = useComparison();
  const busy = run.status === "running";
  const parsed = useMemo(() => {
    try {
      return { data: parseData(text) };
    } catch (failure) {
      return { error: failure.message, data: { x: [], y: [] } };
    }
  }, [text]);
  const lastLoss = run.loss.at(-1);
  const clear = () => {
    reset();
    setError(null);
    setPrediction(null);
  };
  const generate = (kind, size, amount) => {
    clear();
    setText(sampleData(kind, size, amount));
  };
  const submit = (event) => {
    event.preventDefault();
    setError(null);
    setPrediction(null);
    if (parsed.error) {
      setError(parsed.error);
      return;
    }
    if (!/^[1-9]\d*$/.test(epochs)) {
      setError("Epochs must be a positive whole number.");
      return;
    }
    start({
      ...parsed.data,
      layers,
      learningRate: Number(rate),
      epochs: Number(epochs),
    });
  };
  return (
    <div className="app-shell">
      <header className="header">
        <a className="brand" href="/" aria-label="ML playground home">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <path d="M5 25V7M5 25h23M8 21l6-7 5 3 8-11" />
          </svg>
          ML playground
        </a>
        <span
          className={`connection ${connected ? "online" : ""}`}
          role="status"
        >
          <i />
          {connected ? (busy ? "Training…" : "Ready to train") : "Connecting…"}
        </span>
      </header>
      <main>
        <div className="intro">
          <span className="eyebrow">LINEAR REGRESSION / NEURAL NETWORK</span>
          <h1>Two models. One fair comparison.</h1>
          <p>
            Choose a dataset, train both models, and see how well they predict
            points they haven’t seen.
          </p>
        </div>
        <div className="workspace">
          <aside>
            <form className="panel controls" onSubmit={submit}>
              <fieldset disabled={busy}>
                <legend>
                  <span className="step">1</span> Choose your data
                </legend>
                <label htmlFor="dataset">Dataset</label>
                <select
                  id="dataset"
                  value={preset}
                  onChange={(e) => {
                    setPreset(e.target.value);
                    generate(e.target.value, count, noise);
                  }}
                >
                  <option value="curve">Curve · nonlinear</option>
                  <option value="line">Line · linear</option>
                  <option value="wave">Wave · periodic</option>
                </select>
                <div className="range-label">
                  <label htmlFor="points">Points</label>
                  <output htmlFor="points">{count}</output>
                </div>
                <input
                  id="points"
                  type="range"
                  min="20"
                  max="200"
                  step="10"
                  value={count}
                  onChange={(e) => {
                    const value = Number(e.target.value);
                    setCount(value);
                    generate(preset, value, noise);
                  }}
                />
                <div className="range-label">
                  <label htmlFor="noise">Noise</label>
                  <output htmlFor="noise">{Math.round(noise * 100)}%</output>
                </div>
                <input
                  id="noise"
                  type="range"
                  min="0"
                  max="0.8"
                  step="0.05"
                  value={noise}
                  onChange={(e) => {
                    const value = Number(e.target.value);
                    setNoise(value);
                    generate(preset, count, value);
                  }}
                />
                <details>
                  <summary>Enter your own data</summary>
                  <p className="muted">10–1,000 pairs, separated by commas.</p>
                  <label htmlFor="x-data">X values</label>
                  <textarea
                    id="x-data"
                    rows="3"
                    value={text.x}
                    onChange={(e) => {
                      clear();
                      setText({ ...text, x: e.target.value });
                    }}
                  />
                  <label htmlFor="y-data">Y values</label>
                  <textarea
                    id="y-data"
                    rows="3"
                    value={text.y}
                    onChange={(e) => {
                      clear();
                      setText({ ...text, y: e.target.value });
                    }}
                  />
                </details>
              </fieldset>
              <div className="divider" />
              <fieldset disabled={busy}>
                <legend>
                  <span className="step">2</span> Set up the models
                </legend>
                <div className="model-explainer">
                  <span className="model-line lr" />
                  <div>
                    <strong>Linear regression</strong>
                    <p>A best-fit straight line. No tuning needed.</p>
                  </div>
                </div>
                <div className="model-explainer">
                  <span className="model-line nn" />
                  <div>
                    <strong>Neural network</strong>
                    <p>
                      Learns curved relationships through repeated training.
                    </p>
                  </div>
                </div>
                <details>
                  <summary>Neural network settings</summary>
                  <label htmlFor="layers">Layer sizes</label>
                  <input
                    id="layers"
                    value={layers}
                    onChange={(e) => {
                      clear();
                      setLayers(e.target.value);
                    }}
                    aria-describedby="layers-help"
                  />
                  <small id="layers-help">
                    1-8-1 means 1 input, 8 hidden neurons, 1 output. Up to 4
                    hidden layers of 32 neurons.
                  </small>
                  <div className="field-pair">
                    <div>
                      <label htmlFor="learning-rate">Learning rate</label>
                      <input
                        id="learning-rate"
                        type="number"
                        min="0.000001"
                        max="1"
                        step="any"
                        value={rate}
                        onChange={(e) => {
                          clear();
                          setRate(e.target.value);
                        }}
                      />
                    </div>
                    <div>
                      <label htmlFor="epochs">Epochs</label>
                      <input
                        id="epochs"
                        type="number"
                        min="1"
                        max="10000"
                        step="1"
                        value={epochs}
                        onChange={(e) => {
                          clear();
                          setEpochs(e.target.value);
                        }}
                      />
                    </div>
                  </div>
                  <small>
                    Rate controls each learning step. An epoch is one pass
                    through the training data.
                  </small>
                </details>
              </fieldset>
              <div className="split-note">
                <strong>80% to learn · 20% to test</strong>
                <p>
                  Both models use the same split. Fixed seeds make runs
                  repeatable.
                </p>
              </div>
              <button
                className="primary"
                type="submit"
                disabled={busy || !connected}
              >
                {busy
                  ? "Training models…"
                  : run.result
                    ? "Compare again"
                    : "Compare models"}
                <span aria-hidden="true">{busy ? "" : "→"}</span>
              </button>
              {busy && (
                <button
                  type="button"
                  className="cancel"
                  disabled={!run.id}
                  onClick={cancel}
                >
                  Cancel training
                </button>
              )}
              {(error || run.error) && (
                <p className="error" role="alert">
                  {error || run.error}
                </p>
              )}
              {run.status === "cancelled" && (
                <p className="notice" role="status">
                  Training cancelled. Adjust your settings and try again.
                </p>
              )}
            </form>
          </aside>
          <div className="output">
            <section className="panel plot-panel" aria-labelledby="plot-title">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">THE FIT</span>
                  <h2 id="plot-title">
                    {run.result
                      ? "Predictions meet the data"
                      : "Start with the data"}
                  </h2>
                </div>
                <span className="tag">{parsed.data.x.length} points</span>
              </div>
              <p className="muted">
                {run.result
                  ? "Lines are model outputs. Diamonds are points neither model trained on."
                  : "Train both models to overlay their predictions and compare test error."}
              </p>
              <ComparisonChart
                data={parsed.data}
                result={run.result}
                prediction={prediction}
              />
              {parsed.error && <p className="error">{parsed.error}</p>}
            </section>
            {busy && (
              <section
                className="panel training"
                aria-labelledby="training-title"
              >
                <div className="section-heading">
                  <h2 id="training-title">
                    {busy ? "Learning in progress" : "Training history"}
                  </h2>
                  <span className="tag">
                    {lastLoss
                      ? `${lastLoss.epoch} / ${epochs} epochs`
                      : "Starting…"}
                  </span>
                </div>
                {busy && (
                  <progress
                    aria-label="Training progress"
                    value={lastLoss?.epoch || 0}
                    max={Number(epochs)}
                  />
                )}
                <p className="muted" role="status">
                  {lastLoss
                    ? `Neural network training MSE: ${format(lastLoss.mse)}`
                    : "Fitting the baseline and starting the neural network."}
                </p>
                <details open={busy}>
                  <summary>View loss over time</summary>
                  <LossChart loss={run.loss} />
                </details>
              </section>
            )}
            {run.result && (
              <>
                <Results result={run.result} />
                <Predict
                  key={run.id}
                  predict={predict}
                  prediction={prediction}
                  onPrediction={setPrediction}
                />
                <section className="panel history">
                  <details>
                    <summary>Inspect neural network training history</summary>
                    <p className="muted">
                      Training error over {epochs} epochs, in original Y units
                      squared.
                    </p>
                    <LossChart loss={run.loss} />
                  </details>
                </section>
              </>
            )}
            {!run.result && !busy && (
              <div className="empty-state">
                <span className="step">3</span>
                <p>
                  <strong>Your comparison will appear here.</strong>
                  <br />
                  Look for lower test error, not just a close fit to the
                  training data.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
      <footer>
        Models run in C++ · Results stay in memory for up to 30 minutes while
        connected.
      </footer>
    </div>
  );
}
