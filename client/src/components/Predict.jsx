import { useState } from "react";
import { format } from "../data";

export default function Predict({
  predict,
  prediction,
  onPrediction,
  className = "",
}) {
  const [x, setX] = useState("0");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const submit = async (event) => {
    event.preventDefault();
    setError(null);
    onPrediction(null);
    const value = Number(x);
    if (!x.trim() || !Number.isFinite(value) || Math.abs(value) > 1e6) {
      setError("Enter an X between −1,000,000 and 1,000,000.");
      return;
    }
    setBusy(true);
    try {
      onPrediction(await predict(value));
    } catch (failure) {
      setError(failure.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section
      className={`comparison-card ${className}`}
      aria-labelledby="predict-title"
    >
      <h2 id="predict-title">Try a new input</h2>
      <p className="comparison-muted">Evaluate both trained models at any X.</p>
      <form onSubmit={submit} className="predict-form">
        <label htmlFor="prediction-x">X value</label>
        <input
          id="prediction-x"
          type="number"
          step="any"
          value={x}
          onChange={(e) => setX(e.target.value)}
          disabled={busy}
        />
        <button disabled={busy} type="submit">
          {busy ? "Predicting…" : "Predict both"}
        </button>
      </form>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {prediction && (
        <div aria-live="polite">
          <div className="prediction-values">
            <p>
              <span className="comparison-lr">Linear regression</span>
              <strong>{format(prediction.lr)}</strong>
            </p>
            <p>
              <span className="comparison-nn">Neural network</span>
              <strong>{format(prediction.nn)}</strong>
            </p>
          </div>
          <p className="comparison-muted">
            Predicted Y at X = {format(prediction.x)}.
          </p>
          {prediction.extrapolation && (
            <p className="notice">
              This X is outside your dataset. Extrapolated predictions may be
              unreliable.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
