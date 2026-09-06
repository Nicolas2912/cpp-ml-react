import { format } from "../data";

export default function Results({ result, className = "" }) {
  const { lr, nn, split } = result;
  const comparable = lr?.test.mse != null && nn?.test.mse != null;
  const difference = comparable ? lr.test.mse - nn.test.mse : 0;
  const tied =
    comparable &&
    Math.abs(difference) <=
      Math.max(1e-10, Math.max(lr.test.mse, nn.test.mse) * 0.001);
  const winner = comparable && !tied ? (difference > 0 ? "nn" : "lr") : null;
  const rows = [
    [
      "Test MSE",
      "Lower is better · original Y units²",
      (model) => format(model.test.mse),
    ],
    [
      "Training MSE",
      "Fit on the points used to learn",
      (model) => format(model.train.mse),
    ],
    [
      "Test R²",
      "1 is best · 0 matches the test mean",
      (model) => format(model.test.r2),
    ],
    [
      "Compute time",
      "Training + engine evaluation",
      (model) => `${format(model.timeMs)} ms`,
    ],
  ];
  return (
    <section
      className={`comparison-card ${className}`}
      aria-labelledby="results-title"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2
          id="results-title"
          className="text-2xl font-semibold tracking-tight"
        >
          Compare the results
        </h2>
        <span className="comparison-tag">Same dataset · same split</span>
      </div>
      <p className="comparison-muted">
        {split.test.length
          ? `${split.train.length} training points · ${split.test.length} held-out test points. Both models use the same split.`
          : "Training-only run. Add at least 5 points to get held-out test scores."}
      </p>
      <div className="comparison-table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col" className="comparison-lr">
                Linear regression
              </th>
              <th scope="col" className="comparison-nn">
                Neural network
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, hint, get], index) => (
              <tr key={name} className={index === 0 ? "primary-metric" : ""}>
                <th scope="row">
                  {name}
                  <small>{hint}</small>
                </th>
                <td className={index === 0 && winner === "lr" ? "best" : ""}>
                  {lr ? get(lr) : "Not trained"}
                </td>
                <td className={index === 0 && winner === "nn" ? "best" : ""}>
                  {nn ? get(nn) : "Not trained"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="comparison-muted">
        {!lr || !nn
          ? "Train the other model or choose Compare both models to complete the comparison."
          : !comparable
            ? "Test scores need held-out data."
            : tied
              ? "The models have similar test error on this split."
              : `${winner === "lr" ? "Linear regression" : "The neural network"} has lower test error on this split.`}{" "}
        MSE is measured in original Y units squared.
      </p>
      {split.test.length > 0 &&
        (lr?.test.r2 === null || nn?.test.r2 === null) && (
          <p className="comparison-muted">
            R² is undefined with fewer than two test points or constant test
            targets. More data gives a more useful comparison.
          </p>
        )}
    </section>
  );
}
