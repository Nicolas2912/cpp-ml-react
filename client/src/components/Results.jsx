import { format } from "../data";

export default function Results({ result }) {
  const { lr, nn } = result;
  const difference = lr.test.mse - nn.test.mse;
  const tied =
    Math.abs(difference) <=
    Math.max(1e-10, Math.max(lr.test.mse, nn.test.mse) * 0.001);
  const winner = tied ? null : difference > 0 ? "nn" : "lr";
  const rows = [
    ["Test MSE", "Lower is better", (model) => format(model.test.mse)],
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
    <section className="panel results" aria-labelledby="results-title">
      <div className="section-heading">
        <h2 id="results-title">Compare the results</h2>
        <span className="tag">Same test points</span>
      </div>
      <p className="muted">
        Both models learned from {result.split.train.length} points. These
        scores use {result.split.test.length} held-out points.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col" className="lr">
                Linear regression
              </th>
              <th scope="col" className="nn">
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
                  {get(lr)}
                </td>
                <td className={index === 0 && winner === "nn" ? "best" : ""}>
                  {get(nn)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="result-note">
        {tied
          ? "The models have similar test error on this split."
          : `${winner === "lr" ? "Linear regression" : "The neural network"} has lower test error on this split.`}{" "}
        MSE is measured in original Y units squared.
      </p>
      {(lr.test.r2 === null || nn.test.r2 === null) && (
        <p className="muted">R² is undefined when every test Y is equal.</p>
      )}
    </section>
  );
}
