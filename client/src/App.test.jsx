import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App from "./App";
import { mergeUpdate } from "./hooks/useComparison";
import { parseData } from "./data";
import { parseArchitecture, generateFlowElements } from "./NNVisualizer";

vi.mock("react-chartjs-2", () => ({
  Scatter: () => <div>Scatter chart</div>,
  Line: () => <div>Loss chart</div>,
}));
vi.mock("reactflow", () => ({
  default: ({ nodes, onNodeClick, children }) => (
    <div>
      {nodes.map((node) => (
        <button key={node.id} onClick={() => onNodeClick({}, node)}>
          {node.data.label}
        </button>
      ))}
      {children}
    </div>
  ),
  Controls: () => <span>Zoom controls</span>,
  Background: () => null,
  Position: { Right: "right", Left: "left" },
}));
class Socket {
  static instances = [];
  constructor() {
    Socket.instances.push(this);
  }
  close() {}
  emit(event) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
}
const result = {
  split: {
    train: [
      { x: 1, y: 2 },
      { x: 2, y: 3 },
      { x: 3, y: 4 },
      { x: 4, y: 5 },
    ],
    test: [{ x: 5, y: 5 }],
  },
  lr: {
    slope: 1,
    intercept: 1,
    train: { mse: 0.3, r2: 0.8 },
    test: { mse: 0.5, r2: null },
    timeMs: 1,
    curve: [],
  },
  nn: {
    train: { mse: 0.1, r2: 0.9 },
    test: { mse: 0.2, r2: null },
    timeMs: 20,
    curve: [],
  },
};
let socket;
const json = (body) => ({ ok: true, json: async () => body });
function boot() {
  render(<App />);
  socket = Socket.instances.at(-1);
  act(() => socket.emit({ type: "session", token: "owner" }));
}
async function start(name = "Compare both models") {
  fetch.mockClear();
  fireEvent.click(screen.getByRole("button", { name, exact: true }));
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
}
async function complete(mode = "both") {
  act(() =>
    socket.emit({
      id: "job",
      type: "completed",
      status: "completed",
      result: {
        ...result,
        lr: mode === "nn" ? null : result.lr,
        nn: mode === "lr" ? null : result.nn,
      },
      loss: mode === "lr" ? [] : [{ epoch: 1000, mse: 0.1 }],
    }),
  );
  await screen.findByRole("table");
}
beforeEach(() => {
  const storage = new Map();
  vi.stubGlobal("localStorage", {
    getItem: (key) => storage.get(key) || null,
    setItem: (key, value) => storage.set(key, value),
  });
  Socket.instances = [];
  vi.stubGlobal("WebSocket", Socket);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url, options) => {
      if (url === "/api/jobs") return json({ id: "job" });
      if (url.endsWith("/predict"))
        return json({
          x: JSON.parse(options.body).x,
          lr: 4.2,
          nn: 3.1,
          extrapolation: true,
        });
      if (url.endsWith("/cancel"))
        return json({ id: "job", status: "cancelled", loss: [] });
      return json({ id: "job", status: "running", loss: [] });
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

test("preserves the original composer, model tabs, dual charts, and theme toggle", () => {
  boot();
  expect(screen.getByLabelText("X series (comma separated)")).toHaveValue(
    "1, 2, 3, 4, 5",
  );
  expect(screen.getByLabelText("Number of points")).toHaveValue(20);
  expect(screen.getByLabelText("Linearity")).toHaveValue("0.7");
  expect(
    screen.getByRole("button", { name: "Generate dataset" }),
  ).toBeEnabled();
  expect(screen.getByText("Raw distribution")).toBeInTheDocument();
  expect(
    screen.getByText("Compare predictions with truth"),
  ).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Toggle theme" }));
  expect(document.querySelector("[data-theme]")).toHaveAttribute(
    "data-theme",
    "dark",
  );
  expect(localStorage.getItem("ml-theme")).toBe("dark");
});

test("trains both models, compares metrics, and sends true prediction requests", async () => {
  boot();
  await start();
  expect(JSON.parse(fetch.mock.calls[0][1].body).model).toBe("both");
  expect(fetch.mock.calls[0][1].headers.Authorization).toBe("Bearer owner");
  await complete();
  const table = screen.getByRole("table");
  expect(within(table).getByText("0.5")).toBeInTheDocument();
  expect(within(table).getByText("0.2")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("X value"), {
    target: { value: "12.137" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Predict both" }));
  expect(
    await screen.findByText(/Predicted Y at X = 12.14/),
  ).toBeInTheDocument();
  const predictions = fetch.mock.calls.filter(([url]) =>
    url.endsWith("/predict"),
  );
  expect(
    predictions.map(([, options]) => JSON.parse(options.body).model),
  ).toEqual(["lr", "nn"]);
  expect(screen.getByText(/outside your dataset/)).toBeInTheDocument();
});

test("keeps LR results when training NN separately and restores both prediction controls", async () => {
  boot();
  await start("Train Linear Regression");
  expect(JSON.parse(fetch.mock.calls[0][1].body).model).toBe("lr");
  await complete("lr");
  expect(screen.getByText("Slope (m)")).toBeInTheDocument();
  fireEvent.click(
    screen.getByRole("button", { name: "Neural Network", exact: true }),
  );
  await start("Train NN & Predict");
  expect(JSON.parse(fetch.mock.calls[0][1].body).model).toBe("nn");
  await complete("nn");
  expect(
    within(screen.getByRole("table")).getByText("0.5"),
  ).toBeInTheDocument();
  expect(screen.getByLabelText("Predict Y for a chosen X")).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Predict", exact: true }));
  await screen.findByText(/NN predicts 3.1/);
  expect(JSON.parse(fetch.mock.calls.at(-1)[1].body).model).toBe("nn");
  fireEvent.click(
    screen.getByRole("button", { name: "Linear Regression", exact: true }),
  );
  expect(screen.getByLabelText("Predict Y for a chosen X")).toBeEnabled();
});

test("edits layers and neurons in the restored blueprint and trains that architecture", async () => {
  boot();
  fireEvent.click(
    screen.getByRole("button", { name: "Neural Network", exact: true }),
  );
  expect(await screen.findByText("Layer topology preview")).toBeInTheDocument();
  fireEvent.change(await screen.findByLabelText("Hidden layer 1 neurons"), {
    target: { value: "6" },
  });
  expect(screen.getByLabelText(/Layer sizes/)).toHaveValue("1-6-1");
  fireEvent.click(screen.getByRole("button", { name: "Add hidden layer" }));
  expect(screen.getByLabelText(/Layer sizes/)).toHaveValue("1-6-4-1");
  fireEvent.click(screen.getByRole("button", { name: "L1 · Neuron 1" }));
  expect(screen.getByText(/L1 · Neuron 1 selected/)).toBeInTheDocument();
  await start("Train NN & Predict");
  expect(JSON.parse(fetch.mock.calls[0][1].body).layers).toBe("1-6-4-1");
  expect(
    screen.getByRole("button", { name: "Add hidden layer" }),
  ).toBeDisabled();
});

test("cancels a run and rejects late progress", async () => {
  boot();
  await start();
  fireEvent.click(screen.getByRole("button", { name: "Cancel training" }));
  await screen.findByText(/Training cancelled/);
  act(() => socket.emit({ id: "job", type: "progress", epoch: 100, mse: 8 }));
  expect(
    screen.getByRole("button", { name: "Compare both models" }),
  ).toBeEnabled();
});

test("surfaces failures, invalid input, and clears stale models on data changes", async () => {
  boot();
  await start();
  act(() =>
    socket.emit({
      id: "job",
      type: "failed",
      status: "failed",
      error: "Training timed out.",
    }),
  );
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Training timed out.",
  );
  await start();
  await complete();
  fireEvent.click(screen.getByRole("button", { name: "Curve sample" }));
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("X series (comma separated)"), {
    target: { value: "1,,3" },
  });
  fetch.mockClear();
  fireEvent.click(screen.getByRole("button", { name: "Compare both models" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    /matching X and Y/,
  );
  expect(fetch).not.toHaveBeenCalled();
});

test("disconnection invalidates models and disables training", async () => {
  boot();
  await start();
  await complete();
  act(() => socket.onclose());
  await waitFor(() =>
    expect(screen.queryByRole("table")).not.toBeInTheDocument(),
  );
  expect(
    screen.getByRole("button", { name: "Compare both models" }),
  ).toBeDisabled();
  expect(await screen.findByRole("alert")).toHaveTextContent(/Connection lost/);
});

test("ignores other jobs and old snapshots; safely parses architectures and small datasets", () => {
  const current = { id: "a", status: "completed", result, loss: [] };
  expect(mergeUpdate(current, { id: "b", status: "failed" })).toBe(current);
  expect(mergeUpdate(current, { id: "a", status: "running", loss: [] })).toBe(
    current,
  );
  expect(parseData({ x: "2,1", y: "4,2" }).x).toEqual([2, 1]);
  expect(() => parseData({ x: "1,,3", y: "1,2,3" })).toThrow(/finite/);
  expect(parseArchitecture("1-999999-1")).toBeNull();
  expect(parseArchitecture("1-0-1")).toBeNull();
  const graph = generateFlowElements([1, 4, 3, 1], null, false);
  expect(graph.nodes).toHaveLength(9);
  expect(graph.edges).toHaveLength(19);
});
