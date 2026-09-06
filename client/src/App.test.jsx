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

vi.mock("react-chartjs-2", () => ({
  Scatter: () => <div>Fit chart</div>,
  Line: () => <div>Loss chart</div>,
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
    train: Array.from({ length: 48 }, (_, i) => ({ x: i, y: i })),
    test: Array.from({ length: 12 }, (_, i) => ({ x: i, y: i })),
  },
  lr: {
    train: { mse: 0.3 },
    test: { mse: 0.5, r2: 0.7 },
    timeMs: 1,
    curve: [],
  },
  nn: {
    train: { mse: 0.1 },
    test: { mse: 0.2, r2: 0.9 },
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
async function start() {
  fireEvent.click(screen.getByRole("button", { name: /Compare models/ }));
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
}
function complete() {
  act(() =>
    socket.emit({
      id: "job",
      type: "completed",
      status: "completed",
      result,
      loss: [{ epoch: 1500, mse: 0.1 }],
    }),
  );
}
beforeEach(() => {
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

test("trains, compares test metrics, and requests actual model predictions", async () => {
  boot();
  await start();
  expect(JSON.parse(fetch.mock.calls[0][1].body).x).toHaveLength(60);
  expect(fetch.mock.calls[0][1].headers.Authorization).toBe("Bearer owner");
  act(() => socket.emit({ id: "job", type: "progress", epoch: 20, mse: 2.5 }));
  expect(
    screen.getByText("Neural network training MSE: 2.5"),
  ).toBeInTheDocument();
  complete();
  const table = screen.getByRole("table");
  expect(within(table).getByText("0.5")).toBeInTheDocument();
  expect(within(table).getByText("0.2")).toBeInTheDocument();
  expect(
    screen.getByText(/The neural network has lower test error/),
  ).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("X value"), {
    target: { value: "12.137" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Predict both" }));
  expect(await screen.findByText("3.1")).toBeInTheDocument();
  expect(fetch.mock.calls.at(-1)[0]).toBe("/api/jobs/job/predict");
  expect(JSON.parse(fetch.mock.calls.at(-1)[1].body)).toEqual({ x: 12.137 });
  expect(screen.getByText(/outside your dataset/)).toBeInTheDocument();
});

test("cancels training and ignores late loss events", async () => {
  boot();
  await start();
  fireEvent.click(screen.getByRole("button", { name: "Cancel training" }));
  expect(await screen.findByText(/Training cancelled/)).toBeInTheDocument();
  act(() => socket.emit({ id: "job", type: "progress", epoch: 100, mse: 8 }));
  expect(screen.queryByText(/MSE: 8/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Compare models/ })).toBeEnabled();
});

test("shows failures and invalidates results when the dataset changes", async () => {
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
  expect(screen.getByRole("alert")).toHaveTextContent("Training timed out.");
  fetch.mockClear();
  await start();
  complete();
  fireEvent.change(screen.getByLabelText("Dataset"), {
    target: { value: "line" },
  });
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Predict both" }),
  ).not.toBeInTheDocument();
});

test("rejects bad manual data before requesting training", () => {
  boot();
  fireEvent.change(screen.getByLabelText("X values"), {
    target: { value: "1,2,,3" },
  });
  fireEvent.click(screen.getByRole("button", { name: /Compare models/ }));
  expect(screen.getByRole("alert")).toHaveTextContent(/matching X and Y/);
  expect(fetch).not.toHaveBeenCalled();
});

test("disconnection invalidates models and disables training", async () => {
  boot();
  await start();
  complete();
  act(() => socket.onclose());
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Compare models/ })).toBeDisabled();
  expect(screen.getByRole("alert")).toHaveTextContent(/Connection lost/);
});

test("ignores other jobs and stale snapshots after completion", () => {
  const current = { id: "a", status: "completed", result, loss: [] };
  expect(mergeUpdate(current, { id: "b", status: "failed" })).toBe(current);
  expect(mergeUpdate(current, { id: "a", status: "running", loss: [] })).toBe(
    current,
  );
});

test("manual parsing preserves unsorted pairs and rejects missing or non-finite values", () => {
  const data = { x: "9,1,8,2,7,3,6,4,5,0", y: "9,1,8,2,7,3,6,4,5,0" };
  expect(parseData(data).x[0]).toBe(9);
  expect(() => parseData({ ...data, x: "9,1,8,2,7,,6,4,5,0" })).toThrow(
    /finite/,
  );
  expect(() => parseData({ ...data, x: "9,1,8,2,7,Infinity,6,4,5,0" })).toThrow(
    /finite/,
  );
});
