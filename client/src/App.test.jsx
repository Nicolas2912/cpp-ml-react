import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import App from "./App";

vi.mock("react-chartjs-2", async () => {
  const React = await import("react");
  const MockChart = React.forwardRef((props, ref) => (
    <div ref={ref} data-testid={props["data-testid"] || "chart-mock"} />
  ));
  return {
    Line: MockChart,
    Scatter: MockChart,
  };
});

class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = MockWebSocket.OPEN;
    this.close = vi.fn(() => {
      this.readyState = MockWebSocket.CLOSED;
      if (this.onclose) {
        this.onclose({ target: this });
      }
    });
    this.send = vi.fn();
    this.onopen = null;
    this.onclose = null;
    this.onerror = null;
    this.onmessage = null;
    MockWebSocket.instances.push(this);
  }
}

MockWebSocket.OPEN = 1;
MockWebSocket.CLOSED = 3;
MockWebSocket.instances = [];

const originalWebSocket = global.WebSocket;
let fetchMock;

beforeAll(() => {
  if (!window.HTMLCanvasElement.prototype.getContext) {
    Object.defineProperty(window.HTMLCanvasElement.prototype, "getContext", {
      value: vi.fn(() => ({
        canvas: document.createElement("canvas"),
        fillRect: vi.fn(),
        clearRect: vi.fn(),
        getImageData: vi.fn(() => ({ data: [] })),
        putImageData: vi.fn(),
        createImageData: vi.fn(() => []),
        setTransform: vi.fn(),
        drawImage: vi.fn(),
        save: vi.fn(),
        restore: vi.fn(),
        beginPath: vi.fn(),
        closePath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        clip: vi.fn(),
        stroke: vi.fn(),
        translate: vi.fn(),
        scale: vi.fn(),
        rotate: vi.fn(),
        arc: vi.fn(),
        fill: vi.fn(),
        measureText: vi.fn(() => ({ width: 0 })),
        transform: vi.fn(),
        rect: vi.fn(),
        fillText: vi.fn(),
        strokeText: vi.fn(),
        createLinearGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
      })),
      configurable: true,
    });
  }
});

beforeEach(() => {
  global.WebSocket = MockWebSocket;
  MockWebSocket.instances = [];
  fetchMock = vi.fn();
  global.fetch = fetchMock;
});

afterEach(() => {
  vi.clearAllMocks();
  delete global.fetch;
});

afterAll(() => {
  global.WebSocket = originalWebSocket;
});

describe("App core workflows", () => {
  test("trains the linear regression model and surfaces metrics", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        slope: 1.2,
        intercept: 0.5,
        trainingTimeMs: 12,
        mse: 0.34,
        r_squared: 0.89,
      }),
    });

    render(<App />);

    const trainButton = await screen.findByRole("button", {
      name: /train linear regression/i,
    });

    await userEvent.click(trainButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:3001/api/lr_train",
        expect.objectContaining({ method: "POST" }),
      );
    });

    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeTruthy();
    const payload = JSON.parse(firstCall[1].body);
    expect(payload).toEqual({
      x_values: [1, 2, 3, 4, 5],
      y_values: [2, 4, 5, 4, 5],
    });

    expect(await screen.findByText("Slope (m)")).toBeInTheDocument();
    expect(screen.getByText("1.2000")).toBeInTheDocument();
    expect(screen.getByText("0.5000")).toBeInTheDocument();
    expect(screen.getByText("0.3400")).toBeInTheDocument();
    expect(screen.getByText(/89\.00%/)).toBeInTheDocument();
    expect(screen.getByText(/12 ms/)).toBeInTheDocument();
  });

  test("predicts with the trained linear regression model", async () => {
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          slope: 1.2,
          intercept: 0.5,
          trainingTimeMs: 12,
          mse: 0.34,
          r_squared: 0.89,
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ prediction: 42 }),
      });

    render(<App />);

    const trainButton = await screen.findByRole("button", {
      name: /train linear regression/i,
    });

    await userEvent.click(trainButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:3001/api/lr_train",
        expect.objectContaining({ method: "POST" }),
      );
    });

    const predictButton = await screen.findByRole("button", { name: /^predict$/i });
    await waitFor(() => expect(predictButton).toBeEnabled());

    await userEvent.click(predictButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:3001/api/lr_predict",
        expect.objectContaining({ method: "POST" }),
      );
    });

    const predictCall = fetchMock.mock.calls[1];
    expect(predictCall).toBeTruthy();
    expect(JSON.parse(predictCall[1].body)).toEqual({ x_value: 6 });

    expect(
      await screen.findByText(/At X = 6.00, LR predicts 42.0000/i),
    ).toBeInTheDocument();
  });
});
