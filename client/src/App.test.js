import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

jest.mock("react-chartjs-2", () => {
  const React = require("react");
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
    this.close = jest.fn(() => {
      this.readyState = MockWebSocket.CLOSED;
      if (this.onclose) {
        this.onclose({ target: this });
      }
    });
    this.send = jest.fn();
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
      value: jest.fn(() => ({
        canvas: document.createElement("canvas"),
        fillRect: jest.fn(),
        clearRect: jest.fn(),
        getImageData: jest.fn(() => ({ data: [] })),
        putImageData: jest.fn(),
        createImageData: jest.fn(() => []),
        setTransform: jest.fn(),
        drawImage: jest.fn(),
        save: jest.fn(),
        restore: jest.fn(),
        beginPath: jest.fn(),
        closePath: jest.fn(),
        moveTo: jest.fn(),
        lineTo: jest.fn(),
        clip: jest.fn(),
        stroke: jest.fn(),
        translate: jest.fn(),
        scale: jest.fn(),
        rotate: jest.fn(),
        arc: jest.fn(),
        fill: jest.fn(),
        measureText: jest.fn(() => ({ width: 0 })),
        transform: jest.fn(),
        rect: jest.fn(),
        fillText: jest.fn(),
        strokeText: jest.fn(),
        createLinearGradient: jest.fn(() => ({ addColorStop: jest.fn() })),
      })),
      configurable: true,
    });
  }
});

beforeEach(() => {
  global.WebSocket = MockWebSocket;
  MockWebSocket.instances = [];
  fetchMock = jest.fn();
  global.fetch = fetchMock;
});

afterEach(() => {
  jest.clearAllMocks();
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
