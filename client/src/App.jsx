// --- START OF FILE App.js ---

import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LogarithmicScale, // <-- Import LogarithmicScale
} from "chart.js";
import { Scatter, Line } from "react-chartjs-2";
import throttle from "lodash.throttle"; // <-- Import throttle instead of debounce
import "./App.css";
import NNVisualizer from "./NNVisualizer";

// Register necessary components
ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LogarithmicScale, // <-- Register LogarithmicScale
);

const API_URL = "http://localhost:3001/api"; // Base API URL
const WS_URL = "ws://localhost:3001"; // WebSocket URL

function App() {
  // ... (keep existing state variables) ...
  const [xInput, setXInput] = useState("1, 2, 3, 4, 5");
  const [yInput, setYInput] = useState("2, 4, 5, 4, 5");
  const [predictXInput, setPredictXInput] = useState("6"); // For LR prediction
  const [predictXInputNN, setPredictXInputNN] = useState("6"); // For NN prediction
  const [activeTheme, setActiveTheme] = useState("light");
  const [numRandomPoints, setNumRandomPoints] = useState(20);
  const [linearityFactor, setLinearityFactor] = useState(0.7);

  // --- Linear Regression Model/Result states ---
  const [lrSlope, setLrSlope] = useState(null);
  const [lrIntercept, setLrIntercept] = useState(null);
  const [lrPrediction, setLrPrediction] = useState(null);
  const [lastPredictedXLr, setLastPredictedXLr] = useState(null);
  const [isLrTrained, setIsLrTrained] = useState(false);
  const [lrTrainingTime, setLrTrainingTime] = useState(null);
  const [lrMse, setLrMse] = useState(null);
  const [lrRSquared, setLrRSquared] = useState(null);

  // --- Neural Network Model/Result states ---
  const [nnLayerInput, setNnLayerInput] = useState("1-4-1"); // Default NN structure
  const [nnLearningRateInput, setNnLearningRateInput] = useState(0.01);
  const [nnEpochsInput, setNnEpochsInput] = useState(1000);
  const [nnResults, setNnResults] = useState(null); // Stores { trainingTimeMs, finalMse, predictions }
  const [nnError, setNnError] = useState(null);
  const [loadingNN, setLoadingNN] = useState(false);
  const [lossHistory, setLossHistory] = useState([]); // Stores { epoch, mse } pairs
  const [wsError, setWsError] = useState(null); // For WebSocket errors
  const [nnPrediction, setNnPrediction] = useState(null); // Stores prediction for a specific x value
  const [lastPredictedXNN, setLastPredictedXNN] = useState(null); // Stores the last x value used for prediction
  const [loadingNNPredict, setLoadingNNPredict] = useState(false); // Loading state for NN prediction

  // UI states
  const [loadingLRLinear, setLoadingLRLinear] = useState(false); // Renamed
  const [loadingLRPredict, setLoadingLRPredict] = useState(false); // Renamed
  const [lrError, setLrError] = useState(null); // Renamed

  const isDark = activeTheme === "dark";

  const chartRef = useRef(null);
  const lossChartRef = useRef(null); // Ref for the loss chart

  // --- Ref to store batched loss updates ---
  const lossUpdateBatchRef = useRef([]); // <--- ADDED

  // --- Debounced function to process loss updates ---
  const scheduleProcessLossBatch = useMemo(() => {
    const processLossBatch = () => {
      const batch = lossUpdateBatchRef.current;
      if (batch.length === 0) {
        return; // Nothing to process
      }

      setLossHistory((prev) => {
        // Filter out duplicates/older points from the batch relative to the current state
        const latestEpochInState =
          prev.length > 0 ? prev[prev.length - 1].epoch : -1;
        const validNewPoints = batch.filter(
          (p) => p.epoch > latestEpochInState,
        );
        // Sort just in case messages arrived out of order within the batch
        validNewPoints.sort((a, b) => a.epoch - b.epoch);
        return [...prev, ...validNewPoints];
      });

      lossUpdateBatchRef.current = []; // Clear the batch after processing

      // --- Manually trigger chart update ---
      if (lossChartRef.current) {
        lossChartRef.current.update("none"); // Update without animation
      }
    };
    // Throttle the processing function: run at most once per 150ms
    return throttle(processLossBatch, 150, { leading: true, trailing: true });
  }, []); // Empty dependency array ensures the function is stable
  // --- REMOVED old debouncedSetLossHistory and lossUpdateQueue ---

  // --- WebSocket Connection ---
  const ws = useRef(null);

  useEffect(() => {
    // Function to clean up WebSocket and related resources
    const cleanupWs = () => {
      // Ensure we cancel pending throttled calls
      if (
        scheduleProcessLossBatch &&
        typeof scheduleProcessLossBatch.cancel === "function"
      ) {
        scheduleProcessLossBatch.cancel();
        console.log("Cleanup: Throttled batch processing cancelled.");
      }

      lossUpdateBatchRef.current = []; // Clear batch on cleanup
      console.log("Cleanup: Batch ref cleared.");

      // Check if ws.current exists and has a close method before calling
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        console.log("Closing WebSocket connection.");
        ws.current.close();
      }
      ws.current = null; // Clear ref on unmount/cleanup
    };

    // Create WebSocket connection
    if (!ws.current) {
      console.log("Creating WebSocket connection to:", WS_URL);
      ws.current = new WebSocket(WS_URL);
    }

    // WebSocket setup
    ws.current.onopen = () => {
      console.log("WebSocket connected");
      setWsError(null);
    };

    ws.current.onclose = () => {
      console.log("WebSocket disconnected");
      // Clear the ref when closed
      ws.current = null;
    };

    ws.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      setWsError("WebSocket connection error. Is the server running?");
      // Clear the ref on error
      ws.current = null;
    };

    ws.current.onmessage = (event) => {
      // --- ADDED LOG TO SEE RAW MESSAGE ---
      console.log("WebSocket message received:", event.data);
      // --- END LOG ---
      try {
        const message = JSON.parse(event.data);
        switch (message.type) {
          case "loss_update":
            // Add the new point to the batch ref
            lossUpdateBatchRef.current.push({
              epoch: message.epoch,
              mse: message.mse,
            });
            // Schedule the throttled processing function
            scheduleProcessLossBatch(); // <-- MODIFIED
            break;
          case "final_result":
            // Flush any pending throttled updates before processing final results
            scheduleProcessLossBatch.flush(); // <-- MODIFIED
            lossUpdateBatchRef.current = []; // Clear batch ref <-- ADDED
            setNnResults({
              trainingTimeMs: message.trainingTimeMs,
              finalMse: message.finalMse,
              predictions: message.predictions,
            });
            setLoadingNN(false);
            setNnError(null);
            setWsError(null);
            break;
          case "error":
            // Flush and clear queue on error too
            scheduleProcessLossBatch.flush(); // <-- MODIFIED
            lossUpdateBatchRef.current = []; // Clear batch ref <-- ADDED
            setNnError(`Training failed: ${message.message}`);
            setLoadingNN(false);
            break;
          default:
            console.warn("Unknown WebSocket message type:", message.type);
        }
      } catch (err) {
        console.error("Failed to parse WebSocket message:", event.data, err);
      }
    };

    // Cleanup function
    return () => {
      cleanupWs();
    };
    // Include throttled function in dependency array as it's used in the effect's cleanup
  }, [scheduleProcessLossBatch]); // <-- MODIFIED Dependency

  // --- Helper functions ---
  const parseInputString = (input) => {
    // ... (keep existing function) ...
    if (!input || typeof input !== "string") return [];
    return input
      .split(",")
      .map((val) => parseFloat(val.trim()))
      .filter((val) => !isNaN(val));
  };

  // --- Memoize parsed input data ---
  const parsedX = useMemo(() => parseInputString(xInput), [xInput]);
  const parsedY = useMemo(() => parseInputString(yInput), [yInput]);

  // --- Calculate Axis Bounds (useMemo) ---
  const axisBounds = useMemo(() => {
    // ... (keep existing logic) ...
    const xDataValues = parsedX;
    let yDataValues = parsedY; // Start with original Y

    // Include NN predictions in Y data if available
    if (nnResults?.predictions) {
      yDataValues = yDataValues.concat(
        nnResults.predictions.filter((p) => !isNaN(p)),
      );
    }

    let minXVal = xDataValues.length > 0 ? Math.min(...xDataValues) : 0;
    let maxXVal = xDataValues.length > 0 ? Math.max(...xDataValues) : 10;
    let minYVal = yDataValues.length > 0 ? Math.min(...yDataValues) : 0;
    let maxYVal = yDataValues.length > 0 ? Math.max(...yDataValues) : 10;

    // Include LR prediction point in range calculation
    if (lastPredictedXLr !== null && !isNaN(lastPredictedXLr)) {
      minXVal = Math.min(minXVal, lastPredictedXLr);
      maxXVal = Math.max(maxXVal, lastPredictedXLr);
    }
    if (lrPrediction !== null && !isNaN(lrPrediction)) {
      minYVal = Math.min(minYVal, lrPrediction);
      maxYVal = Math.max(maxYVal, lrPrediction);
    }

    // Include NN prediction point in range calculation
    if (lastPredictedXNN !== null && !isNaN(lastPredictedXNN)) {
      minXVal = Math.min(minXVal, lastPredictedXNN);
      maxXVal = Math.max(maxXVal, lastPredictedXNN);
    }
    if (nnPrediction !== null && !isNaN(nnPrediction)) {
      minYVal = Math.min(minYVal, nnPrediction);
      maxYVal = Math.max(maxYVal, nnPrediction);
    }

    // Handle single point case or flat data
    if (minXVal === maxXVal) {
      maxXVal += 1;
      minXVal -= 1;
    }
    if (minYVal === maxYVal) {
      maxYVal += 1;
      minYVal -= 1;
    }
    if (maxXVal < minXVal) [minXVal, maxXVal] = [maxXVal, minXVal]; // Swap if needed
    if (maxYVal < minYVal) [minYVal, maxYVal] = [maxYVal, minYVal]; // Swap if needed

    // Calculate padding (ensure padding isn't overly large for small ranges)
    const xRange = Math.max(maxXVal - minXVal, 0.1); // Prevent zero range
    const yRange = Math.max(maxYVal - minYVal, 0.1);
    const xPadding = Math.max(xRange * 0.1, 0.5); // Min padding of 0.5
    const yPadding = Math.max(yRange * 0.1, 0.5);

    // Apply padding
    return {
      minX: minXVal - xPadding,
      maxX: maxXVal + xPadding,
      minY: minYVal - yPadding,
      maxY: maxYVal + yPadding,
    };
  }, [
    parsedX,
    parsedY,
    lastPredictedXLr,
    lrPrediction,
    nnResults,
    lastPredictedXNN,
    nnPrediction,
  ]);

  // --- Memoize the generated Chart Datasets ---
  const memoizedDatasets = useMemo(() => {
    // ... (keep existing logic) ...
    const xValues = parsedX;
    const yValues = parsedY;
    const currentLrSlope = lrSlope;
    const currentLrIntercept = lrIntercept;
    const predictionXLr = lastPredictedXLr;
    const predictionYLr = lrPrediction;
    const predictionXNN = lastPredictedXNN;
    const predictionYNN = nnPrediction;
    const nnPreds = nnResults?.predictions; // Get NN predictions array
    const bounds = axisBounds; // Use calculated bounds

    const dataPoints = xValues.map((x, index) => ({ x: x, y: yValues[index] }));

    const datasets = [
      {
        label: "Data Points (Y)",
        data: dataPoints,
        backgroundColor: "rgba(255, 99, 132, 1)", // Red
        pointRadius: 5,
        order: 3, // Draw data points on top
      },
    ];

    // Add Linear Regression line
    if (currentLrSlope !== null && currentLrIntercept !== null && bounds) {
      const lineStartX = bounds.minX;
      const lineEndX = bounds.maxX;
      datasets.push({
        label: "Linear Regression",
        data: [
          {
            x: lineStartX,
            y: currentLrSlope * lineStartX + currentLrIntercept,
          },
          { x: lineEndX, y: currentLrSlope * lineEndX + currentLrIntercept },
        ],
        borderColor: "rgba(54, 162, 235, 1)", // Blue
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        tension: 0,
        showLine: true,
        type: "line",
        order: 1, // Draw LR line behind predictions
      });
    }

    // Add Linear Regression prediction visualization
    if (
      predictionXLr !== null &&
      predictionYLr !== null &&
      !isNaN(predictionXLr) &&
      !isNaN(predictionYLr) &&
      bounds
    ) {
      const lrPredictionStroke = isDark
        ? "rgba(255, 255, 255, 1)" // White X in dark mode
        : "rgba(0, 0, 0, 1)"; // Black X in light mode
      datasets.push({
        label: "LR Prediction",
        data: [{ x: predictionXLr, y: predictionYLr }],
        backgroundColor: "rgba(0, 255, 0, 1)", // Keep green accent for visibility
        borderColor: lrPredictionStroke, // X color adapts to theme
        borderWidth: 2, // Added for thicker lines
        pointRadius: 8, // Keep size
        pointStyle: "crossRot",
        showLine: false,
        order: 4, // Draw LR prediction on top
      });
      // Add guide lines
      datasets.push({
        label: "LR Prediction Guide X", // Label for tooltip filtering
        data: [
          { x: predictionXLr, y: bounds.minY }, // Start from bottom axis
          { x: predictionXLr, y: predictionYLr },
        ],
        borderColor: "rgba(128, 128, 128, 0.7)", // Grey guide line
        borderWidth: 1,
        borderDash: [5, 5], // Dashed line
        pointRadius: 0,
        fill: false,
        tension: 0,
        showLine: true,
        type: "line",
        order: 0, // Draw behind everything
      });
      datasets.push({
        label: "LR Prediction Guide Y", // Label for tooltip filtering
        data: [
          { x: bounds.minX, y: predictionYLr }, // Start from left axis
          { x: predictionXLr, y: predictionYLr },
        ],
        borderColor: "rgba(128, 128, 128, 0.7)", // Grey guide line
        borderWidth: 1,
        borderDash: [5, 5], // Dashed line
        pointRadius: 0,
        fill: false,
        tension: 0,
        showLine: true,
        type: "line",
        order: 0, // Draw behind everything
      });
    }

    // Add Neural Network Predictions
    if (nnPreds && nnPreds.length === xValues.length && bounds) {
      // --- Sort the predictions by x-value before plotting ---
      const nnPredictionPoints = xValues
        .map((x, index) => ({ x: x, y: nnPreds[index] }))
        .sort((a, b) => a.x - b.x); // Sort by x for correct line plotting

      // --- DEBUGGING LOGS ---
      console.log("NN Preds Data:", nnPreds);
      console.log("NN Prediction Points (sorted):", nnPredictionPoints);
      console.log("Rendering NN Prediction Dataset with showLine: true");
      // --- END DEBUGGING LOGS ---

      datasets.push({
        label: "Neural Network Predictions",
        data: nnPredictionPoints,
        backgroundColor: "rgba(255, 159, 64, 1)", // Orange points
        pointRadius: 6, // Slightly larger
        pointStyle: "triangle", // Different shape
        // --- Changes for Interpolation Line ---
        showLine: true, // ENABLE the line
        borderColor: "rgba(255, 159, 64, 0.7)", // Use a slightly transparent orange for the line
        borderWidth: 2, // Define line thickness
        tension: 0.4, // ADD smoothness (adjust 0.1 to 0.5 as needed)
        type: "line", // Explicitly set type to 'line'
        // --- End Changes ---
        order: 2, // Draw NN points/line below data but above LR line
      });
    }

    // Add Neural Network prediction visualization (similar to LR prediction)
    if (
      predictionXNN !== null &&
      predictionYNN !== null &&
      !isNaN(predictionXNN) &&
      !isNaN(predictionYNN) &&
      bounds
    ) {
      datasets.push({
        label: "NN Prediction",
        data: [{ x: predictionXNN, y: predictionYNN }],
        backgroundColor: "rgba(153, 102, 255, 1)", // Purple
        borderColor: "rgba(0, 0, 0, 1)",
        borderWidth: 2,
        pointRadius: 8,
        pointStyle: "rect", // Different shape from LR
        showLine: false,
        order: 4, // Draw NN prediction on top
      });
      // Add guide lines
      datasets.push({
        label: "NN Prediction Guide X",
        data: [
          { x: predictionXNN, y: bounds.minY },
          { x: predictionXNN, y: predictionYNN },
        ],
        borderColor: "rgba(153, 102, 255, 0.7)",
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
        tension: 0,
        showLine: true,
        type: "line",
        order: 0,
      });
      datasets.push({
        label: "NN Prediction Guide Y",
        data: [
          { x: bounds.minX, y: predictionYNN },
          { x: predictionXNN, y: predictionYNN },
        ],
        borderColor: "rgba(153, 102, 255, 0.7)",
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
        tension: 0,
        showLine: true,
        type: "line",
        order: 0,
      });
    }

    return datasets;
  }, [
    parsedX,
    parsedY,
    lrSlope,
    lrIntercept,
    lastPredictedXLr,
    lrPrediction,
    axisBounds,
    nnResults,
    lastPredictedXNN,
    nnPrediction,
    isDark,
  ]);

  const dataOnlyDatasets = useMemo(() => {
    const points = parsedX
      .map((x, idx) => {
        const y = parsedY[idx];
        if (
          typeof x === "number" &&
          typeof y === "number" &&
          !Number.isNaN(x) &&
          !Number.isNaN(y)
        ) {
          return { x, y };
        }
        return null;
      })
      .filter(Boolean);

    return [
      {
        label: "Data Points",
        data: points,
        backgroundColor: "rgba(255, 99, 132, 1)",
        pointRadius: 5,
        order: 1,
      },
    ];
  }, [parsedX, parsedY]);

  // --- Theme Toggle Effect ---
  useEffect(() => {
    // ... (keep existing logic) ...
    document.documentElement.setAttribute("data-theme", activeTheme);
  }, [activeTheme]);

  const toggleTheme = () => {
    // ... (keep existing logic) ...
    setActiveTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  // --- Generate Random Data ---
  const handleGenerateRandomData = () => {
    // ... (keep existing logic) ...
    setLrError(null);
    setNnError(null);
    // Reset LR state
    setLrSlope(null);
    setLrIntercept(null);
    setLrPrediction(null);
    setLastPredictedXLr(null);
    setIsLrTrained(false);
    setLrTrainingTime(null);
    setLrMse(null);
    setLrRSquared(null);
    // Reset NN state
    setNnResults(null);
    setLoadingNN(false);
    setLossHistory([]); // Also clear loss history

    const num = parseInt(numRandomPoints, 10);
    if (isNaN(num) || num <= 0) {
      setLrError("Please enter a valid positive number of points to generate.");
      return;
    }

    const randomSlope = (Math.random() - 0.5) * 5;
    const randomIntercept = Math.random() * 10;
    const minX = 0;
    const maxX = 50;
    const minYLinear = randomSlope * minX + randomIntercept;
    const maxYLinear = randomSlope * maxX + randomIntercept;
    const yRangeLinear = Math.abs(maxYLinear - minYLinear);
    const baseNoise = 1.0;
    const maxRelativeNoiseFactor = 0.4;
    let noiseMagnitude = Math.max(
      baseNoise,
      yRangeLinear * maxRelativeNoiseFactor,
    );
    noiseMagnitude = noiseMagnitude * (1 - linearityFactor);

    const randomX = [];
    const randomY = [];
    for (let i = 0; i < num; i++) {
      const x = Math.random() * maxX;
      const y_linear = randomSlope * x + randomIntercept;
      const noise = (Math.random() - 0.5) * 2 * noiseMagnitude;
      const y = y_linear + noise;
      randomX.push(x.toFixed(2));
      randomY.push(y.toFixed(2));
    }

    setXInput(randomX.join(", "));
    setYInput(randomY.join(", "));
  };

  // --- handleTrainLR ---
  const handleTrainLR = async () => {
    // ... (keep existing logic) ...
    setLrError(null);
    setLoadingLRLinear(true);
    // Reset LR state only
    setLrSlope(null);
    setLrIntercept(null);
    setLrPrediction(null);
    setLastPredictedXLr(null);
    setIsLrTrained(false);
    setLrTrainingTime(null);
    setLrMse(null);
    setLrRSquared(null);
    // Don't reset NN state here

    const x_values = parsedX;
    const y_values = parsedY;
    if (
      x_values.length === 0 ||
      y_values.length === 0 ||
      x_values.length !== y_values.length
    ) {
      setLrError(
        "Invalid input: X and Y must be comma-separated numbers and have the same length.",
      );
      setLoadingLRLinear(false);
      return;
    }

    try {
      const response = await fetch(`${API_URL}/lr_train`, {
        // Use renamed endpoint
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x_values, y_values }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `HTTP error! Status: ${response.status}`);
      }

      setLrSlope(data.slope);
      setLrIntercept(data.intercept);
      setLrTrainingTime(data.trainingTimeMs);
      setLrMse(data.mse);
      setLrRSquared(data.r_squared);
      setIsLrTrained(true);
    } catch (err) {
      console.error("LR Training failed:", err);
      setLrError(
        err.message || "An unknown error occurred during LR training.",
      );
    } finally {
      setLoadingLRLinear(false);
    }
  };

  // --- handlePredictLR ---
  const handlePredictLR = async () => {
    console.log("[FRONTEND] handlePredictLR: Function called."); // <-- Log Entry

    if (!isLrTrained) {
      console.warn("[FRONTEND] handlePredictLR: Model not trained. Aborting.");
      setLrError("Train the LR model before predicting.");
      return;
    }

    setLrError(null);
    setLoadingLRPredict(true);
    setLrPrediction(null); // Reset previous prediction display
    console.log(
      "[FRONTEND] handlePredictLR: State reset (loading=true, prediction=null).",
    );

    const x_value_parsed = parseFloat(predictXInput);

    if (isNaN(x_value_parsed)) {
      console.error(
        "[FRONTEND] handlePredictLR: Invalid X value input:",
        predictXInput,
      );
      setLrError("Invalid input: LR Prediction X value must be a number.");
      setLoadingLRPredict(false); // <-- Make sure loading is stopped on error
      return;
    }

    console.log(
      `[FRONTEND] handlePredictLR: Parsed X value: ${x_value_parsed}`,
    );
    console.log(
      `[FRONTEND] handlePredictLR: Attempting fetch to ${API_URL}/lr_predict`,
    );

    try {
      const bodyPayload = JSON.stringify({ x_value: x_value_parsed });
      console.log("[FRONTEND] handlePredictLR: Sending payload:", bodyPayload);

      const response = await fetch(`${API_URL}/lr_predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: bodyPayload,
      });

      console.log(
        `[FRONTEND] handlePredictLR: Fetch response received. Status: ${response.status}, OK: ${response.ok}`,
      );

      if (!response.ok) {
        let errorMsg = `HTTP error! Status: ${response.status}`;
        let errorData = null;
        try {
          // Attempt to read error details from response body
          errorData = await response.json();
          console.log(
            "[FRONTEND] handlePredictLR: Received error data from server:",
            errorData,
          );
          errorMsg = errorData.error || errorMsg; // Use server error if available
        } catch (jsonError) {
          console.warn(
            "[FRONTEND] handlePredictLR: Could not parse error response as JSON.",
            jsonError,
          );
          // Fallback to status text if JSON parsing fails or no body
          errorMsg = `${errorMsg} - ${response.statusText || "Server returned non-JSON error"}`;
        }
        throw new Error(errorMsg); // Throw error to be caught below
      }

      // --- Response seems OK, try parsing JSON ---
      console.log("[FRONTEND] handlePredictLR: Attempting response.json()");
      const data = await response.json();
      console.log(
        "[FRONTEND] handlePredictLR: Successfully parsed JSON data:",
        data,
      );

      // Validate the received data structure
      if (
        data.prediction === undefined ||
        data.prediction === null ||
        isNaN(data.prediction)
      ) {
        console.error(
          "[FRONTEND] handlePredictLR: Invalid data received:",
          data,
        );
        throw new Error("Invalid prediction value received from server.");
      }

      // --- Data is valid, update state ---
      console.log(
        `[FRONTEND] handlePredictLR: Setting prediction state to: ${data.prediction}`,
      );
      setLrPrediction(data.prediction);
      setLastPredictedXLr(x_value_parsed); // Use the parsed value
      console.log("[FRONTEND] handlePredictLR: State update complete.");
    } catch (err) {
      // Catches fetch errors (network, DNS), response.ok=false errors, .json() errors, and validation errors
      console.error("[FRONTEND] handlePredictLR: Catch block executed.", err);
      setLrError(
        err.message || "An unknown error occurred during LR prediction.",
      );
      setLastPredictedXLr(null); // Clear prediction point visualization on error
      console.log("[FRONTEND] handlePredictLR: Error state updated.");
    } finally {
      // --- CRUCIAL: Ensure loading state is always reset ---
      setLoadingLRPredict(false);
      console.log(
        "[FRONTEND] handlePredictLR: Finally block executed. Loading state set to false.",
      );
    }
  };

  // --- handleTrainPredictNN ---
  const handleTrainPredictNN = async () => {
    // Prevent rapid successive clicks
    if (loadingNN) return;

    setNnError(null);
    setWsError(null); // Clear previous WS errors
    setLoadingNN(true);
    setNnResults(null); // Clear previous results
    setLossHistory([]); // Clear previous loss history <--- IMPORTANT

    // Cancel any pending throttled updates
    scheduleProcessLossBatch.cancel();
    lossUpdateBatchRef.current = [];

    const x_values = parsedX;
    const y_values = parsedY;
    const layers = nnLayerInput;
    const learning_rate = parseFloat(nnLearningRateInput);
    const epochs = parseInt(nnEpochsInput, 10);

    // Validation (keep existing)
    if (
      x_values.length === 0 ||
      y_values.length === 0 ||
      x_values.length !== y_values.length
    ) {
      setNnError(
        "Invalid input: X and Y must be non-empty and have the same length.",
      );
      setLoadingNN(false);
      return;
    }
    if (
      !layers ||
      typeof layers !== "string" ||
      !layers.match(/^(\d+-)+\d+$/)
    ) {
      setNnError(
        'Invalid layers format. Use dash-separated numbers (e.g., "1-3-1").',
      );
      setLoadingNN(false);
      return;
    }
    if (isNaN(learning_rate)) {
      setNnError("Learning rate must be a valid number.");
      setLoadingNN(false);
      return;
    }
    if (isNaN(epochs) || epochs <= 0) {
      setNnError("Epochs must be a positive whole number.");
      setLoadingNN(false);
      return;
    }

    // Check WebSocket connection (keep existing)
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setNnError(
        "WebSocket is not connected. Please ensure the server is running and refresh.",
      );
      setLoadingNN(false);
      return;
    }

    try {
      // Make the POST request (keep existing)
      const response = await fetch(`${API_URL}/nn_train_predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x_values,
          y_values,
          layers,
          learning_rate,
          epochs,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `HTTP error! Status: ${response.status}`);
      }
      console.log(
        "NN training initiated via HTTP, waiting for WebSocket updates...",
      );
    } catch (err) {
      console.error("NN Training Initiation failed:", err);
      setNnError(
        err.message || "An unknown error occurred initiating NN training.",
      );
      setLoadingNN(false);
    }
  };

  // --- Add a function to predict with the Neural Network
  const handlePredictNN = async () => {
    if (!nnResults) {
      setNnError("Train the Neural Network model before predicting.");
      return;
    }

    setNnError(null);
    setLoadingNNPredict(true);
    setNnPrediction(null); // Reset previous prediction

    const x_value_parsed = parseFloat(predictXInputNN);

    if (isNaN(x_value_parsed)) {
      setNnError("Invalid input: NN Prediction X value must be a number.");
      setLoadingNNPredict(false);
      return;
    }

    try {
      // We're using interpolation directly with the original data points
      const trainingPredictions = nnResults.predictions;

      // Calculate prediction by interpolation for the new X value
      const index = parsedX.findIndex((x) => x >= x_value_parsed);

      let prediction;
      if (index === -1) {
        // If beyond all training data, use the last prediction
        prediction = trainingPredictions[trainingPredictions.length - 1];
      } else if (index === 0) {
        // If before all training data, use the first prediction
        prediction = trainingPredictions[0];
      } else {
        // Interpolate between two closest predictions
        const x1 = parsedX[index - 1];
        const x2 = parsedX[index];
        const y1 = trainingPredictions[index - 1];
        const y2 = trainingPredictions[index];

        // Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        prediction = y1 + ((x_value_parsed - x1) * (y2 - y1)) / (x2 - x1);
      }

      // Update state with the prediction
      setNnPrediction(prediction);
      setLastPredictedXNN(x_value_parsed);
    } catch (err) {
      console.error("NN Prediction failed:", err);
      setNnError(
        err.message || "An unknown error occurred during NN prediction.",
      );
      setLastPredictedXNN(null);
    } finally {
      setLoadingNNPredict(false);
    }
  };

  // --- Formatters (keep existing) ---
  const formatTime = (timeMs) => {
    if (timeMs === null || isNaN(timeMs) || timeMs < 0) return "N/A";
    if (timeMs < 1) return "< 1 ms";
    if (timeMs < 1000) return `${timeMs.toFixed(0)} ms`;
    return `${(timeMs / 1000).toFixed(2)} s`;
  };

  const formatMetric = (value, type = "float", precision = 4) => {
    // Default precision to 4 if not provided or invalid
    const prec =
      typeof precision === "number" && precision >= 0 ? precision : 4;
    if (value === null || value === undefined || isNaN(value)) return "N/A";

    switch (type) {
      case "mse":
        // Use exponential notation for very small or large numbers
        if (
          Math.abs(value) > 10000 ||
          (Math.abs(value) < 0.0001 && value !== 0)
        ) {
          return value.toExponential(); // Or use .toFixed()
        }
        return value.toFixed(prec);
      case "r2":
        return (value * 100).toFixed(Math.max(0, prec - 2)) + "%"; // Adjust precision for percentage
      case "time": // Added a case for time specifically
        return formatTime(value); // Use the existing formatTime helper
      case "float":
      default:
        return value.toFixed(prec);
    }
  };
  // --- Base Chart Options (useMemo) ---
  const baseChartOptions = useMemo(() => {
    const gridColor = isDark ? "rgba(226, 232, 240, 0.22)" : undefined;
    const tickColor = isDark ? "rgba(226, 232, 240, 0.85)" : undefined;

    const buildAxisOptions = () => ({
      grid: {
        color: gridColor,
        drawBorder: false,
        lineWidth: isDark ? 1 : 0.75,
      },
      ticks: {
        color: tickColor,
      },
      title: {
        color: tickColor,
      },
    });

    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: tickColor,
          },
        },
        tooltip: {
          mode: "nearest",
          intersect: false,
        },
      },
      scales: {
        x: buildAxisOptions(),
        y: buildAxisOptions(),
      },
      interaction: {
        mode: "nearest",
        intersect: false,
      },
    };
  }, [isDark]);

  const dataOnlyChartOptions = useMemo(() => {
    const plugins = baseChartOptions.plugins || {};
    const scales = baseChartOptions.scales || {};
    return {
      ...baseChartOptions,
      plugins: {
        ...plugins,
        legend: { display: false },
      },
      scales: {
        ...scales,
        x: { ...(scales.x || {}), min: axisBounds.minX, max: axisBounds.maxX },
        y: { ...(scales.y || {}), min: axisBounds.minY, max: axisBounds.maxY },
      },
    };
  }, [baseChartOptions, axisBounds]);

  const overlayChartOptions = useMemo(() => {
    const plugins = baseChartOptions.plugins || {};
    const scales = baseChartOptions.scales || {};
    return {
      ...baseChartOptions,
      plugins: {
        ...plugins,
      },
      scales: {
        ...scales,
        x: { ...(scales.x || {}), min: axisBounds.minX, max: axisBounds.maxX },
        y: { ...(scales.y || {}), min: axisBounds.minY, max: axisBounds.maxY },
      },
    };
  }, [baseChartOptions, axisBounds]);

  // --- Loss Chart Data and Options ---
  const lossChartData = useMemo(
    () => ({
      // Use epoch as the direct label for the x-axis
      labels: lossHistory.map((p) => p.epoch),
      datasets: [
        {
          label: "Training Loss (MSE)",
          // Data is just the MSE values
          data: lossHistory.map((p) => p.mse),
          borderColor: "rgb(255, 99, 132)",
          backgroundColor: "rgba(255, 99, 132, 0.5)",
          tension: 0.1, // Slight smoothing
          pointRadius: 1, // Small points
          borderWidth: 1.5, // Slightly thicker line
        },
      ],
    }),
    [lossHistory],
  ); // Depend only on lossHistory

  const lossChartOptions = useMemo(() => {
    const gridColor = isDark ? "rgba(226, 232, 240, 0.22)" : undefined;
    const tickColor = isDark ? "rgba(226, 232, 240, 0.85)" : undefined;

    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        // Add animation configuration
        duration: 50, // Re-enable very short animation (might help ResizeObserver)
      },
      resizeDelay: 100,
      plugins: {
        legend: {
          display: false, // Keep legend hidden
        },
        title: {
          display: true,
          text: "NN Training Loss (MSE) vs Epoch",
          color: tickColor,
        },
        tooltip: {
          mode: "index",
          intersect: false,
        },
      },
      scales: {
        x: {
          // Was CategoryScale, now implicitly handled by labels/data structure
          title: { display: true, text: "Epoch", color: tickColor },
          ticks: {
            autoSkip: true, // Skip some labels if too crowded
            maxTicksLimit: 15, // Limit the number of visible ticks
            color: tickColor,
          },
          grid: {
            color: gridColor,
            drawBorder: false,
            lineWidth: isDark ? 1 : 0.75,
          },
        },
        y: {
          title: { display: true, text: "MSE (Log Scale)", color: tickColor }, // Update title
          type: "logarithmic", // Use log scale
          min: 0.00001, // Adjust min slightly if needed, avoid zero/negative
          grid: {
            color: gridColor,
            drawBorder: false,
            lineWidth: isDark ? 1 : 0.75,
          },
          ticks: {
            color: tickColor,
            // Custom formatter for log scale ticks if needed
            callback: function (value, index, values) {
              // Show only powers of 10 or specific intervals on log scale
              if (
                value === 1 ||
                value === 0.1 ||
                value === 0.01 ||
                value === 0.001 ||
                value === 0.0001 ||
                value === 0.00001
              ) {
                return value.toExponential(); // Or use .toFixed()
              }
              if (Math.log10(value) % 1 === 0) {
                return value.toExponential(); // Show 1e-N labels
              }
              return null; // Hide other labels
            },
          },
        },
      },
    };
  }, [isDark]);

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
    "w-full rounded-2xl border px-4 py-3 text-base shadow-sm transition focus:outline-none focus:ring-2 focus:ring-offset-0 disabled:opacity-60 disabled:cursor-not-allowed";
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

  const dataPointCount = Math.min(parsedX.length, parsedY.length);
  const datasetReady = parsedX.length > 0 && parsedX.length === parsedY.length;
  const [activeModelPanel, setActiveModelPanel] = useState("lr");
  const [blueprintVisibility, setBlueprintVisibility] = useState("hidden");

  useEffect(() => {
    if (activeModelPanel !== "nn") {
      setBlueprintVisibility("hidden");
    }
  }, [activeModelPanel]);

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

          <div className="grid gap-8 items-start xl:grid-cols-[minmax(0,400px)_minmax(0,1fr)] 2xl:grid-cols-[minmax(0,480px)_minmax(0,1fr)]">
            <div className="space-y-8 xl:flex xl:flex-col xl:space-y-6">
              <section className={`${panelClass} p-7 space-y-6`}>
                <div className="space-y-4">
                  <header className="space-y-3">
                    <span className={sectionLabelClass}>Data Composer</span>
                    <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">Handcraft or randomize points</h2>
                    <p className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}>
                      Supply comma-separated X and Y values or spin up a synthetic sample to explore model behaviour.
                    </p>
                  </header>

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
                        disabled={loadingLRLinear || loadingNN}
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
                        disabled={loadingLRLinear || loadingNN}
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
                        disabled={loadingLRLinear || loadingNN}
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
                        disabled={loadingLRLinear || loadingNN}
                        className={`w-full ${sliderTrackClass}`}
                      />
                    </div>
                  </div>

                  <button
                    type="button"
                    className={`${secondaryButtonClass} w-full`}
                    onClick={handleGenerateRandomData}
                    disabled={loadingLRLinear || loadingNN}
                  >
                    Generate dataset
                  </button>
                </div>
              </section>
              <section className={`${panelClass} p-7 space-y-6`}>
                <div className="flex flex-col gap-4">
                  <div className="space-y-3">
                    <span className={sectionLabelClass}>Model Lab</span>
                    <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">Configure, train, and inspect</h2>
                    <p className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}>
                      Toggle between linear regression and neural workflows to adjust inputs and view results in real time.
                    </p>
                  </div>
                  <div
                    className={`inline-flex w-fit rounded-full border p-1 text-xs sm:text-sm font-semibold ${isDark ? 'border-slate-700 bg-slate-900/50' : 'border-slate-200 bg-slate-100/80'}`}
                  >
                    <button
                      type="button"
                      onClick={() => setActiveModelPanel('lr')}
                      className={`rounded-full px-4 py-2 transition ${activeModelPanel === 'lr' ? (isDark ? 'bg-indigo-500/90 text-white shadow-lg shadow-indigo-900/40' : 'bg-indigo-600 text-white shadow-lg shadow-indigo-300/40') : (isDark ? 'text-slate-300 hover:text-white' : 'text-slate-500 hover:text-slate-700')}`}
                    >
                      Linear Regression
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveModelPanel('nn')}
                      className={`rounded-full px-4 py-2 transition ${activeModelPanel === 'nn' ? (isDark ? 'bg-indigo-500/90 text-white shadow-lg shadow-indigo-900/40' : 'bg-indigo-600 text-white shadow-lg shadow-indigo-300/40') : (isDark ? 'text-slate-300 hover:text-white' : 'text-slate-500 hover:text-slate-700')}`}
                    >
                      Neural Network
                    </button>
                  </div>
                </div>

                {activeModelPanel === 'lr' ? (
                  <div className="space-y-6">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div className="space-y-1">
                        <span className={sectionLabelClass}>Stage 1 · Linear Regression</span>
                        <h3 className="text-xl sm:text-2xl font-semibold tracking-tight">Fit the baseline line</h3>
                      </div>
                      <button
                        type="button"
                        className={`${primaryButtonClass} sm:w-52 ${loadingLRLinear ? 'animate-pulse' : ''}`}
                        onClick={handleTrainLR}
                        disabled={loadingLRLinear || loadingNN}
                      >
                        {loadingLRLinear ? 'Training…' : 'Train Linear Regression'}
                      </button>
                    </div>

                    <div className="space-y-4">
                      <p className={`text-xs font-semibold uppercase tracking-[0.3em] ${mutedTextClass}`}>Model readout</p>
                      {isLrTrained ? (
                        <div className="grid gap-4 sm:grid-cols-2">
                          <div className={`${metricCardClass} space-y-1`}>
                            <p className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}>Slope (m)</p>
                            <p className="text-base sm:text-xl font-mono text-indigo-400">{formatMetric(lrSlope)}</p>
                          </div>
                          <div className={`${metricCardClass} space-y-1`}>
                            <p className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}>Intercept (b)</p>
                            <p className="text-base sm:text-xl font-mono text-indigo-400">{formatMetric(lrIntercept)}</p>
                          </div>
                          <div className={`${metricCardClass} space-y-1`}>
                            <p className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}>MSE</p>
                            <p className="text-base sm:text-xl font-mono text-emerald-400">{formatMetric(lrMse, 'mse')}</p>
                          </div>
                          <div className={`${metricCardClass} space-y-1`}>
                            <p className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}>R² &nbsp;|&nbsp; time</p>
                            <p className="text-base sm:text-xl font-mono text-emerald-400">
                              {formatMetric(lrRSquared, 'r2')} | {formatMetric(lrTrainingTime, 'time')}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className={`rounded-2xl border px-4 py-5 text-sm sm:text-base ${isDark ? 'border-slate-800/70 bg-slate-900/40' : 'border-slate-200 bg-slate-50/80'} ${mutedTextClass}`}>
                          Train the model to populate slope, intercept, and error metrics.
                        </div>
                      )}
                    </div>

                    <div className="space-y-3">
                      <label className={fieldLabelClass} htmlFor="predict-lr">Predict Y for a chosen X</label>
                      <div className="flex flex-col gap-3 sm:flex-row">
                        <input
                          id="predict-lr"
                          type="number"
                          className={`${inputClassName} sm:flex-1`}
                          value={predictXInput}
                          onChange={(e) => setPredictXInput(e.target.value)}
                          disabled={loadingLRPredict || !isLrTrained || loadingNN}
                          placeholder="Enter X"
                        />
                        <button
                          type="button"
                          className={`${accentButtonClass} sm:w-auto`}
                          onClick={handlePredictLR}
                          disabled={loadingLRPredict || !isLrTrained || loadingNN}
                        >
                          {loadingLRPredict ? 'Predicting…' : 'Predict'}
                        </button>
                      </div>
                      {lrPrediction !== null && lastPredictedXLr !== null && (
                        <p className="text-sm sm:text-base font-medium text-emerald-400">
                          At X = {formatMetric(lastPredictedXLr, 'float', 2)}, LR predicts {formatMetric(lrPrediction)}
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                      <div className="space-y-1">
                        <span className={sectionLabelClass}>Stage 2 · Neural Network</span>
                        <h3 className="text-xl sm:text-2xl font-semibold tracking-tight">Experiment with a neural stack</h3>
                      </div>
                      <button
                        type="button"
                        className={`${primaryButtonClass} sm:w-56 ${loadingNN ? 'animate-pulse' : ''}`}
                        onClick={handleTrainPredictNN}
                        disabled={loadingNN || loadingLRLinear}
                      >
                        {loadingNN ? 'Training neural network…' : 'Train NN & Predict'}
                      </button>
                    </div>

                    {(loadingNN || nnResults || lossHistory.length > 0) && (
                      <div className={`rounded-2xl border p-5 ${isDark ? 'border-slate-800/80 bg-slate-900/50' : 'border-slate-200 bg-slate-50/80'}`}>
                        <div className="flex items-center justify-between gap-3">
                          <p className={`text-xs font-semibold uppercase tracking-[0.3em] ${mutedTextClass}`}>Loss stream</p>
                          <span className={pillClass}>{lossHistory.length || 0} points</span>
                        </div>
                        <div className="mt-4 h-52 sm:h-60">
                          <Line ref={lossChartRef} data={lossChartData} options={lossChartOptions} />
                        </div>
                      </div>
                    )}

                    <div className="space-y-6">
                      <div className="grid gap-4 sm:grid-cols-2 sm:items-end">
                        <div className="space-y-2">
                          <label className={fieldLabelClass} htmlFor="layer-sizes">Layer sizes (e.g., 1-4-1)</label>
                          <input
                            id="layer-sizes"
                            type="text"
                            className={inputClassName}
                            value={nnLayerInput}
                            onChange={(e) => setNnLayerInput(e.target.value)}
                            disabled={loadingNN || loadingLRLinear}
                            placeholder="Input-Hidden-Output"
                          />
                        </div>
                        <div className="space-y-2">
                          <label className={fieldLabelClass} htmlFor="learning-rate">Learning rate</label>
                          <input
                            id="learning-rate"
                            type="number"
                            step="0.001"
                            className={inputClassName}
                            value={nnLearningRateInput}
                            onChange={(e) => setNnLearningRateInput(e.target.value)}
                            disabled={loadingNN || loadingLRLinear}
                            placeholder="0.01"
                          />
                        </div>
                        <div className="space-y-2 sm:col-span-2">
                          <label className={fieldLabelClass} htmlFor="epochs">Epochs</label>
                          <input
                            id="epochs"
                            type="number"
                            min="1"
                            step="100"
                            className={inputClassName}
                            value={nnEpochsInput}
                            onChange={(e) => setNnEpochsInput(e.target.value)}
                            disabled={loadingNN || loadingLRLinear}
                            placeholder="1000"
                          />
                        </div>
                      </div>

                      {nnResults && !loadingNN && (
                        <div className="space-y-4">
                          <p className={`text-xs font-semibold uppercase tracking-[0.3em] ${mutedTextClass}`}>Training results</p>
                          <div className="grid gap-4 sm:grid-cols-2">
                            <div className={`${metricCardClass} space-y-1`}>
                              <p className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}>Final MSE</p>
                              <p className="text-base sm:text-xl font-mono text-emerald-400">{formatMetric(nnResults.finalMse, 'mse')}</p>
                            </div>
                            <div className={`${metricCardClass} space-y-1`}>
                              <p className={`text-xs sm:text-sm font-medium ${mutedTextClass}`}>Train time</p>
                              <p className="text-base sm:text-xl font-mono text-emerald-400">{formatMetric(nnResults.trainingTimeMs, 'time')}</p>
                            </div>
                          </div>
                        </div>
                      )}

                      {nnResults && !loadingNN && (
                        <div className="space-y-3">
                          <label className={fieldLabelClass} htmlFor="predict-nn">Predict Y for a chosen X</label>
                          <div className="flex flex-col gap-3 sm:flex-row">
                            <input
                              id="predict-nn"
                              type="number"
                              className={`${inputClassName} sm:flex-1`}
                              value={predictXInputNN}
                              onChange={(e) => setPredictXInputNN(e.target.value)}
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
                          {nnPrediction !== null && lastPredictedXNN !== null && (
                            <p className="text-sm sm:text-base font-medium text-emerald-400">
                              At X = {formatMetric(lastPredictedXNN, "float", 2)}, NN predicts {formatMetric(nnPrediction)}
                            </p>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="space-y-3">
                      <label className={fieldLabelClass} htmlFor="network-blueprint-visibility">
                        Network blueprint
                      </label>
                      <div className="relative">
                        <select
                          id="network-blueprint-visibility"
                          className={selectClassName}
                          value={blueprintVisibility}
                          onChange={(e) => setBlueprintVisibility(e.target.value)}
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
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 9l6 6 6-6" />
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
                  <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">Raw distribution</h2>
                  <p className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}>Explore your dataset without model overlays to understand its natural structure.</p>
                </header>
                <div className={`relative rounded-3xl border p-4 shadow-inner ${isDark ? 'border-slate-800/70 bg-slate-950/40' : 'border-slate-200 bg-white'}`}>
                  <div className="h-[24rem] sm:h-[26rem] lg:h-[28rem]">
                    <Scatter data={{ datasets: dataOnlyDatasets }} options={dataOnlyChartOptions} />
                  </div>
                </div>
              </section>

              <section className={`${panelClass} p-7 space-y-6`}>
                <header className="space-y-3">
                  <span className={sectionLabelClass}>Model Overlay</span>
                  <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">Compare predictions with truth</h2>
                  <p className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}>Visualize regression and neural outputs alongside your original data.</p>
                </header>
                <div className={`relative rounded-3xl border p-4 shadow-inner ${isDark ? 'border-slate-800/70 bg-slate-950/40' : 'border-slate-200 bg-white'}`}>
                  <div className="h-[26rem] sm:h-[28rem] lg:h-[32rem]">
                    <Scatter ref={chartRef} data={{ datasets: memoizedDatasets }} options={overlayChartOptions} />
                  </div>
                </div>
                <div className={`grid gap-3 rounded-2xl border p-5 text-xs sm:text-sm leading-relaxed ${isDark ? 'border-slate-800/70 bg-slate-900/50 text-slate-300' : 'border-slate-200 bg-slate-50 text-slate-600'}`}>
                  <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-[0.7rem] font-semibold uppercase tracking-[0.3em] ${isDark ? 'bg-indigo-500/20 text-indigo-200' : 'bg-indigo-100 text-indigo-700'}`}>Legend</div>
                  <p>• <span className="font-semibold">Red circles</span> represent original data points.</p>
                  <p>• <span className="font-semibold">Azure line</span> is the linear regression fit.</p>
                  <p>• <span className="font-semibold">Green cross</span> is the latest LR prediction.</p>
                  {nnResults && <p>• <span className="font-semibold">Orange trail</span> follows neural predictions across X.</p>}
                  {nnPrediction !== null && <p>• <span className="font-semibold">Purple square</span> marks the NN prediction for your chosen X.</p>}
                </div>
              </section>

              {blueprintVisibility === "visible" && (
                <section className={`${panelClass} p-7 space-y-6`}>
                  <header className="space-y-3">
                    <span className={sectionLabelClass}>Network Blueprint</span>
                    <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight">Layer topology preview</h2>
                    <p className={`text-sm sm:text-base leading-relaxed ${mutedTextClass}`}>
                      Inspect how inputs, hidden units, and outputs connect for the current architecture.
                    </p>
                  </header>
                  <div className={`relative rounded-3xl border p-4 shadow-inner ${isDark ? 'border-slate-800/70 bg-slate-950/40' : 'border-slate-200 bg-white'}`}>
                    <div className="h-[26rem] lg:h-[30rem]">
                      <NNVisualizer layerStructure={nnLayerInput} height="100%" />
                    </div>
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
// --- END OF FILE App.js ---
