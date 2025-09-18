// --- START OF FILE App.js ---

import React, { useState, useEffect, useRef, useMemo } from 'react';
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
} from 'chart.js';
import { Scatter, Line } from 'react-chartjs-2';
import throttle from 'lodash.throttle'; // <-- Import throttle instead of debounce
import './App.css';
import NNVisualizer from './NNVisualizer';

// Register necessary components
ChartJS.register(
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    Title,
    CategoryScale,
    LogarithmicScale // <-- Register LogarithmicScale
);

const API_URL = 'http://localhost:3001/api'; // Base API URL
const WS_URL = 'ws://localhost:3001'; // WebSocket URL

function App() {
  // ... (keep existing state variables) ...
  const [xInput, setXInput] = useState('1, 2, 3, 4, 5');
  const [yInput, setYInput] = useState('2, 4, 5, 4, 5');
  const [predictXInput, setPredictXInput] = useState('6'); // For LR prediction
  const [predictXInputNN, setPredictXInputNN] = useState('6'); // For NN prediction
  const [activeTheme, setActiveTheme] = useState('light');
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
  const [nnLayerInput, setNnLayerInput] = useState('1-4-1'); // Default NN structure
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
 
          setLossHistory(prev => {
              // Filter out duplicates/older points from the batch relative to the current state
              const latestEpochInState = prev.length > 0 ? prev[prev.length - 1].epoch : -1;
              const validNewPoints = batch.filter(p => p.epoch > latestEpochInState);
              // Sort just in case messages arrived out of order within the batch
              validNewPoints.sort((a, b) => a.epoch - b.epoch);
              return [...prev, ...validNewPoints];
          });

          lossUpdateBatchRef.current = []; // Clear the batch after processing
          
          // --- Manually trigger chart update --- 
          if (lossChartRef.current) {
            lossChartRef.current.update('none'); // Update without animation
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
      if (scheduleProcessLossBatch && typeof scheduleProcessLossBatch.cancel === 'function') {
        scheduleProcessLossBatch.cancel();
        console.log('Cleanup: Throttled batch processing cancelled.');
      }
      
      lossUpdateBatchRef.current = []; // Clear batch on cleanup
      console.log('Cleanup: Batch ref cleared.');
      
      // Check if ws.current exists and has a close method before calling
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        console.log('Closing WebSocket connection.');
        ws.current.close();
      }
      ws.current = null; // Clear ref on unmount/cleanup
    };

    // Create WebSocket connection
    if (!ws.current) {
      console.log('Creating WebSocket connection to:', WS_URL);
      ws.current = new WebSocket(WS_URL);
    }

    // WebSocket setup
    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setWsError(null);
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      // Clear the ref when closed
      ws.current = null;
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsError('WebSocket connection error. Is the server running?');
      // Clear the ref on error
      ws.current = null;
    };

    ws.current.onmessage = (event) => {
      // --- ADDED LOG TO SEE RAW MESSAGE ---
      console.log('WebSocket message received:', event.data);
      // --- END LOG ---
      try {
          const message = JSON.parse(event.data);
          switch (message.type) {
              case 'loss_update':
                  // Add the new point to the batch ref
                  lossUpdateBatchRef.current.push({ epoch: message.epoch, mse: message.mse });
                  // Schedule the throttled processing function
                  scheduleProcessLossBatch(); // <-- MODIFIED
                  break;
              case 'final_result':
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
              case 'error':
                   // Flush and clear queue on error too
                   scheduleProcessLossBatch.flush(); // <-- MODIFIED
                   lossUpdateBatchRef.current = []; // Clear batch ref <-- ADDED
                   setNnError(`Training failed: ${message.message}`);
                   setLoadingNN(false);
                   break;
              default:
                  console.warn('Unknown WebSocket message type:', message.type);
          }
      } catch (err) {
           console.error('Failed to parse WebSocket message:', event.data, err);
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
    if (!input || typeof input !== 'string') return [];
    return input.split(',')
      .map(val => parseFloat(val.trim()))
      .filter(val => !isNaN(val));
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
         yDataValues = yDataValues.concat(nnResults.predictions.filter(p => !isNaN(p)));
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
    if (minXVal === maxXVal) { maxXVal += 1; minXVal -= 1; }
    if (minYVal === maxYVal) { maxYVal += 1; minYVal -= 1; }
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
  }, [parsedX, parsedY, lastPredictedXLr, lrPrediction, nnResults, lastPredictedXNN, nnPrediction]);

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

     const datasets = [{
         label: 'Data Points (Y)',
         data: dataPoints,
         backgroundColor: 'rgba(255, 99, 132, 1)', // Red
         pointRadius: 5,
         order: 3 // Draw data points on top
     }];

     // Add Linear Regression line
     if (currentLrSlope !== null && currentLrIntercept !== null && bounds) {
         const lineStartX = bounds.minX;
         const lineEndX = bounds.maxX;
         datasets.push({
             label: 'Linear Regression',
             data: [
               { x: lineStartX, y: currentLrSlope * lineStartX + currentLrIntercept },
               { x: lineEndX, y: currentLrSlope * lineEndX + currentLrIntercept },
             ],
             borderColor: 'rgba(54, 162, 235, 1)', // Blue
             borderWidth: 2,
             pointRadius: 0,
             fill: false,
             tension: 0,
             showLine: true,
             type: 'line',
             order: 1 // Draw LR line behind predictions
         });
     }

     // Add Linear Regression prediction visualization
     if (predictionXLr !== null && predictionYLr !== null && !isNaN(predictionXLr) && !isNaN(predictionYLr) && bounds) {
         datasets.push({
             label: 'LR Prediction',
             data: [{ x: predictionXLr, y: predictionYLr }],
             backgroundColor: 'rgba(0, 255, 0, 1)', // Changed to Green
             borderColor: 'rgba(0, 0, 0, 1)', // Match background for solid color
             borderWidth: 2, // Added for thicker lines
             pointRadius: 8, // Keep size
             pointStyle: 'crossRot',
             showLine: false,
             order: 4 // Draw LR prediction on top
         });
          // Add guide lines
          datasets.push({
             label: 'LR Prediction Guide X', // Label for tooltip filtering
             data: [
                 { x: predictionXLr, y: bounds.minY }, // Start from bottom axis
                 { x: predictionXLr, y: predictionYLr }
             ],
             borderColor: 'rgba(128, 128, 128, 0.7)', // Grey guide line
             borderWidth: 1,
             borderDash: [5, 5], // Dashed line
             pointRadius: 0,
             fill: false,
             tension: 0,
             showLine: true,
             type: 'line',
             order: 0 // Draw behind everything
          });
          datasets.push({
             label: 'LR Prediction Guide Y', // Label for tooltip filtering
             data: [
                 { x: bounds.minX, y: predictionYLr }, // Start from left axis
                 { x: predictionXLr, y: predictionYLr }
             ],
             borderColor: 'rgba(128, 128, 128, 0.7)', // Grey guide line
             borderWidth: 1,
             borderDash: [5, 5], // Dashed line
             pointRadius: 0,
             fill: false,
             tension: 0,
             showLine: true,
             type: 'line',
             order: 0 // Draw behind everything
          });
     }

     // Add Neural Network Predictions
     if (nnPreds && nnPreds.length === xValues.length && bounds) {
          // --- Sort the predictions by x-value before plotting --- 
          const nnPredictionPoints = xValues.map((x, index) => ({ x: x, y: nnPreds[index] }))
                                           .sort((a, b) => a.x - b.x); // Sort by x for correct line plotting

          // --- DEBUGGING LOGS --- 
          console.log("NN Preds Data:", nnPreds); 
          console.log("NN Prediction Points (sorted):", nnPredictionPoints);
          console.log("Rendering NN Prediction Dataset with showLine: true");
          // --- END DEBUGGING LOGS ---

          datasets.push({
              label: 'Neural Network Predictions',
              data: nnPredictionPoints,
              backgroundColor: 'rgba(255, 159, 64, 1)', // Orange points
              pointRadius: 6, // Slightly larger
              pointStyle: 'triangle', // Different shape
              // --- Changes for Interpolation Line ---
              showLine: true, // ENABLE the line
              borderColor: 'rgba(255, 159, 64, 0.7)', // Use a slightly transparent orange for the line
              borderWidth: 2, // Define line thickness
              tension: 0.4, // ADD smoothness (adjust 0.1 to 0.5 as needed)
              type: 'line', // Explicitly set type to 'line'
              // --- End Changes ---
              order: 2 // Draw NN points/line below data but above LR line
          });
     }

     // Add Neural Network prediction visualization (similar to LR prediction)
     if (predictionXNN !== null && predictionYNN !== null && !isNaN(predictionXNN) && !isNaN(predictionYNN) && bounds) {
         datasets.push({
             label: 'NN Prediction',
             data: [{ x: predictionXNN, y: predictionYNN }],
             backgroundColor: 'rgba(153, 102, 255, 1)', // Purple
             borderColor: 'rgba(0, 0, 0, 1)', 
             borderWidth: 2,
             pointRadius: 8,
             pointStyle: 'rect', // Different shape from LR
             showLine: false,
             order: 4 // Draw NN prediction on top
         });
         // Add guide lines
         datasets.push({
             label: 'NN Prediction Guide X',
             data: [
                 { x: predictionXNN, y: bounds.minY },
                 { x: predictionXNN, y: predictionYNN }
             ],
             borderColor: 'rgba(153, 102, 255, 0.7)',
             borderWidth: 1,
             borderDash: [5, 5],
             pointRadius: 0,
             fill: false,
             tension: 0,
             showLine: true,
             type: 'line',
             order: 0
         });
         datasets.push({
             label: 'NN Prediction Guide Y',
             data: [
                 { x: bounds.minX, y: predictionYNN },
                 { x: predictionXNN, y: predictionYNN }
             ],
             borderColor: 'rgba(153, 102, 255, 0.7)',
             borderWidth: 1,
             borderDash: [5, 5],
             pointRadius: 0,
             fill: false,
             tension: 0,
             showLine: true,
             type: 'line',
             order: 0
         });
     }

     return datasets;
  }, [parsedX, parsedY, lrSlope, lrIntercept, lastPredictedXLr, lrPrediction, axisBounds, nnResults, lastPredictedXNN, nnPrediction]);

  // --- Theme Toggle Effect ---
  useEffect(() => {
    // ... (keep existing logic) ...
    document.documentElement.setAttribute('data-theme', activeTheme);
  }, [activeTheme]);

  const toggleTheme = () => {
     // ... (keep existing logic) ...
     setActiveTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  // --- Generate Random Data ---
  const handleGenerateRandomData = () => {
    // ... (keep existing logic) ...
    setLrError(null);
    setNnError(null);
    // Reset LR state
    setLrSlope(null); setLrIntercept(null); setLrPrediction(null);
    setLastPredictedXLr(null); setIsLrTrained(false); setLrTrainingTime(null);
    setLrMse(null); setLrRSquared(null);
    // Reset NN state
    setNnResults(null); setLoadingNN(false);
    setLossHistory([]); // Also clear loss history

    const num = parseInt(numRandomPoints, 10);
    if (isNaN(num) || num <= 0) {
      setLrError("Please enter a valid positive number of points to generate.");
      return;
    }

    const randomSlope = (Math.random() - 0.5) * 5;
    const randomIntercept = Math.random() * 10;
    const minX = 0; const maxX = 50;
    const minYLinear = randomSlope * minX + randomIntercept;
    const maxYLinear = randomSlope * maxX + randomIntercept;
    const yRangeLinear = Math.abs(maxYLinear - minYLinear);
    const baseNoise = 1.0;
    const maxRelativeNoiseFactor = 0.4;
    let noiseMagnitude = Math.max(baseNoise, yRangeLinear * maxRelativeNoiseFactor);
    noiseMagnitude = noiseMagnitude * (1 - linearityFactor);

    const randomX = []; const randomY = [];
    for (let i = 0; i < num; i++) {
      const x = Math.random() * maxX;
      const y_linear = randomSlope * x + randomIntercept;
      const noise = (Math.random() - 0.5) * 2 * noiseMagnitude;
      const y = y_linear + noise;
      randomX.push(x.toFixed(2));
      randomY.push(y.toFixed(2));
    }

    setXInput(randomX.join(', '));
    setYInput(randomY.join(', '));
  };

  // --- handleTrainLR ---
  const handleTrainLR = async () => {
    // ... (keep existing logic) ...
    setLrError(null); setLoadingLRLinear(true);
    // Reset LR state only
    setLrSlope(null); setLrIntercept(null); setLrPrediction(null);
    setLastPredictedXLr(null); setIsLrTrained(false); setLrTrainingTime(null);
    setLrMse(null); setLrRSquared(null);
    // Don't reset NN state here

    const x_values = parsedX; const y_values = parsedY;
    if (x_values.length === 0 || y_values.length === 0 || x_values.length !== y_values.length) {
        setLrError("Invalid input: X and Y must be comma-separated numbers and have the same length.");
        setLoadingLRLinear(false); return;
    }

    try {
      const response = await fetch(`${API_URL}/lr_train`, { // Use renamed endpoint
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x_values, y_values }),
      });
      const data = await response.json();
      if (!response.ok) { throw new Error(data.error || `HTTP error! Status: ${response.status}`); }

      setLrSlope(data.slope); setLrIntercept(data.intercept);
      setLrTrainingTime(data.trainingTimeMs); setLrMse(data.mse);
      setLrRSquared(data.r_squared); setIsLrTrained(true);

    } catch (err) {
       console.error("LR Training failed:", err);
       setLrError(err.message || "An unknown error occurred during LR training.");
    } finally {
      setLoadingLRLinear(false);
    }
  };

  // --- handlePredictLR ---
  const handlePredictLR = async () => {
    console.log('[FRONTEND] handlePredictLR: Function called.'); // <-- Log Entry

    if (!isLrTrained) {
        console.warn('[FRONTEND] handlePredictLR: Model not trained. Aborting.');
        setLrError("Train the LR model before predicting.");
        return;
    }

    setLrError(null);
    setLoadingLRPredict(true);
    setLrPrediction(null); // Reset previous prediction display
    console.log('[FRONTEND] handlePredictLR: State reset (loading=true, prediction=null).');

    const x_value_parsed = parseFloat(predictXInput);

    if (isNaN(x_value_parsed)) {
        console.error('[FRONTEND] handlePredictLR: Invalid X value input:', predictXInput);
        setLrError("Invalid input: LR Prediction X value must be a number.");
        setLoadingLRPredict(false); // <-- Make sure loading is stopped on error
        return;
    }

    console.log(`[FRONTEND] handlePredictLR: Parsed X value: ${x_value_parsed}`);
    console.log(`[FRONTEND] handlePredictLR: Attempting fetch to ${API_URL}/lr_predict`);

    try {
        const bodyPayload = JSON.stringify({ x_value: x_value_parsed });
        console.log('[FRONTEND] handlePredictLR: Sending payload:', bodyPayload);

        const response = await fetch(`${API_URL}/lr_predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: bodyPayload,
        });

        console.log(`[FRONTEND] handlePredictLR: Fetch response received. Status: ${response.status}, OK: ${response.ok}`);

        if (!response.ok) {
            let errorMsg = `HTTP error! Status: ${response.status}`;
            let errorData = null;
            try {
                // Attempt to read error details from response body
                errorData = await response.json();
                console.log('[FRONTEND] handlePredictLR: Received error data from server:', errorData);
                errorMsg = errorData.error || errorMsg; // Use server error if available
            } catch (jsonError) {
                console.warn('[FRONTEND] handlePredictLR: Could not parse error response as JSON.', jsonError);
                // Fallback to status text if JSON parsing fails or no body
                errorMsg = `${errorMsg} - ${response.statusText || 'Server returned non-JSON error'}`;
            }
            throw new Error(errorMsg); // Throw error to be caught below
        }

        // --- Response seems OK, try parsing JSON ---
        console.log('[FRONTEND] handlePredictLR: Attempting response.json()');
        const data = await response.json();
        console.log('[FRONTEND] handlePredictLR: Successfully parsed JSON data:', data);

        // Validate the received data structure
        if (data.prediction === undefined || data.prediction === null || isNaN(data.prediction)) {
            console.error('[FRONTEND] handlePredictLR: Invalid data received:', data);
            throw new Error('Invalid prediction value received from server.');
        }

        // --- Data is valid, update state ---
        console.log(`[FRONTEND] handlePredictLR: Setting prediction state to: ${data.prediction}`);
        setLrPrediction(data.prediction);
        setLastPredictedXLr(x_value_parsed); // Use the parsed value
        console.log('[FRONTEND] handlePredictLR: State update complete.');

    } catch (err) {
        // Catches fetch errors (network, DNS), response.ok=false errors, .json() errors, and validation errors
        console.error("[FRONTEND] handlePredictLR: Catch block executed.", err);
        setLrError(err.message || "An unknown error occurred during LR prediction.");
        setLastPredictedXLr(null); // Clear prediction point visualization on error
        console.log('[FRONTEND] handlePredictLR: Error state updated.');

    } finally {
        // --- CRUCIAL: Ensure loading state is always reset ---
        setLoadingLRPredict(false);
        console.log('[FRONTEND] handlePredictLR: Finally block executed. Loading state set to false.');
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
    if (x_values.length === 0 || y_values.length === 0 || x_values.length !== y_values.length) {
        setNnError("Invalid input: X and Y must be non-empty and have the same length.");
        setLoadingNN(false); return;
    }
     if (!layers || typeof layers !== 'string' || !layers.match(/^(\d+-)+\d+$/)) {
         setNnError('Invalid layers format. Use dash-separated numbers (e.g., "1-3-1").');
         setLoadingNN(false); return;
     }
     if (isNaN(learning_rate)) {
         setNnError('Learning rate must be a valid number.');
         setLoadingNN(false); return;
     }
     if (isNaN(epochs) || epochs <= 0) {
         setNnError('Epochs must be a positive whole number.');
         setLoadingNN(false); return;
     }

    // Check WebSocket connection (keep existing)
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
        setNnError("WebSocket is not connected. Please ensure the server is running and refresh.");
        setLoadingNN(false);
        return;
    }

    try {
        // Make the POST request (keep existing)
        const response = await fetch(`${API_URL}/nn_train_predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x_values, y_values, layers, learning_rate, epochs }),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! Status: ${response.status}`);
        }
        console.log('NN training initiated via HTTP, waiting for WebSocket updates...');

    } catch (err) {
       console.error("NN Training Initiation failed:", err);
       setNnError(err.message || "An unknown error occurred initiating NN training.");
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
      const index = parsedX.findIndex(x => x >= x_value_parsed);
      
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
        prediction = y1 + (x_value_parsed - x1) * (y2 - y1) / (x2 - x1);
      }
      
      // Update state with the prediction
      setNnPrediction(prediction);
      setLastPredictedXNN(x_value_parsed);
      
    } catch (err) {
      console.error("NN Prediction failed:", err);
      setNnError(err.message || "An unknown error occurred during NN prediction.");
      setLastPredictedXNN(null);
    } finally {
      setLoadingNNPredict(false);
    }
  };

  // --- Formatters (keep existing) ---
  const formatTime = (timeMs) => {
    if (timeMs === null || isNaN(timeMs) || timeMs < 0) return 'N/A';
    if (timeMs < 1) return '< 1 ms';
    if (timeMs < 1000) return `${timeMs.toFixed(0)} ms`;
    return `${(timeMs / 1000).toFixed(2)} s`;
};

const formatMetric = (value, type = 'float', precision = 4) => {
  // Default precision to 4 if not provided or invalid
  const prec = typeof precision === 'number' && precision >= 0 ? precision : 4;
  if (value === null || value === undefined || isNaN(value)) return 'N/A';

  switch(type) {
    case 'mse':
        // Use exponential notation for very small or large numbers
        if (Math.abs(value) > 10000 || (Math.abs(value) < 0.0001 && value !== 0)) {
            return value.toExponential(); // Or use .toFixed()
        }
        return value.toFixed(prec);
    case 'r2':
        return (value * 100).toFixed(Math.max(0, prec - 2)) + '%'; // Adjust precision for percentage
    case 'time': // Added a case for time specifically
        return formatTime(value); // Use the existing formatTime helper
    case 'float':
    default:
        return value.toFixed(prec);
  }
};
  // --- Base Chart Options (useMemo) ---
  const baseChartOptions = useMemo(() => ({
    // ... (keep existing logic) ...
    responsive: true,
     maintainAspectRatio: false,
     plugins: { /* ... */ },
     scales: { /* ... */ },
     interaction: { /* ... */ },
  }), []);

  // --- Loss Chart Data and Options ---
  const lossChartData = useMemo(() => ({
    // Use epoch as the direct label for the x-axis
    labels: lossHistory.map(p => p.epoch),
    datasets: [
      {
        label: 'Training Loss (MSE)',
        // Data is just the MSE values
        data: lossHistory.map(p => p.mse),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.1, // Slight smoothing
        pointRadius: 1, // Small points
        borderWidth: 1.5, // Slightly thicker line
      },
    ],
  }), [lossHistory]); // Depend only on lossHistory

  const lossChartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { // Add animation configuration
        duration: 50 // Re-enable very short animation (might help ResizeObserver)
    },
    resizeDelay: 100,
    plugins: {
      legend: {
        display: false, // Keep legend hidden
      },
      title: {
        display: true,
        text: 'NN Training Loss (MSE) vs Epoch',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      x: { // Was CategoryScale, now implicitly handled by labels/data structure
        title: { display: true, text: 'Epoch' },
        ticks: {
            autoSkip: true, // Skip some labels if too crowded
            maxTicksLimit: 15 // Limit the number of visible ticks
        }
      },
      y: {
        title: { display: true, text: 'MSE (Log Scale)' }, // Update title
        type: 'logarithmic', // Use log scale
        min: 0.00001, // Adjust min slightly if needed, avoid zero/negative
        ticks: {
            // Custom formatter for log scale ticks if needed
             callback: function(value, index, values) {
                // Show only powers of 10 or specific intervals on log scale
                if (value === 1 || value === 0.1 || value === 0.01 || value === 0.001 || value === 0.0001 || value === 0.00001) {
                    return value.toExponential(); // Or use .toFixed()
                }
                if (Math.log10(value) % 1 === 0) {
                   return value.toExponential(); // Show 1e-N labels
                }
                return null; // Hide other labels
            }
        }
      },
    },
  }), []);

  // --- JSX Return ---
  return (
    <div data-theme={activeTheme} className="min-h-screen bg-base-100">
      {/* Navbar */}
      <div className="navbar bg-base-300 shadow-lg sticky top-0 z-50">
        <div className="flex-1">
          <a href="/" className="btn btn-ghost text-xl">C++ ML Playground</a>
        </div>
        <div className="flex-none">
          <button className="btn btn-square btn-ghost" onClick={toggleTheme}>
            {activeTheme === 'light' ?
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg>
              :
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" /></svg>
            }
          </button>
        </div>
      </div>

      {/* Increased overall container padding */}
      <div className="container mx-auto p-4 md:p-6">

         {/* --- Error Display Area --- */}
        {(lrError || nnError || wsError) && (
          <div role="alert" className="alert alert-error mb-4 shadow-lg">
             <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
             <div>
                {lrError && <p>Linear Regression Error: {lrError}</p>}
                {nnError && <p>Neural Network Error: {nnError}</p>}
                {wsError && <p>WebSocket Error: {wsError}</p>}
             </div>
             <button className="btn btn-sm btn-ghost" onClick={() => { setLrError(null); setNnError(null); setWsError(null); }}>Dismiss</button>
          </div>
        )}

        {/* --- Main Grid Layout - New horizontal arrangement --- */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">

           {/* Data Input Section - 3 columns */}
           <div className="lg:col-span-3">
             <div className="card bg-base-200 shadow-xl mb-4">
                <div className="card-body p-4">
                    <h2 className="card-title text-xl mb-3">1. Input Data</h2>
                    {/* Shared X Input */}
                    <div className="form-control w-full">
                        <label className="label"><span className="label-text">X Values (comma-separated)</span></label>
                        <textarea value={xInput} onChange={(e) => setXInput(e.target.value)} disabled={loadingLRLinear || loadingNN} className="textarea textarea-bordered w-full h-20" placeholder="e.g., 1, 2, 3, 4, 5"></textarea>
                    </div>
                    <div className="form-control w-full mt-2">
                        <label className="label"><span className="label-text">Y Values (comma-separated)</span></label>
                        <textarea value={yInput} onChange={(e) => setYInput(e.target.value)} disabled={loadingLRLinear || loadingNN} className="textarea textarea-bordered w-full h-20" placeholder="e.g., 2, 4, 5, 4, 5"></textarea>
                    </div>
                    <div className="divider my-2">OR</div>
                    {/* Random Data Generation */}
                    <h3 className="text-md font-semibold mb-2">Generate Random Data</h3>
                    <div className="form-control w-full">
                        <label className="label"><span className="label-text">Number of Points</span></label>
                        <input type="number" value={numRandomPoints} onChange={(e) => setNumRandomPoints(Math.max(1, parseInt(e.target.value, 10) || 1))} min="1" step="1" className="input input-bordered w-full" disabled={loadingLRLinear || loadingNN} />
                    </div>
                    <div className="form-control w-full mt-2">
                        <label className="label"><span className="label-text">Linearity: {linearityFactor.toFixed(2)}</span></label>
                        <input type="range" min="0" max="1" step="0.01" value={linearityFactor} onChange={(e) => setLinearityFactor(parseFloat(e.target.value))} className="range range-accent" disabled={loadingLRLinear || loadingNN} />
                    </div>
                    <div className="card-actions justify-end mt-3">
                        <button className="btn btn-accent" onClick={handleGenerateRandomData} disabled={loadingLRLinear || loadingNN}>Generate</button>
                    </div>
                </div>
             </div>
           </div>

           {/* Linear Regression - 3 columns */}
           <div className="lg:col-span-3">
             <div className="card bg-base-200 shadow-xl mb-4">
               <div className="card-body p-4">
                 <h2 className="card-title text-xl mb-3">2a. Linear Regression</h2>

                 {/* LR Training Button */}
                 <div className="card-actions justify-start">
                   <button className={`btn btn-primary ${loadingLRLinear ? 'loading' : ''}`} onClick={handleTrainLR} disabled={loadingLRLinear || loadingNN}>
                     {loadingLRLinear ? 'Training...' : 'Train LR'}
                   </button>
                 </div>

                 {/* LR Results Display */}
                 {isLrTrained && (
                   <div className="mt-3">
                     <h3 className="text-md font-semibold mb-2">LR Results</h3>
                     <div className="stats stats-vertical shadow bg-base-100 w-full">
                        <div className="stat py-2">
                            <div className="stat-title">Slope (m)</div>
                            <div className="stat-value text-info text-lg font-mono">{formatMetric(lrSlope)}</div>
                        </div>
                        <div className="stat py-2">
                            <div className="stat-title">Intercept (b)</div>
                            <div className="stat-value text-info text-lg font-mono">{formatMetric(lrIntercept)}</div>
                        </div>
                        <div className="stat py-2">
                            <div className="stat-title">MSE</div>
                            <div className="stat-value text-accent text-lg font-mono">{formatMetric(lrMse, 'mse')}</div>
                        </div>
                        <div className="stat py-2">
                            <div className="stat-title">R² | Train Time</div>
                            <div className="stat-value text-accent text-lg font-mono">{formatMetric(lrRSquared, 'r2')} | {formatMetric(lrTrainingTime, 'time')}</div>
                        </div>
                    </div>
                   </div>
                 )}

                 {/* LR Prediction Input */}
                 <div className="form-control mt-3">
                   <label className="label"><span className="label-text">Predict Y at X =</span></label>
                   <div className="join w-full">
                     <input type="number" value={predictXInput} onChange={(e) => setPredictXInput(e.target.value)} disabled={loadingLRPredict || !isLrTrained || loadingNN} className="input input-bordered join-item w-full" placeholder="Enter X" />
                     <button className={`btn btn-secondary join-item ${loadingLRPredict ? 'loading' : ''}`} onClick={handlePredictLR} disabled={loadingLRPredict || !isLrTrained || loadingNN}>
                       {loadingLRPredict ? '...' : 'Predict'}
                     </button>
                   </div>
                 </div>
                 
                 {/* LR Prediction Result */}
                 {lrPrediction !== null && lastPredictedXLr !== null && (
                   <p className="mt-2 text-success font-medium">
                     LR Prediction for X = {formatMetric(lastPredictedXLr, 'float', 2)}:
                     <span className="font-bold ml-2">{formatMetric(lrPrediction)}</span>
                   </p>
                 )}
                 {!isLrTrained && parsedX.length > 0 && (
                   <p className="mt-2 text-warning text-sm">Train the LR model first to enable prediction.</p>
                 )}
               </div>
             </div>
           </div>

           {/* Neural Network - 3 columns */}
           <div className="lg:col-span-3">
             <div className="card bg-base-200 shadow-xl mb-4">
               <div className="card-body p-4">
                 <h2 className="card-title text-xl mb-3">2b. Neural Network</h2>

                 {/* NN Configuration */}
                 <div className="form-control w-full">
                    <label className="label"><span className="label-text">Layer Sizes (e.g., 1-4-1)</span></label>
                    <input type="text" value={nnLayerInput} onChange={(e) => setNnLayerInput(e.target.value)} disabled={loadingNN || loadingLRLinear} className="input input-bordered w-full" placeholder="Input-Hidden-Output"/>
                 </div>
                 
                 <div className="grid grid-cols-2 gap-2 mt-2">
                    <div className="form-control w-full">
                       <label className="label"><span className="label-text">Learning Rate</span></label>
                       <input type="number" value={nnLearningRateInput} onChange={(e) => setNnLearningRateInput(e.target.value)} step="0.001" disabled={loadingNN || loadingLRLinear} className="input input-bordered w-full" placeholder="e.g., 0.01"/>
                    </div>
                    <div className="form-control w-full">
                       <label className="label"><span className="label-text">Epochs</span></label>
                       <input type="number" value={nnEpochsInput} onChange={(e) => setNnEpochsInput(e.target.value)} step="100" min="1" disabled={loadingNN || loadingLRLinear} className="input input-bordered w-full" placeholder="e.g., 1000"/>
                    </div>
                 </div>

                 {/* NN Structure */}
                 <div className="mt-3" style={{ height: "140px", overflow: "hidden" }}>
                     <h3 className="text-md font-semibold mb-1">Network Structure:</h3>
                     <NNVisualizer layerStructure={nnLayerInput} />
                 </div>

                 {/* NN Train Button */}
                 <div className="card-actions justify-start mt-2">
                   <button className={`btn btn-primary ${loadingNN ? 'loading' : ''}`} onClick={handleTrainPredictNN} disabled={loadingNN || loadingLRLinear}>
                     {loadingNN ? 'Training...' : 'Train NN & Predict'}
                   </button>
                 </div>

                 {/* NN Results */}
                 {nnResults && !loadingNN && (
                   <div className="mt-3">
                     <h3 className="text-md font-semibold mb-2">NN Results</h3>
                     <div className="stats stats-vertical shadow bg-base-100 w-full">
                       <div className="stat py-2">
                         <div className="stat-title">Final MSE</div>
                         <div className="stat-value text-accent text-lg font-mono">{formatMetric(nnResults.finalMse, 'mse')}</div>
                       </div>
                       <div className="stat py-2">
                         <div className="stat-title">Train Time</div>
                         <div className="stat-value text-secondary text-lg font-mono">{formatMetric(nnResults.trainingTimeMs, 'time')}</div>
                      </div>
                     </div>
                   </div>
                 )}
                 
                 {/* NN Prediction Input */}
                 {nnResults && !loadingNN && (
                   <div className="form-control mt-3">
                     <label className="label"><span className="label-text">Predict Y at X =</span></label>
                     <div className="join w-full">
                       <input 
                         type="number" 
                         value={predictXInputNN} 
                         onChange={(e) => setPredictXInputNN(e.target.value)} 
                         disabled={loadingNNPredict} 
                         className="input input-bordered join-item w-full" 
                         placeholder="Enter X" 
                       />
                       <button 
                         className={`btn btn-secondary join-item ${loadingNNPredict ? 'loading' : ''}`} 
                         onClick={handlePredictNN} 
                         disabled={loadingNNPredict || !nnResults}
                       >
                         {loadingNNPredict ? '...' : 'Predict'}
                       </button>
                     </div>
                   </div>
                 )}
                 
                 {/* NN Prediction Result */}
                 {nnPrediction !== null && lastPredictedXNN !== null && (
                   <p className="mt-2 text-success font-medium">
                     NN Prediction for X = {formatMetric(lastPredictedXNN, 'float', 2)}:
                     <span className="font-bold ml-2">{formatMetric(nnPrediction)}</span>
                   </p>
                 )}
                 
                 {/* Loss Chart - smaller and responsive */}
                 {(loadingNN || nnResults) && (
                   <div className="mt-2 loss-chart-container" style={{ height: '150px' }}>
                     <Line ref={lossChartRef} data={lossChartData} options={lossChartOptions} />
                   </div>
                 )}
               </div>
             </div>
           </div>

           {/* Visualization - 3 columns */}
           <div className="lg:col-span-3">
             <div className="card bg-base-200 shadow-xl mb-4">
               <div className="card-body p-4">
                 <h2 className="card-title text-xl mb-3">3. Visualization</h2>
                 
                 {/* Chart Container */}
                 <div className="w-full h-96 bg-base-100 rounded-lg chart-container">
                   {/* Use Scatter for the data points with lines */}
                   <Scatter
                    ref={chartRef}
                    data={{ datasets: memoizedDatasets }}
                    options={{
                      ...baseChartOptions,
                      scales: {
                        x: { min: axisBounds.minX, max: axisBounds.maxX },
                        y: { min: axisBounds.minY, max: axisBounds.maxY }
                      }
                    }}
                   />
                 </div>
                 
                 {/* Chart Description */}
                 <div className="mt-2 text-xs opacity-70">
                   <p>• Red circles: Original data points</p>
                   <p>• Blue line: Linear Regression model</p>
                   <p>• Green cross: LR prediction</p>
                   {nnResults && <p>• Orange triangles + line: Neural Network predictions</p>}
                   {nnResults && <p>• Purple cross: NN prediction</p>}
                 </div>
               </div>
             </div>
           </div>

        </div>
      </div>
      {/* Footer */}
      <footer className="footer footer-center p-4 bg-base-300 text-base-content rounded mt-10"> {/* Increased margin top */}
        <div>
           <p>C++ Linear Regression & Neural Network Demo with React + Node.js</p>
        </div>
      </footer>

    </div> // End App Div
  );
}

export default App;
// --- END OF FILE App.js ---