// --- START OF FILE server.js ---

const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const port = 3001;

// --- Configuration ---
const cppExecutablePath = path.join(__dirname, '..', 'cpp', 'linear_regression_app');
const CPP_PROCESS_TIMEOUT_MS = 30000; // 30 seconds timeout for C++ processes
// --- End Configuration ---


// --- Middleware ---
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));
// --- End Middleware ---


// Store LR model (no changes)
let trainedLRModel = {
    slope: null,
    intercept: null,
    trained: false
};

// --- WebSocket Setup --- (no changes)
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const clients = new Set();

wss.on('connection', (ws) => {
    console.log('Client connected via WebSocket');
    clients.add(ws);
    ws.on('message', (message) => { /* ... */ });
    ws.on('close', () => { console.log('Client disconnected'); clients.delete(ws); });
    ws.on('error', (error) => { console.error('WebSocket error:', error); clients.delete(ws); });
});

function broadcast(data) {
    const message = JSON.stringify(data);
    clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message, (err) => {
                if (err) { console.error('Error sending message to client:', err); }
            });
        }
    });
}
// --- End WebSocket Setup ---

// --- Helper Functions ---

// parseCppLine: Handles both loss updates and final stats
function parseCppLine(line) {
    line = line.trim(); // Trim whitespace
    if (!line) return null; // Ignore empty lines

    if (line.includes('=')) {
        // Check for epoch/mse line first
        if (line.startsWith('epoch=') && line.includes('mse=')) {
            const epochMatch = line.match(/epoch=([\d.]+)/);
            const mseMatch = line.match(/mse=([\d.eE+-]+)/); // Handle scientific notation for MSE
            if (epochMatch && mseMatch) {
                const epoch = parseInt(epochMatch[1], 10);
                const mse = parseFloat(mseMatch[1]);
                if (!isNaN(epoch) && !isNaN(mse)) {
                    return { type: 'loss_update', epoch: epoch, mse: mse };
                }
            }
        }

        // Otherwise, treat as a potential key=value stat
        const [key, ...valueParts] = line.split('='); // Handle '=' in value if any (e.g., predictions)
        const value = valueParts.join('=').trim();

        if (key && value !== undefined) {
            const trimmedKey = key.trim();

            if (trimmedKey === 'nn_predictions') {
                const predictions = value.split(',')
                                     .map(v => parseFloat(v.trim()))
                                     .filter(v => !isNaN(v)); // Keep only valid numbers
                return { type: 'final_stat', key: trimmedKey, value: predictions };
            } else if (trimmedKey) {
               // Try parsing as number, fallback to string
               const numValue = parseFloat(value);
               const finalValue = isNaN(numValue) ? value : numValue;
               return { type: 'final_stat', key: trimmedKey, value: finalValue };
            }
        }
    }
    // If line doesn't contain '=', it might be other C++ output (e.g., warnings)
    // console.log("Unparsed C++ line:", line); // Optional: Log lines that aren't key=value
    return null;
}


// scaleData (no changes)
function scaleData(data) {
    // ... (keep existing implementation) ...
    if (!data || data.length === 0) {
        return { scaled: [], min: 0, range: 1 };
    }
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = (max - min) === 0 ? 1 : (max - min);
    const scaled = data.map(val => (val - min) / range);
    return { scaled, min, range };
}

// inverseScaleData (no changes)
function inverseScaleData(scaledData, min, range) {
    // ... (keep existing implementation) ...
    if (!scaledData) return [];
    return scaledData.map(val => (val * range) + min);
}


// --- API Routes ---

// POST /api/lr_train (Modified)
app.post('/api/lr_train', (req, res) => {
    console.log('--- LR Train Request Received ---'); // <-- Log Start
    const { x_values, y_values } = req.body;

    if (!Array.isArray(x_values) || !Array.isArray(y_values) || x_values.length === 0 || x_values.length !== y_values.length) {
        console.error('LR Train Error: Invalid input data received.', { x_count: x_values?.length, y_count: y_values?.length });
        return res.status(400).json({ error: 'Invalid input data. Ensure X and Y are non-empty arrays of the same length.' });
    }

    const x_str = x_values.join(',');
    const y_str = y_values.join(',');
    const args = ['lr_train']; // Argument for C++ main()

    const cppDirectory = path.dirname(cppExecutablePath);
    const isWindows = process.platform === 'win32';
    const command = isWindows ? 'cmd.exe' : cppExecutablePath;
    const commandArgs = isWindows
        ? ['/c', path.basename(cppExecutablePath), ...args]
        : args;

    console.log(`LR Train: Spawning C++ via ${command} ${commandArgs.join(' ')}`);
    const cppProcess = spawn(command, commandArgs, {
        cwd: cppDirectory // Ensure the executable runs relative to its directory
    });

    // Data to send to C++ stdin
    const stdinData = `${x_str}\n${y_str}\n`; // Ensure final newline might help some shells/C++ runtimes
    console.log(`LR Train: Writing to stdin (on spawn):\n${stdinData}`); // <-- Log Input

    let stdoutData = '';
    let stderrData = '';
    let killedForTimeout = false;
    let timeoutHandle = null;

    // Error handler for spawn itself
    cppProcess.on('error', (err) => {
        console.error(`LR Train Error: Failed to start C++ subprocess: ${err.message}`);
        console.error(`LR Train Error: Path was ${cppExecutablePath}`);
        // Ensure response is sent ONLY if headers not already sent
        if (!res.headersSent) {
             res.status(500).json({ error: `Server error: Failed to execute LR training process. Check server logs. Path: ${cppExecutablePath}` });
        }
    });

    // Only write to stdin once the process has actually spawned
    cppProcess.on('spawn', () => {
        // Set up timeout to avoid hanging processes
        timeoutHandle = setTimeout(() => {
            console.error(`LR Train Error: C++ process exceeded timeout of ${CPP_PROCESS_TIMEOUT_MS} ms. Killing process.`);
            killedForTimeout = true;
            try { cppProcess.kill('SIGKILL'); } catch (e) { /* ignore */ }
        }, CPP_PROCESS_TIMEOUT_MS);

        // --- Write data to C++ stdin ---
        cppProcess.stdin.write(stdinData, (err) => {
             if (err) {
                 console.error("LR Train Error: Failed to write to C++ stdin:", err);
                  // Try to close stdin anyway to free resources
                  try { cppProcess.stdin.end(); } catch (e) { /* ignore */ }
             } else {
                 console.log("LR Train: Successfully wrote to stdin.");
                 // --- IMPORTANT: End the stdin stream ---
                 cppProcess.stdin.end(() => {
                     console.log("LR Train: stdin stream ended.");
                 });
             }
        });
    });

    // Stderr listener
    cppProcess.stderr.on('data', (data) => {
        const errChunk = data.toString();
        stderrData += errChunk;
        console.error(`LR Train C++ Stderr Chunk: ${errChunk}`); // <-- Log Stderr
    });

    // Stdout listener
    cppProcess.stdout.on('data', (data) => {
        const outChunk = data.toString();
        stdoutData += outChunk;
        console.log(`LR Train C++ Stdout Chunk: ${outChunk}`); // <-- Log Stdout
    });

    // Close listener (crucial)
    cppProcess.on('close', (code) => {
        if (timeoutHandle) { clearTimeout(timeoutHandle); timeoutHandle = null; }
        console.log(`--- LR Train C++ process exited with code ${code} ---`);
        if (stderrData) {
            console.error(`LR Train C++ Final Stderr:\n${stderrData}`);
        }
        console.log(`LR Train C++ Final Stdout:\n${stdoutData}`); // <-- Log Final Output

        if (res.headersSent) {
             console.warn("LR Train: Headers already sent before C++ close event. Cannot send response.");
             return;
        }

        if (killedForTimeout) {
            return res.status(504).json({ error: `LR Training timed out after ${CPP_PROCESS_TIMEOUT_MS} ms.` });
        }

        if (code !== 0) {
            const errorMsg = stderrData.trim() || `C++ process failed with exit code ${code}`;
            console.error(`LR Train Error: C++ process exited abnormally. Code: ${code}`);
            return res.status(500).json({ error: `LR Training failed in C++: ${errorMsg}` });
        }

        // --- Parse final results ONLY for LR ---
        const results = {};
        let parseError = false;
        try {
            stdoutData.split('\n').forEach(line => {
                if (line.trim()) { // Process non-empty lines
                    const parsed = parseCppLine(line); // Use your existing helper
                    if (parsed && parsed.type === 'final_stat') {
                        results[parsed.key] = parsed.value;
                    } else if (line.includes('=')){ // Log if it looked like a stat but didn't parse
                        console.warn(`LR Train: Could not parse final stat line: ${line.trim()}`);
                    }
                }
             });
        } catch (e) {
            console.error("LR Train Error: Exception during output parsing:", e);
            parseError = true;
        }

        console.log("LR Train: Parsed results:", results); // <-- Log Parsed Results

        // --- Validate parsed results ---
        if (parseError || results.slope === undefined || results.intercept === undefined || isNaN(results.slope) || isNaN(results.intercept) || results.training_time_ms === undefined || isNaN(results.training_time_ms) || results.mse === undefined || isNaN(results.mse) || results.r_squared === undefined || isNaN(results.r_squared)) {
             console.error(`LR Train Error: Failed to parse essential C++ LR output or values invalid/missing. Raw output:\n${stdoutData}`);
             return res.status(500).json({ error: 'Failed to parse valid LR training results from C++ process.' });
        }

        // --- Success: Update server state and send response ---
        trainedLRModel.slope = results.slope;
        trainedLRModel.intercept = results.intercept;
        trainedLRModel.trained = true; // Mark model as trained

        console.log('LR Train: Sending success response to frontend.');
        res.json({
            slope: trainedLRModel.slope,
            intercept: trainedLRModel.intercept,
            trainingTimeMs: results.training_time_ms,
            mse: results.mse,
            r_squared: results.r_squared
            // Add any other metrics C++ outputs
        });
    });

    // Note: stdin write is handled in the 'spawn' event above
});

// POST /api/lr_predict (no changes)
app.post('/api/lr_predict', (req, res) =>{
    console.log('--- [BACKEND] LR Predict Request Received ---'); // Log Entry
    const { x_value } = req.body;

    // Check 1: Is the model trained?
    console.log('[BACKEND] LR Predict: Current trainedLRModel state:', trainedLRModel);
    if (!trainedLRModel.trained) {
        console.error('[BACKEND] LR Predict Error: Server state indicates model not trained.');
        // Send Response on Error
        return res.status(400).json({ error: 'Linear Regression Model not trained yet. Train the LR model first.' });
    }

    // Check 2: Is x_value valid?
    console.log(`[BACKEND] LR Predict: Received x_value: ${x_value} (Type: ${typeof x_value})`);
    if (typeof x_value !== 'number' || isNaN(x_value)) {
        console.error('[BACKEND] LR Predict Error: Invalid x_value received.');
        // Send Response on Error
        return res.status(400).json({ error: 'Invalid input. x_value must be a number.' });
    }

    // Check 3: Are stored parameters valid?
    console.log(`[BACKEND] LR Predict: Using slope=${trainedLRModel.slope}, intercept=${trainedLRModel.intercept}`);
    if (trainedLRModel.slope === null || isNaN(trainedLRModel.slope) || trainedLRModel.intercept === null || isNaN(trainedLRModel.intercept)) {
         console.error('[BACKEND] LR Predict Error: Server has invalid slope/intercept stored.');
         // Send Response on Error
        return res.status(500).json({ error: 'Server error: Stored model parameters are invalid.' });
    }

    // Perform Calculation
    let prediction;
    try {
        prediction = trainedLRModel.slope * x_value + trainedLRModel.intercept;
        console.log('[BACKEND] LR Predict: Calculated prediction:', prediction);
        if (isNaN(prediction)){
             throw new Error("Calculation resulted in NaN"); // Catch potential NaN result
        }
    } catch (calcError) {
        console.error('[BACKEND] LR Predict Error: Error during calculation:', calcError);
        // Send Response on Error
        return res.status(500).json({ error: 'Server error during prediction calculation.' });
    }


    // Send Response
    console.log('[BACKEND] LR Predict: Sending success response:', { prediction: prediction });
    // Ensure response is sent ONLY if headers not already sent (good practice)
    if (!res.headersSent) {
        res.json({ prediction: prediction });
    } else {
        console.warn("[BACKEND] LR Predict: Headers already sent before sending success response. This shouldn't happen.");
    }
});

// POST /api/nn_train_predict (MODIFIED for Streaming and Final Results)
app.post('/api/nn_train_predict', (req, res) => {
    const { x_values, y_values, layers, learning_rate, epochs } = req.body;

    // --- Input Validation (no changes) ---
    if (!Array.isArray(x_values) || /* ... */ !Number.isInteger(epochs) || epochs <= 0) {
         return res.status(400).json({ error: 'Invalid input data...' }); // Add specific error messages
    }
    // --- End Input Validation ---

    // --- Scale Data (no changes) ---
    const { scaled: scaled_x, min: minX, range: rangeX } = scaleData(x_values);
    const { scaled: scaled_y, min: minY, range: rangeY } = scaleData(y_values);
    // --- End Scale Data ---

    // Args for C++: command name must match C++ main() logic
    const scaled_x_str = scaled_x.join(',');
    const scaled_y_str = scaled_y.join(',');
    const args = [
        'nn_train_predict', // Command for C++ main() to trigger train_for_epochs
        layers,
        String(learning_rate),
        String(epochs)
    ];

    console.log(`Spawning NN Train: ${cppExecutablePath} ${args.join(' ')}`);
    const cppProcess = spawn(cppExecutablePath, args);
    const stdinData = `${scaled_x_str}\n${scaled_y_str}\n`;

    // --- Immediately respond to HTTP request ---
    res.json({ status: 'Training started. Check WebSocket for updates.' });
    // --- End Immediate response ---

    let stdoutBuffer = '';
    let stderrData = '';
    const finalResults = {}; // Store final stats parsed from stdout (key-value pairs)

    cppProcess.stdin.write(stdinData);
    cppProcess.stdin.end();

    // --- Handle C++ stdout Stream ---
    cppProcess.stdout.on('data', (data) => {
        stdoutBuffer += data.toString();
        let newlineIndex;
        while ((newlineIndex = stdoutBuffer.indexOf('\n')) >= 0) {
            const line = stdoutBuffer.substring(0, newlineIndex); // Don't trim here yet
            stdoutBuffer = stdoutBuffer.substring(newlineIndex + 1);

            if (line.trim()) { // Process non-empty lines after potential trim
                // console.log(`C++ stdout line raw: [${line}]`); // Debug raw lines
                const parsedData = parseCppLine(line); // parseCppLine handles trimming
                if (parsedData) {
                    // console.log("Parsed data:", parsedData); // Debug parsed data
                    if (parsedData.type === 'loss_update') {
                        broadcast(parsedData); // Send loss update via WebSocket
                    } else if (parsedData.type === 'final_stat') {
                        // Store final stats as they arrive (overwriting if key repeats, shouldn't happen for final stats)
                        finalResults[parsedData.key] = parsedData.value;
                    }
                } else {
                     // Only log warnings for lines that look like they *should* parse but didn't
                     if(line.includes('=')) {
                        console.warn(`Could not parse C++ stdout line: ${line.trim()}`);
                     }
                }
            }
        }
    });
    // --- End Handle C++ stdout Stream ---

    cppProcess.stderr.on('data', (data) => { stderrData += data.toString(); });

    // --- Handle C++ Process Exit ---
    cppProcess.on('close', (code) => {
        console.log(`C++ process (nn_train_predict) exited with code ${code}`);
        if (stderrData) { console.error(`C++ Stderr (NN Train): ${stderrData}`); }

        // Process any remaining data in the stdout buffer
        if (stdoutBuffer.trim()) {
             console.log(`C++ stdout final buffer: ${stdoutBuffer.trim()}`);
             stdoutBuffer.split('\n').forEach(line => {
                 if(line.trim()){
                     const parsedData = parseCppLine(line);
                     if (parsedData && parsedData.type === 'final_stat') {
                         finalResults[parsedData.key] = parsedData.value;
                     } else if (line.includes('=')) {
                         console.warn(`Could not parse final C++ stdout line: ${line.trim()}`);
                     }
                 }
             });
        }

        if (code !== 0) {
            const errorMsg = stderrData.trim() || `C++ process failed with exit code ${code}`;
            broadcast({ type: 'error', message: `NN Training failed: ${errorMsg}` });
            return; // Don't send final results on error
        }

        // --- Validate and Send Final Results ---
        console.log("Final collected results:", finalResults); // Debug final stats

        // Check if essential final results were received
        if (finalResults.training_time_ms === undefined || isNaN(finalResults.training_time_ms) ||
            finalResults.final_mse === undefined || isNaN(finalResults.final_mse) ||
            !Array.isArray(finalResults.nn_predictions) ) {
             console.error('Error: Missing or invalid final C++ NN stats:', finalResults);
             broadcast({ type: 'error', message: 'Failed to parse expected final NN results (time, mse, predictions) from C++.' });
             return;
        }
         // Check prediction length consistency (allow empty input case if needed)
         if (x_values.length > 0 && finalResults.nn_predictions.length !== x_values.length) {
             console.error(`Error: Mismatched prediction count. Expected ${x_values.length}, got ${finalResults.nn_predictions.length}. Results:`, finalResults);
             broadcast({ type: 'error', message: `Prediction length mismatch from C++. Expected ${x_values.length}, got ${finalResults.nn_predictions.length}.` });
             return;
         }


        // Inverse Scale Predictions
        const originalScalePredictions = inverseScaleData(finalResults.nn_predictions, minY, rangeY);

        // Broadcast final results via WebSocket
        broadcast({
            type: 'final_result',
            trainingTimeMs: finalResults.training_time_ms,
            finalMse: finalResults.final_mse,
            predictions: originalScalePredictions
        });
        // --- End Validate and Send Final Results ---
    });
    // --- End Handle C++ Process Exit ---

    cppProcess.on('error', (err) => {
        console.error(`Failed to start NN C++ subprocess: ${err.message}`);
        broadcast({ type: 'error', message: `Server error: Failed to execute NN process. Path: ${cppExecutablePath}` });
    });
});


// --- End API Routes ---


// --- Start Server --- (no changes)
server.listen(port, () => {
    console.log(`Node.js server with WebSocket listening on http://localhost:${port}`);
    console.log(`Expecting C++ executable at: ${cppExecutablePath}`);
    require('fs').access(cppExecutablePath, require('fs').constants.X_OK, (err) => {
        if (err) { console.warn(`Warning: C++ executable not found or not executable at ${cppExecutablePath}`); }
        else { console.log(`C++ executable found and seems executable.`); }
    });
});
// --- End Start Server ---
// --- END OF FILE server.js ---
