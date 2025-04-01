// --- START OF FILE server.js ---

const express = require('express');
const { spawn } = require('child_process'); // Using spawn
const path = require('path');
const cors = require('cors');

const app = express();
const port = 3001; // Port for the backend server

// --- Configuration ---
const cppExecutablePath = path.join(__dirname, '..', 'cpp', 'linear_regression_app');
// --- End Configuration ---


// --- Middleware ---
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));
// --- End Middleware ---


// Store the last trained LINEAR REGRESSION model parameters in memory
let trainedLRModel = {
    slope: null,
    intercept: null,
    trained: false
};
// NOTE: We are NOT storing NN state here as it's handled within one C++ call

// --- Helper Function ---
function parseCppOutput(output) {
    const results = {};
    output.split('\n').forEach(line => {
        if (line.includes('=')) {
            const [key, value] = line.split('=', 2);
            if (key && value !== undefined) { // Check value is not undefined
                const trimmedKey = key.trim();
                const trimmedValue = value.trim();
                // Special handling for predictions array
                if (trimmedKey === 'nn_predictions') {
                    results[trimmedKey] = trimmedValue.split(',')
                                             .map(v => parseFloat(v.trim()))
                                             .filter(v => !isNaN(v)); // Parse to numbers
                } else if (trimmedKey) { // Ensure key is not empty
                   // Try parsing as float, store original string if failed but not empty
                   const numValue = parseFloat(trimmedValue);
                   results[trimmedKey] = isNaN(numValue) ? trimmedValue : numValue;
                }
            }
        }
    });
    return results;
}


// --- API Routes ---

// POST /api/lr_train (Renamed)
app.post('/api/lr_train', (req, res) => {
    const { x_values, y_values } = req.body;

    if (!Array.isArray(x_values) || !Array.isArray(y_values) || x_values.length === 0 || x_values.length !== y_values.length) {
        return res.status(400).json({ error: 'Invalid input data. Ensure X and Y are non-empty arrays of the same length.' });
    }

    const x_str = x_values.join(',');
    const y_str = y_values.join(',');

    const args = ['lr_train']; // Use new command

    console.log(`Spawning: ${cppExecutablePath} ${args.join(' ')} (data via stdin)`);
    const cppProcess = spawn(cppExecutablePath, args);
    const stdinData = `${x_str}\n${y_str}\n`;

    let stdoutData = '';
    let stderrData = '';

    cppProcess.stdin.write(stdinData);
    cppProcess.stdin.end();

    cppProcess.stdout.on('data', (data) => { stdoutData += data.toString(); });
    cppProcess.stderr.on('data', (data) => { stderrData += data.toString(); });

    cppProcess.on('close', (code) => {
        console.log(`C++ process (lr_train) exited with code ${code}`);
        if (stderrData) { console.error(`C++ Stderr: ${stderrData}`); }

        if (code !== 0) {
            const errorMsg = stderrData || `C++ process failed with exit code ${code}`;
            return res.status(500).json({ error: `LR Training failed: ${errorMsg}` });
        }

        console.log(`C++ Stdout: ${stdoutData}`);
        const results = parseCppOutput(stdoutData);

        if (results.slope === undefined || results.intercept === undefined || isNaN(results.slope) || isNaN(results.intercept)) {
             console.error(`Error parsing C++ LR output: ${stdoutData}`);
             return res.status(500).json({ error: 'Failed to parse LR training results from C++.' });
        }

        // Store trained LR parameters
        trainedLRModel.slope = results.slope;
        trainedLRModel.intercept = results.intercept;
        trainedLRModel.trained = true;

        res.json({
            slope: trainedLRModel.slope,
            intercept: trainedLRModel.intercept,
            trainingTimeMs: results.training_time_ms,
            mse: results.mse,
            r_squared: results.r_squared
        });
    });

    cppProcess.on('error', (err) => {
        console.error(`Failed to start C++ subprocess: ${err.message}`);
        res.status(500).json({ error: `Server error: Failed to execute training process. Path: ${cppExecutablePath}` });
    });
});

// POST /api/lr_predict (Renamed)
app.post('/api/lr_predict', (req, res) => {
    const { x_value } = req.body;

    if (!trainedLRModel.trained) {
        return res.status(400).json({ error: 'Linear Regression Model not trained yet. Train the LR model first.' });
    }
    if (typeof x_value !== 'number') {
        return res.status(400).json({ error: 'Invalid input. x_value must be a number.' });
    }

    // We can directly calculate the prediction here if the model is simple
     const prediction = trainedLRModel.slope * x_value + trainedLRModel.intercept;
     console.log(`Calculating LR prediction directly: ${prediction} for x=${x_value}`);
     res.json({ prediction: prediction });

    /* // --- Alternative: Call C++ for prediction (if predict logic was complex) ---
    const args = [
        'lr_predict', // Use new command
        String(trainedLRModel.slope),
        String(trainedLRModel.intercept),
        String(x_value)
    ];

    console.log(`Spawning: ${cppExecutablePath} ${args.join(' ')}`); // Log execution
    const cppProcess = spawn(cppExecutablePath, args); // Use spawn

    let stdoutData = '';
    let stderrData = '';

    cppProcess.stdout.on('data', (data) => { stdoutData += data.toString(); });
    cppProcess.stderr.on('data', (data) => { stderrData += data.toString(); });

    cppProcess.on('close', (code) => {
        console.log(`C++ process (lr_predict) exited with code ${code}`);
        if (stderrData) { console.error(`C++ Stderr: ${stderrData}`); }

        if (code !== 0) {
            const errorMsg = stderrData || `C++ process failed with exit code ${code}`;
            return res.status(500).json({ error: `LR Prediction failed: ${errorMsg}` });
        }

        console.log(`C++ Stdout: ${stdoutData}`);
        const results = parseCppOutput(stdoutData);

        if (results.prediction === undefined || isNaN(results.prediction)) {
            console.error(`Error parsing C++ prediction output: ${stdoutData}`);
            return res.status(500).json({ error: 'Failed to parse LR prediction result from C++.' });
        }

        res.json({ prediction: results.prediction });
    });

     cppProcess.on('error', (err) => {
        console.error(`Failed to start C++ subprocess: ${err.message}`);
        res.status(500).json({ error: `Server error: Failed to execute prediction process. Path: ${cppExecutablePath}` });
    });
    // --- End Alternative --- */
});


// POST /api/nn_train_predict (New Endpoint)
app.post('/api/nn_train_predict', (req, res) => {
    const { x_values, y_values, layers, learning_rate, epochs } = req.body;

    // Basic input validation
    if (!Array.isArray(x_values) || !Array.isArray(y_values) || x_values.length === 0 || x_values.length !== y_values.length) {
        return res.status(400).json({ error: 'Invalid input data. Ensure X and Y are non-empty arrays of the same length.' });
    }
    if (typeof layers !== 'string' || !layers.match(/^(\d+-)+\d+$/)) {
         return res.status(400).json({ error: 'Invalid layers format. Expected string like "1-3-1".' });
    }
     if (typeof learning_rate !== 'number' || isNaN(learning_rate)) {
         return res.status(400).json({ error: 'Invalid learning rate. Expected a number.' });
     }
     if (typeof epochs !== 'number' || !Number.isInteger(epochs) || epochs <= 0) {
         return res.status(400).json({ error: 'Invalid epochs. Expected a positive integer.' });
     }

    // Format data and args for C++
    const x_str = x_values.join(',');
    const y_str = y_values.join(',');
    const args = [
        'nn_train_predict', // The new command
        layers,
        String(learning_rate),
        String(epochs)
    ];

    console.log(`Spawning: ${cppExecutablePath} ${args.join(' ')} (data via stdin)`);
    const cppProcess = spawn(cppExecutablePath, args);
    const stdinData = `${x_str}\n${y_str}\n`;

    let stdoutData = '';
    let stderrData = '';

    cppProcess.stdin.write(stdinData);
    cppProcess.stdin.end();

    cppProcess.stdout.on('data', (data) => { stdoutData += data.toString(); });
    cppProcess.stderr.on('data', (data) => { stderrData += data.toString(); });

    cppProcess.on('close', (code) => {
        console.log(`C++ process (nn_train_predict) exited with code ${code}`);
        if (stderrData) { console.error(`C++ Stderr: ${stderrData}`); }

        if (code !== 0) {
            const errorMsg = stderrData || `C++ process failed with exit code ${code}`;
            return res.status(500).json({ error: `NN Training/Prediction failed: ${errorMsg}` });
        }

        console.log(`C++ Stdout: ${stdoutData}`);
        const results = parseCppOutput(stdoutData);

        // Validate crucial NN results
        if (results.training_time_ms === undefined || isNaN(results.training_time_ms) ||
            results.final_mse === undefined || isNaN(results.final_mse) ||
            !Array.isArray(results.nn_predictions) || results.nn_predictions.length !== x_values.length)
        {
             console.error(`Error parsing C++ NN output or mismatched predictions: ${stdoutData}`);
             console.error('Parsed results:', results); // Log parsed results for debugging
             return res.status(500).json({ error: 'Failed to parse valid NN results or predictions from C++.' });
        }

        // Return all parsed results
        res.json({
            trainingTimeMs: results.training_time_ms,
            finalMse: results.final_mse, // Use the key from C++ output
            predictions: results.nn_predictions // The array of predicted Y values
        });
    });

    cppProcess.on('error', (err) => {
        console.error(`Failed to start C++ subprocess: ${err.message}`);
        res.status(500).json({ error: `Server error: Failed to execute NN process. Path: ${cppExecutablePath}` });
    });
});


// --- End API Routes ---


// --- Start Server ---
app.listen(port, () => {
    console.log(`Node.js server listening on http://localhost:${port}`);
    console.log(`Expecting C++ executable at: ${cppExecutablePath}`);
    // Check if executable exists (optional sanity check)
    require('fs').access(cppExecutablePath, require('fs').constants.X_OK, (err) => {
        if (err) {
            console.warn(`Warning: C++ executable not found or not executable at ${cppExecutablePath}`);
        } else {
             console.log(`C++ executable found and seems executable.`);
        }
    });
});
// --- End Start Server ---
// --- END OF FILE server.js ---