# C++ ML Playground: Linear Regression & Neural Network

[![C++](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/cpp.yml/badge.svg?branch=master)](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/cpp.yml)
[![Client](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/client.yml/badge.svg?branch=master)](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/client.yml)
[![Server](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/server.yml/badge.svg?branch=master)](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/server.yml)

A full-stack application demonstrating Linear Regression and a basic Feedforward Neural Network trained using C++ and visualized with React, Node.js, and WebSockets.

## Features

-   **Linear Regression**:
    -   Train a model using the analytical solution (normal equation).
    -   Visualize data points and the resulting regression line.
    -   Make predictions on new data points.
    -   View model performance metrics (MSE, R² score).
-   **Neural Network**:
    -   Define custom network structures (e.g., "1-4-1").
    -   Train a feedforward network using gradient descent.
    -   Specify learning rate and number of epochs.
    -   Real-time training loss visualization (MSE vs. Epoch) via WebSockets.
    -   Visualize NN predictions (orange triangles) alongside original data points, connected by a smooth interpolated line.
    -   View final MSE and training time.
-   **General**:
    -   Input data manually or generate random datasets with adjustable linearity.
    -   Interactive charts powered by Chart.js.
    -   Modern UI built with React and DaisyUI/Tailwind CSS.
    -   Dark/light theme toggle.

## Project Structure

-   **`client/`**: React frontend using `create-react-app`. Handles UI, visualization, and WebSocket communication.
-   **`server/`**: Node.js/Express backend API. Manages requests, invokes the C++ executable, and relays NN training progress via WebSockets.
-   **`cpp/`**: C++ engine containing:
    -   `linear_regression.h/.cpp`: Implementation of the Linear Regression model.
    -   `neural_network.h/.cpp`: Implementation of the Feedforward Neural Network.
    -   `main_server.cpp`: Main C++ application handling command-line arguments (`lr_train`, `nn_train_predict`) and interacting with the Node.js server via stdin/stdout.
    -   `Makefile`: Used to build the C++ executable.

## Installation

### Prerequisites

-   Node.js (v14 or higher recommended)
-   npm (usually comes with Node.js)
-   A C++ compiler supporting C++11 or later (e.g., g++, Clang, MSVC)
-   `make` build tool (standard on Linux/macOS, can be installed on Windows e.g., via Chocolatey `choco install make` or MinGW/MSYS2)

### Setup

1.  **Clone the repository**:
    ```powershell
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Build the C++ executable**:
    The server expects the executable `linear_regression_app` (or `linear_regression_app.exe` on Windows) in the `cpp/` directory.
    ```powershell
    cd cpp
    make # This should create the 'linear_regression_app' executable
    cd ..
    ```
    *(Note: Verify the `Makefile` target name matches `linear_regression_app`)*

3.  **Install Server Dependencies and Start**:
    ```powershell
    cd server
    npm install
    npm start
    ```
    *(Keep this terminal running)*

4.  **Install Client Dependencies and Start**:
    Open a *new* terminal in the project root.
    ```powershell
    cd client
    npm install
    npm start
    ```

The application should now open automatically in your browser, typically at `http://localhost:3000`. The server runs on `http://localhost:3001`.

## Usage

1.  **Input Data**:
    -   Enter comma-separated X and Y values in the text areas.
    -   *Alternatively*, configure the number of points and linearity factor, then click "Generate" to create random data.

2.  **Linear Regression**:
    -   Click "Train Linear Regression".
    -   Results (Slope, Intercept, MSE, R², Time) will appear below the button.
    -   The regression line will be drawn on the chart.
    -   Enter an X value under "Predict Y for LR Model" and click "Predict" to see the predicted point (green cross) and value.

3.  **Neural Network**:
    -   Configure the "Layer Sizes" (e.g., `1-4-1`), "Learning Rate", and "Epochs".
    -   Click "Train NN & Predict".
    -   Training progress (MSE vs. Epoch) will stream to the loss chart below the button.
    -   Once complete, final results (Final MSE, Time) will appear.
    -   NN predictions for the input X values will be plotted as orange triangles on the main chart.

4.  **Visualization**:
    -   The main chart displays data points, the LR line (if trained), LR predictions (if made), and NN predictions (triangles connected by an interpolated curve, if trained).
    -   Hover over points/lines for details.
    -   The chart automatically adjusts its axes to fit the data and predictions.

## Technologies

-   **Frontend**: React, Chart.js, DaisyUI, Tailwind CSS
-   **Backend**: Node.js, Express, ws (for WebSockets)
-   **C++ Engine**: Standard C++ (C++11) for Linear Regression (analytical solution) and Neural Network (gradient descent).
-   **Build Tools**: Make, npm

## License

MIT

## Acknowledgments

- [DaisyUI](https://daisyui.com/) for the UI components
- [Chart.js](https://www.chartjs.org/) for data visualization
