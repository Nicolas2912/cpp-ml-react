# Linear Regression Application

A full-stack application for training and visualizing a linear regression model using C++, Node.js, and React.

## Features

- Train a linear regression model with custom parameters (learning rate, iterations, batch size)
- Visualize data points and the resulting regression line
- Make predictions on new data points
- View model performance metrics (MSE, R² score)
- Dark/light theme toggle

## Project Structure

- **Client**: React frontend with interactive UI using DaisyUI
- **Server**: Node.js/Express API to handle requests
- **C++ Engine**: Core linear regression implementation in C++

## Installation

### Prerequisites

- Node.js (14+)
- C++ compiler (g++ or MSVC)
- Make (for compiling C++ code)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/linear-regression.git
   cd linear-regression
   ```

2. Build the C++ executable:
   ```
   cd cpp
   make
   ```

3. Install and start the server:
   ```
   cd ../server
   npm install
   npm start
   ```

4. Install and start the client:
   ```
   cd ../client
   npm install
   npm start
   ```

The application should now be running at http://localhost:3000

## Usage

1. **Train the Model**:
   - Enter comma-separated X and Y values
   - Adjust hyperparameters as needed (learning rate, iterations, batch size)
   - Click "Train Model"

2. **View Results**:
   - The model's slope, intercept, and performance metrics will display
   - The visualization shows data points and the regression line

3. **Make Predictions**:
   - Enter an X value in the prediction field
   - Click "Predict" to get the corresponding Y value
   - The prediction point will appear on the chart

## Technologies

- **Frontend**: React, Chart.js, DaisyUI/Tailwind CSS
- **Backend**: Node.js, Express
- **Algorithm**: C++ implementation of gradient descent

## Performance

The C++ implementation provides superior performance compared to JavaScript alternatives:
- Efficient gradient descent algorithm
- Mini-batch training for improved scalability
- Fast prediction capabilities

## License

MIT

## Acknowledgments

- [DaisyUI](https://daisyui.com/) for the UI components
- [Chart.js](https://www.chartjs.org/) for data visualization
