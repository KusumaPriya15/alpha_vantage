1. Use Case Details

Title of the Use Case

Predicting Stock Market Trends Using Alpha Vantage API

Problem Statement

Stock market investors and traders rely on technical indicators to make informed decisions.

Analyzing large volumes of stock data efficiently is critical to identifying profitable trading opportunities.

Incorrect or delayed analysis can lead to financial losses.

Target Users: Retail investors, traders, and financial analysts seeking a data-driven approach to stock market trend prediction.

Selected API

API Name: Alpha Vantage API

API Contribution: Provides historical stock prices, technical indicators, and real-time market data, essential for training predictive models and trend analysis.

Proposed Solution

Extract stock data (Open, High, Low, Close, Volume) using the Alpha Vantage API.

Compute technical indicators like moving averages (SMA_5, SMA_10) and volatility.

Train a machine learning model to predict stock price trends.

Display insights and recommendations on a dashboard for informed decision-making.

2. Technical Implementation

Core Functionality

Predicts stock trading volume based on historical price data.

Uses HistGradientBoostingRegressor for machine learning predictions.

Provides a Flask-based web interface for user input and results visualization.

ML Project Architecture

End-to-End Workflow:

Data Collection → Data Processing → Model Training → Evaluation → Deployment

Data Flow:

API fetches stock price data → Preprocessed into DataFrame → Trained on ML model → Served via Flask API

ML Pipeline Components:

Data Acquisition: Alpha Vantage API

Preprocessing: Feature selection & scaling

Model Training: HistGradientBoostingRegressor

Evaluation: Performance metrics & validation

Deployment: Flask + Render.com hosting

Scalability Strategies:

API-based data retrieval ensures scalability.

Model retraining can be automated periodically.

Cloud hosting allows easy expansion.

Model Selection & Justification

Chosen Model: HistGradientBoostingRegressor

Handles non-linear relationships effectively.

Efficient on structured datasets like stock prices.

Robust against overfitting.

Alternative Models Tested:

Compared against Linear Regression and Random Forest.

HistGradientBoostingRegressor showed superior accuracy.

Training Process

Data Split:

80% Training, 10% Validation, 10% Testing

Techniques Used:

Feature scaling for better performance.

Outlier removal to enhance model accuracy.

Early stopping to prevent overfitting.

Loss Function:

Root Mean Square Error (RMSE)

Hyperparameter Tuning:

Used Optuna for parameter optimization.

Model Evaluation & Performance Metrics

R² Score: 0.5701

Optimization & Engineering Enhancements

Performance Optimization:

Reduced computation time by selecting key features.

Implemented parallel processing for faster training.

Code Efficiency & Best Practices:

Used Flask & Render.com for smooth deployment.

Modular code structure for maintainability.

Model Deployment & Integration

Deployment Platform: Hosted on Render.com

Integration:

Flask API serves predictions.

User submits prices via HTML form.

Prediction displayed dynamically.

Challenges Faced & Solutions:

API Rate Limits: Implemented request caching.

Model Latency: Optimized with feature selection.

3. UI/UX Implementation

Overview of the User Interface (UI)

Application UI Description:

Provides an intuitive web interface for users to enter stock price details and receive predicted trading volume.

UI consists of an input form (index.html) and a result page (output.html) styled with CSS (style.css).

Core UI Components:

Input Form: Users enter Open, High, Low, and Close prices.

Submit Button: Triggers model prediction.

Prediction Display: Shows forecasted stock volume.

Navigation Link: Allows users to return and make another prediction.

User Flow & Navigation

User visits the homepage (index.html).

Inputs stock price values and submits the form.

Backend processes the data and generates predictions.

User is redirected to output.html with the predicted stock volume.

User can return to the homepage for a new prediction.

Design Principles & Aesthetics

Minimalist Design:

Dark-themed stock market background for a professional look.

Semi-transparent containers ensure readability.

Simple typography (Georgia, sans-serif) enhances clarity.

Usability Enhancements:

Clear labeling for input fields.

Responsive design for different screen sizes.

Hover effects for buttons to improve interactivity.

Responsiveness & Cross-Platform Compatibility

Responsive Features:

CSS media queries ensure proper display on mobile, tablet, and desktop.

Flexible container widths adjust dynamically.

Testing Across Devices:

Verified on Chrome, Firefox, Edge, and Safari.

Ensured consistency across different screen sizes.

User Accessibility & Inclusivity

Accessibility Considerations:

High contrast text for readability.

Large input fields & buttons for ease of use.

Keyboard navigable form fields.

Catering to Users with Disabilities:

Screen-reader compatibility.

Keyboard shortcuts for navigation.

User Testing & Feedback

Testing Conducted:

Internal team testing for UI responsiveness & functionality.

Beta testing with users to gather usability feedback.

User Feedback & Improvements:

Issue: Initial UI lacked clarity in button styling → Solution: Enhanced button contrast and hover effects.

Issue: Users requested a "Clear Input" button → Solution: Added reset functionality.

4. Conclusion

The project successfully integrates real-time stock data with machine learning predictions.

The UI/UX design ensures accessibility and ease of use for traders and analysts.

The model's performance is optimized for accuracy and scalability.

Future enhancements could include additional technical indicators and real-time alerts for traders.
