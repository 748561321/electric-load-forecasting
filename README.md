# Short-Term Electric Load Forecasting

Research code for short-term electric-load forecasting. The project compares
statistical baselines and recurrent neural-network variants, and explores a
CNN-LSTM architecture with temporal/business feature engineering.

## Highlights

- Time, trend, holiday and weather-related feature construction
- ARIMA, SARIMA and Prophet baselines
- RNN, LSTM, Seq2Seq and CNN-LSTM variants
- Attention and residual-style feature extraction experiments
- MSE, MAE and MAPE evaluation utilities

The original experiment achieved a test MAPE of 1.46% on the project dataset.
That number is reported as an experiment result, not as a universal benchmark.

## Repository structure

- `src/preprocessing`: data cleaning and feature engineering
- `src/models`: deep-learning experiments
- `src/baselines`: statistical forecasting baselines
- `data/sample`: a small, anonymized example time series

## Notes

This repository is a cleaned research archive. Machine credentials, local
paths, checkpoints, generated figures and bulk intermediate data are excluded.
Some scripts are retained in their original experimental form and may require
path and parameter adjustments before execution.

## Usage

Create a Python environment with NumPy, pandas, scikit-learn, matplotlib,
statsmodels, Prophet and PyTorch as needed, then select a script under `src`.

## License

The code is provided for academic reference. The sample data may be used only
for reproducing the included demonstrations; verify the source and license
before redistributing or using it commercially.
