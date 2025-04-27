# Real Estate Investment AI

This project provides an end-to-end pipeline for identifying, evaluating, and simulating real estate investment opportunities using Python and PyTorch. It supports automated investing and portfolio tracking and can be run seamlessly on Google Colab.

## Features
- Data ingestion and preprocessing of property listings  
- Neural network models for property valuation and rental estimation  
- Cash flow, ROI, and portfolio simulation over time  
- Automated investment strategy with customizable criteria  
- Save and load trained models and simulation results  

## Installation
1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/real-estate-ai.git
   cd real-estate-ai
   ```

3. Place your property listings CSV (e.g. `property_listings.csv`) in the project root.

## Running on Google Colab

1. Open a new Colab notebook.  
2. Upload all files under the ./Model folder
Make sure the `property_listings.csv` and `Model.py` are included in the into the Colab environment. 

3. Run the simulation:  
   ```python
   from run_simulation import run_real_estate_simulation

   ai = run_real_estate_simulation(
       initial_capital=200000,
       num_properties=5,
       min_cash_flow=100
   )
   ```
    All these settings are hyperparameters that can be tuned. 

6. View generated plots and access saved models (`*.pt`, `*.pkl`) in your workspace or Drive.

## Outputs

- **real_estate_investment_performance.png** – Performance plots  
- **real_estate_investment_results.json** – Simulation summary  
- **real_estate_ai_model_*.pt** – Saved PyTorch model weights  
- **real_estate_ai_model_*_scaler.pkl** – Saved feature scalers  
- **real_estate_ai_model_features.pkl** – Saved feature lists  

## Saving and Loading Models

- **Save:**  
  ```python
  ai.save_model(filename_prefix='real_estate_ai_model')
  ```
- **Load:**  
  ```python
  ai.load_model(filename_prefix='real_estate_ai_model')
  ```

## License

This project is licensed under the MIT License.
