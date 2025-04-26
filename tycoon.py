import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime as dt
import re
import os
import warnings
from tqdm import tqdm
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')


class RealEstateAI:
    def __init__(self, data_path, initial_capital=200000, mortgage_down_payment=0.20,
                 mortgage_interest_rate=0.05, mortgage_term_years=30,
                 maintenance_cost_percentage=0.01, property_tax_rate=0.01,
                 insurance_rate=0.005, vacancy_rate=0.05, property_appreciation=0.03,
                 rental_income_growth=0.02):
        """
        Initialize the Real Estate AI with investment parameters
        """
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.mortgage_down_payment = mortgage_down_payment
        self.mortgage_interest_rate = mortgage_interest_rate
        self.mortgage_term_years = mortgage_term_years
        self.maintenance_cost_percentage = maintenance_cost_percentage
        self.property_tax_rate = property_tax_rate
        self.insurance_rate = insurance_rate
        self.vacancy_rate = vacancy_rate
        self.property_appreciation = property_appreciation
        self.rental_income_growth = rental_income_growth

        # Portfolio tracking
        self.portfolio = []
        self.monthly_income = 0
        self.monthly_expenses = 0
        self.monthly_cash_flow = 0
        self.net_worth = initial_capital
        self.total_equity = 0
        self.total_debt = 0

        # Track purchased property IDs to avoid duplicates
        self.purchased_property_ids = set()

        # Performance history
        self.history = {
            'date': [],
            'capital': [],
            'net_worth': [],
            'monthly_cash_flow': [],
            'num_properties': [],
            'total_equity': [],
            'total_debt': [],
            'roi': []
        }

        # Load and preprocess the data
        self.load_data(data_path)

        # Build valuation and rental prediction models with PyTorch
        self.build_models()

        # Current date for simulation
        self.current_date = dt.datetime.now()

        print(f"Real Estate Investment AI initialized with ${initial_capital:,.2f} capital")

    def load_data(self, data_path):
        """Load and preprocess the property data"""
        print("Loading property data...")

        # Read the CSV file
        self.raw_data = pd.read_csv(data_path)

        # Basic data cleaning
        self.data = self.raw_data.copy()

        # Extract key columns
        # Standardize column names
        if self.data.shape[1] >= 20:  # Check if we have enough columns
            self.data.columns = [
                                    'id', 'price', 'status', 'property_type', 'listing_date', 'address', 'city',
                                    'state', 'zip', 'county', 'year_built', 'sqft', 'sqft_unit', 'lot_size',
                                    'stories', 'bedrooms', 'views', 'favorites', 'ratio', 'days_on_market',
                                    'status_text', 'url', 'timestamp'
                                ][:self.data.shape[1]]  # Trim to actual number of columns

        # Clean up data types
        # Convert price to numeric
        self.data['price'] = pd.to_numeric(self.data['price'], errors='coerce')

        # Handle missing values
        self.data['year_built'] = pd.to_numeric(self.data['year_built'], errors='coerce')
        current_year = dt.datetime.now().year
        # Fill missing year_built with median by zip code, or overall median
        try:
            median_year = self.data.groupby('zip')['year_built'].transform(lambda x: x.median())
            self.data['year_built'] = self.data['year_built'].fillna(median_year)
        except:
            pass
        self.data['year_built'] = self.data['year_built'].fillna(self.data['year_built'].median())

        # Extract numeric square footage
        self.data['sqft'] = pd.to_numeric(self.data['sqft'], errors='coerce')

        # Convert views and favorites to numeric
        if 'views' in self.data.columns:
            self.data['views'] = pd.to_numeric(self.data['views'], errors='coerce')
        if 'favorites' in self.data.columns:
            self.data['favorites'] = pd.to_numeric(self.data['favorites'], errors='coerce')

        # Fill missing bedrooms with median by property type
        if 'bedrooms' in self.data.columns:
            self.data['bedrooms'] = pd.to_numeric(self.data['bedrooms'], errors='coerce')
            try:
                median_bedrooms = self.data.groupby('property_type')['bedrooms'].transform(lambda x: x.median())
                self.data['bedrooms'] = self.data['bedrooms'].fillna(median_bedrooms)
            except:
                pass
            self.data['bedrooms'] = self.data['bedrooms'].fillna(self.data['bedrooms'].median())

        # Convert listing_date to datetime
        if 'listing_date' in self.data.columns:
            self.data['listing_date'] = pd.to_datetime(self.data['listing_date'], errors='coerce')

        # Calculate property age
        self.data['age'] = current_year - self.data['year_built']

        # Calculate price per square foot
        self.data['price_per_sqft'] = self.data['price'] / self.data['sqft']

        # Filter to only include properties for sale
        if 'status' in self.data.columns:
            try:
                self.data = self.data[self.data['status'] == 'For Sale']
            except:
                pass

        # Extract numeric days on market values from strings
        if 'days_on_market' in self.data.columns:
            def extract_days_on_market(dom_str):
                """Convert days on market string to numeric value"""
                if pd.isna(dom_str) or not isinstance(dom_str, str):
                    return np.nan

                # Extract first number from string
                match = re.search(r'(\d+)', str(dom_str))
                if match:
                    num = float(match.group(1))
                    # Convert to approximate days based on unit
                    if 'minute' in dom_str.lower():
                        return num / (24 * 60)  # minutes to days
                    elif 'hour' in dom_str.lower():
                        return num / 24  # hours to days
                    else:
                        return num  # already in days or no unit specified
                return np.nan

            # Convert days_on_market to numeric
            self.data['days_on_market_numeric'] = self.data['days_on_market'].apply(extract_days_on_market)

        # Remove extreme outliers for price and square footage
        self.data = self.data[(self.data['price'] > 50000) & (self.data['price'] < 2000000)]
        self.data = self.data[(self.data['sqft'] > 500) & (self.data['sqft'] < 10000)]

        # Focus on residential properties if possible
        if 'property_type' in self.data.columns:
            residential_types = ['Single Family', 'Townhouse', 'Condo', 'Multi Family']
            try:
                self.data = self.data[self.data['property_type'].isin(residential_types)]
            except:
                pass

        # Drop rows with missing critical values
        critical_columns = ['price', 'sqft']
        if 'bedrooms' in self.data.columns:
            critical_columns.append('bedrooms')
        if 'zip' in self.data.columns:
            critical_columns.append('zip')

        self.data = self.data.dropna(subset=critical_columns)

        # Create a feature for zip code popularity (by listing count)
        if 'zip' in self.data.columns:
            zip_counts = self.data['zip'].value_counts().to_dict()
            self.data['zip_popularity'] = self.data['zip'].map(zip_counts)

        # Calculate zip code averages using the numeric version of days on market
        if 'zip' in self.data.columns and 'days_on_market_numeric' in self.data.columns:
            try:
                zip_dom = self.data.groupby('zip')['days_on_market_numeric'].mean().to_dict()
                self.data['zip_avg_dom'] = self.data['zip'].map(zip_dom)
            except:
                self.data['zip_avg_dom'] = self.data['days_on_market_numeric'].mean()

        # Create a feature for average price by zip code
        if 'zip' in self.data.columns:
            try:
                zip_price = self.data.groupby('zip')['price'].mean().to_dict()
                self.data['zip_avg_price'] = self.data['zip'].map(zip_price)
            except:
                self.data['zip_avg_price'] = self.data['price'].mean()

        # Calculate market interest indicators
        if all(col in self.data.columns for col in ['views', 'favorites', 'days_on_market_numeric']):
            self.data['interest_ratio'] = (self.data['views'] + self.data['favorites']) / (
                        self.data['days_on_market_numeric'] + 1)

        # Create features for market analysis
        if all(col in self.data.columns for col in ['price', 'zip_avg_price']):
            self.data['price_to_zip_ratio'] = self.data['price'] / self.data['zip_avg_price']

        # Add features for location-based analysis
        if 'state' in self.data.columns:
            state_counts = self.data['state'].value_counts().to_dict()
            self.data['state_popularity'] = self.data['state'].map(state_counts)

        # Fill any remaining NaNs in the features we'll use for modeling
        for col in self.data.columns:
            if self.data[col].dtype == 'float64' or self.data[col].dtype == 'int64':
                self.data[col] = self.data[col].fillna(self.data[col].median())

        print(f"Processed {len(self.data)} property listings")

    def build_models(self):
        """Build PyTorch models for property valuation and rental estimation"""
        print("Building predictive models with PyTorch...")

        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Define features for valuation model based on available columns
        valuation_features = ['sqft', 'year_built', 'age', 'price_per_sqft']

        # Add additional features if available
        if 'bedrooms' in self.data.columns:
            valuation_features.append('bedrooms')
        if 'zip_popularity' in self.data.columns:
            valuation_features.append('zip_popularity')
        if 'zip_avg_dom' in self.data.columns:
            valuation_features.append('zip_avg_dom')
        if 'zip_avg_price' in self.data.columns:
            valuation_features.append('zip_avg_price')
        if 'price_to_zip_ratio' in self.data.columns:
            valuation_features.append('price_to_zip_ratio')
        if 'interest_ratio' in self.data.columns:
            valuation_features.append('interest_ratio')

        # Ensure all these columns exist and have valid data
        valuation_features = [f for f in valuation_features if f in self.data.columns]

        # Prepare data for valuation model
        X = self.data[valuation_features].copy()
        y = self.data['price']

        # Handle any remaining NaN values
        X = X.fillna(X.mean())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        self.valuation_scaler = StandardScaler()
        X_train_scaled = self.valuation_scaler.fit_transform(X_train)
        X_test_scaled = self.valuation_scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(self.device)

        # Create DataLoader for batch processing
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Define neural network for valuation
        class ValuationModel(nn.Module):
            def __init__(self, input_size):
                super(ValuationModel, self).__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                return x

        # Initialize model, loss function, and optimizer
        input_size = X_train_scaled.shape[1]
        self.valuation_nn = ValuationModel(input_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.valuation_nn.parameters(), lr=0.001, weight_decay=1e-5)

        # Train the valuation model
        num_epochs = 100
        for epoch in range(num_epochs):
            self.valuation_nn.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = self.valuation_nn(batch_X).squeeze()
                loss = criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

        # Evaluate valuation model
        self.valuation_nn.eval()
        with torch.no_grad():
            val_predictions = self.valuation_nn(X_test_tensor).squeeze().cpu().numpy()
            val_mae = np.mean(np.abs(val_predictions - y_test.values))
            val_r2 = 1 - np.sum((y_test.values - val_predictions) ** 2) / np.sum((y_test.values - y_test.mean()) ** 2)

        print(f"Valuation model MAE: ${val_mae:.2f}, R²: {val_r2:.4f}")

        # Build rental estimation model
        # Assuming 0.8% of property value is typical monthly rent (rule of thumb)
        self.data['estimated_rent'] = self.data['price'] * 0.008

        # Features for rental model
        rental_features = ['price', 'sqft', 'year_built', 'age']

        # Add additional features if available
        if 'bedrooms' in self.data.columns:
            rental_features.append('bedrooms')
        if 'zip_popularity' in self.data.columns:
            rental_features.append('zip_popularity')
        if 'zip_avg_dom' in self.data.columns:
            rental_features.append('zip_avg_dom')
        if 'zip_avg_price' in self.data.columns:
            rental_features.append('zip_avg_price')

        # Ensure all these columns exist
        rental_features = [f for f in rental_features if f in self.data.columns]

        # Prepare data for rental model
        X_rent = self.data[rental_features].copy()
        y_rent = self.data['estimated_rent']

        # Handle NaN values
        X_rent = X_rent.fillna(X_rent.mean())

        # Split data
        X_rent_train, X_rent_test, y_rent_train, y_rent_test = train_test_split(
            X_rent, y_rent, test_size=0.2, random_state=42
        )

        # Scale the features
        self.rental_scaler = StandardScaler()
        X_rent_train_scaled = self.rental_scaler.fit_transform(X_rent_train)
        X_rent_test_scaled = self.rental_scaler.transform(X_rent_test)

        # Convert to PyTorch tensors
        X_rent_train_tensor = torch.tensor(X_rent_train_scaled, dtype=torch.float32).to(self.device)
        y_rent_train_tensor = torch.tensor(y_rent_train.values, dtype=torch.float32).to(self.device)
        X_rent_test_tensor = torch.tensor(X_rent_test_scaled, dtype=torch.float32).to(self.device)
        y_rent_test_tensor = torch.tensor(y_rent_test.values, dtype=torch.float32).to(self.device)

        # Create DataLoader for batch processing
        rent_dataset = TensorDataset(X_rent_train_tensor, y_rent_train_tensor)
        rent_loader = DataLoader(rent_dataset, batch_size=64, shuffle=True)

        # Define neural network for rental estimation
        class RentalModel(nn.Module):
            def __init__(self, input_size):
                super(RentalModel, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 16)
                self.fc4 = nn.Linear(16, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                return x

        # Initialize model, loss function, and optimizer
        rental_input_size = X_rent_train_scaled.shape[1]
        self.rental_nn = RentalModel(rental_input_size).to(self.device)
        rental_criterion = nn.MSELoss()
        rental_optimizer = optim.AdamW(self.rental_nn.parameters(), lr=0.001, weight_decay=1e-5)

        # Train the rental model
        num_epochs = 100
        for epoch in range(num_epochs):
            self.rental_nn.train()
            total_loss = 0
            for batch_X, batch_y in rent_loader:
                # Forward pass
                outputs = self.rental_nn(batch_X).squeeze()
                loss = rental_criterion(outputs, batch_y)

                # Backward and optimize
                rental_optimizer.zero_grad()
                loss.backward()
                rental_optimizer.step()

                total_loss += loss.item()

            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f'Rental Model Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(rent_loader):.4f}')

        # Evaluate rental model
        self.rental_nn.eval()
        with torch.no_grad():
            rent_predictions = self.rental_nn(X_rent_test_tensor).squeeze().cpu().numpy()
            rent_mae = np.mean(np.abs(rent_predictions - y_rent_test.values))
            rent_r2 = 1 - np.sum((y_rent_test.values - rent_predictions) ** 2) / np.sum(
                (y_rent_test.values - y_rent_test.mean()) ** 2)

        print(f"Rental model MAE: ${rent_mae:.2f}, R²: {rent_r2:.4f}")

        # Create wrapper classes for models to make them compatible with the rest of the code
        class TorchModelWrapper:
            def __init__(self, model, scaler, device):
                self.model = model
                self.scaler = scaler
                self.device = device

            def predict(self, X):
                # Explicitly set model to eval mode to disable dropout during inference
                self.model.eval()

                # Handle both DataFrame and NumPy array inputs
                if isinstance(X, pd.DataFrame):
                    X_values = X.values
                else:
                    X_values = X

                X_scaled = self.scaler.transform(X_values)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    predictions = self.model(X_tensor).cpu().numpy()

                # FIX: Handle scalar results properly and convert to Python float
                if predictions.ndim == 0:  # It's a scalar
                    return np.array([float(predictions)])  # Return as 1D array with Python float

                # Convert to Python float to avoid float32 JSON issues
                return np.array([float(x) for x in predictions.reshape(-1)])

        # Create wrappers for both models
        self.valuation_model = TorchModelWrapper(self.valuation_nn, self.valuation_scaler, self.device)
        self.rental_model = TorchModelWrapper(self.rental_nn, self.rental_scaler, self.device)

        # Store the feature names for later use
        self.valuation_features = valuation_features
        self.rental_features = rental_features

    def estimate_property_value(self, property_data):
        """Estimate the fair market value of a property"""
        # Ensure all required features are present
        for feature in self.valuation_features:
            if feature not in property_data:
                # If a feature is missing, use the median value from the training data
                property_data[feature] = self.data[feature].median()

        # Prepare property data for prediction
        property_vector = property_data[self.valuation_features].copy().values.reshape(1, -1)

        # Predict property value using our PyTorch wrapper
        predictions = self.valuation_model.predict(property_vector)

        # FIX: Safely get the first element
        estimated_value = predictions[0] if predictions.size > 0 else predictions

        return estimated_value

    def estimate_rental_income(self, property_data):
        """Estimate the monthly rental income for a property"""
        # Ensure all required features are present
        for feature in self.rental_features:
            if feature not in property_data:
                # If a feature is missing, use the median value from the training data
                property_data[feature] = self.data[feature].median()

        # Prepare property data for prediction
        property_vector = property_data[self.rental_features].copy().values.reshape(1, -1)

        # Predict monthly rent
        predictions = self.rental_model.predict(property_vector)

        # FIX: Safely get the first element
        estimated_rent = predictions[0] if predictions.size > 0 else predictions

        return estimated_rent

    def calculate_mortgage_payment(self, property_price):
        """Calculate the monthly mortgage payment for a property"""
        loan_amount = property_price * (1 - self.mortgage_down_payment)
        monthly_interest_rate = self.mortgage_interest_rate / 12
        num_payments = self.mortgage_term_years * 12

        # Calculate monthly mortgage payment using the standard formula
        if monthly_interest_rate == 0:
            monthly_payment = loan_amount / num_payments
        else:
            monthly_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** num_payments) / (
                        (1 + monthly_interest_rate) ** num_payments - 1)

        return monthly_payment

    def calculate_monthly_expenses(self, property_price):
        """Calculate the monthly expenses for a property"""
        # Mortgage payment
        mortgage_payment = self.calculate_mortgage_payment(property_price)

        # Property tax (annual amount divided by 12)
        monthly_property_tax = (property_price * self.property_tax_rate) / 12

        # Insurance (annual amount divided by 12)
        monthly_insurance = (property_price * self.insurance_rate) / 12

        # Maintenance (annual amount divided by 12)
        monthly_maintenance = (property_price * self.maintenance_cost_percentage) / 12

        # Total monthly expenses
        total_monthly_expenses = mortgage_payment + monthly_property_tax + monthly_insurance + monthly_maintenance

        return {
            'mortgage': mortgage_payment,
            'property_tax': monthly_property_tax,
            'insurance': monthly_insurance,
            'maintenance': monthly_maintenance,
            'total': total_monthly_expenses
        }

    def calculate_cash_flow(self, property_data):
        """Calculate the monthly cash flow for a property"""
        property_price = property_data['price']

        # Estimate rental income
        gross_monthly_rent = self.estimate_rental_income(property_data)

        # Account for vacancy
        effective_monthly_rent = gross_monthly_rent * (1 - self.vacancy_rate)

        # Calculate expenses
        monthly_expenses = self.calculate_monthly_expenses(property_price)

        # Calculate cash flow
        monthly_cash_flow = effective_monthly_rent - monthly_expenses['total']

        return {
            'gross_rent': gross_monthly_rent,
            'effective_rent': effective_monthly_rent,
            'expenses': monthly_expenses,
            'cash_flow': monthly_cash_flow
        }

    def calculate_roi(self, property_data):
        """Calculate the projected ROI for a property"""
        property_price = property_data['price']

        # Down payment
        down_payment = property_price * self.mortgage_down_payment

        # Closing costs (estimated at 3% of purchase price)
        closing_costs = property_price * 0.03

        # Initial investment
        initial_investment = down_payment + closing_costs

        # Annual cash flow
        cash_flow = self.calculate_cash_flow(property_data)
        annual_cash_flow = cash_flow['cash_flow'] * 12

        # First-year ROI
        cash_on_cash_roi = (annual_cash_flow / initial_investment) * 100

        # 5-year ROI including appreciation
        property_value_5yr = property_price * (1 + self.property_appreciation) ** 5
        equity_5yr = property_value_5yr - (
                    property_price * (1 - self.mortgage_down_payment) * 0.9)  # Assuming 10% of loan paid off

        # 5-year accumulated cash flow
        accumulated_cash_flow = 0
        for year in range(1, 6):
            # Increase rent annually by rental_income_growth
            annual_rent = cash_flow['gross_rent'] * 12 * (1 + self.rental_income_growth) ** (year - 1)
            annual_effective_rent = annual_rent * (1 - self.vacancy_rate)

            # Expenses stay relatively constant (mortgage is fixed, others rise with inflation but it's simplified)
            annual_expenses = cash_flow['expenses']['total'] * 12

            # Annual cash flow for this year
            year_cash_flow = annual_effective_rent - annual_expenses
            accumulated_cash_flow += year_cash_flow

        # Total return over 5 years
        total_return = (equity_5yr + accumulated_cash_flow - initial_investment) / initial_investment * 100

        return {
            'initial_investment': initial_investment,
            'annual_cash_flow': annual_cash_flow,
            'cash_on_cash_roi': cash_on_cash_roi,
            'five_year_roi': total_return,
            'projected_value_5yr': property_value_5yr,
            'projected_equity_5yr': equity_5yr,
            'accumulated_cash_flow_5yr': accumulated_cash_flow
        }

    def identify_investment_opportunities(self, top_n=10, min_bedrooms=1, max_price=None, min_cash_flow=0):
        """Identify the top investment opportunities based on ROI"""
        if max_price is None:
            # Set max price based on available capital (considering down payment)
            max_price = self.available_capital / self.mortgage_down_payment

        # Initial filtering of properties
        potential_investments = self.data.copy()

        # Apply bedroom filter if column exists
        if 'bedrooms' in potential_investments.columns:
            potential_investments = potential_investments[potential_investments['bedrooms'] >= min_bedrooms]

        # Apply price filter
        potential_investments = potential_investments[potential_investments['price'] <= max_price]

        # FIXED: Filter out already purchased properties
        if len(self.purchased_property_ids) > 0:
            potential_investments = potential_investments[
                ~potential_investments['id'].isin(self.purchased_property_ids)]

        if len(potential_investments) == 0:
            print("No properties match your investment criteria.")
            return pd.DataFrame()

        # Calculate metrics for each property
        results = []

        for idx, property_data in tqdm(potential_investments.iterrows(), total=len(potential_investments),
                                       desc="Analyzing properties"):
            try:
                # Calculate cash flow
                cash_flow = self.calculate_cash_flow(property_data)

                # Skip properties with negative cash flow if minimum is set
                if cash_flow['cash_flow'] < min_cash_flow:
                    continue

                # Calculate ROI
                roi = self.calculate_roi(property_data)

                # Estimated market value
                estimated_value = self.estimate_property_value(property_data)

                # Value ratio (estimated value / price)
                value_ratio = estimated_value / property_data['price']

                # Property information
                property_info = {
                    'id': property_data['id'],
                    'price': property_data['price'],
                    'sqft': property_data['sqft'],
                    'year_built': property_data['year_built'],
                    'estimated_value': estimated_value,
                    'value_ratio': value_ratio,
                    'monthly_rent': cash_flow['gross_rent'],
                    'monthly_cash_flow': cash_flow['cash_flow'],
                    'cash_on_cash_roi': roi['cash_on_cash_roi'],
                    'five_year_roi': roi['five_year_roi'],
                    'investment_score': roi['cash_on_cash_roi'] * value_ratio  # Combined score
                }

                # Add location information if available
                for field in ['address', 'city', 'state', 'zip']:
                    if field in property_data:
                        property_info[field] = property_data[field]

                # Add bedrooms if available
                if 'bedrooms' in property_data:
                    property_info['bedrooms'] = property_data['bedrooms']

                results.append(property_info)
            except Exception as e:
                # Skip properties that cause errors without crash
                continue

        # Convert to DataFrame
        opportunities = pd.DataFrame(results)

        if len(opportunities) == 0:
            print("No valid investment opportunities found.")
            return pd.DataFrame()

        # Sort by investment score (descending)
        opportunities = opportunities.sort_values('investment_score', ascending=False)

        # Return top N opportunities
        return opportunities.head(top_n)

    def purchase_property(self, property_id):
        """Purchase a property and add it to the portfolio"""
        # Check if property was already purchased
        if property_id in self.purchased_property_ids:
            print(f"Property ID {property_id} already in portfolio. Cannot purchase again.")
            return False

        # Find the property in the dataset
        property_data = self.data[self.data['id'] == property_id]

        if len(property_data) == 0:
            print(f"Property ID {property_id} not found in dataset.")
            return False

        property_data = property_data.iloc[0].to_dict()

        # Check if we have enough capital for down payment
        required_capital = property_data['price'] * self.mortgage_down_payment
        required_capital += property_data['price'] * 0.03  # Closing costs

        if required_capital > self.available_capital:
            print(f"Insufficient capital. Need ${required_capital:,.2f}, have ${self.available_capital:,.2f}")
            return False

        # Calculate mortgage and financial details
        cash_flow = self.calculate_cash_flow(pd.Series(property_data))
        roi = self.calculate_roi(pd.Series(property_data))

        # Create property investment object
        purchase_date = self.current_date
        property_investment = {
            'id': property_data['id'],
            'purchase_price': float(property_data['price']),  # Convert to Python float
            'purchase_date': purchase_date,
            'sqft': float(property_data['sqft']),  # Convert to Python float
            'year_built': float(property_data['year_built']),  # Convert to Python float
            'current_value': float(property_data['price']),  # Convert to Python float
            'initial_investment': float(roi['initial_investment']),  # Convert to Python float
            'loan_amount': float(property_data['price'] * (1 - self.mortgage_down_payment)),  # Convert to Python float
            'current_loan_balance': float(property_data['price'] * (1 - self.mortgage_down_payment)),
            # Convert to Python float
            'monthly_rent': float(cash_flow['gross_rent']),  # Convert to Python float
            'monthly_expenses': float(cash_flow['expenses']['total']),  # Convert to Python float
            'monthly_cash_flow': float(cash_flow['cash_flow']),  # Convert to Python float
            'roi': float(roi['cash_on_cash_roi']),  # Convert to Python float
            'equity': float(property_data['price'] * self.mortgage_down_payment)  # Convert to Python float
        }

        # Add location information if available
        for field in ['address', 'city', 'state', 'zip']:
            if field in property_data:
                property_investment[field] = property_data[field]

        # Add bedrooms if available
        if 'bedrooms' in property_data:
            property_investment['bedrooms'] = property_data['bedrooms']

        # Update capital
        self.available_capital -= roi['initial_investment']

        # Add to portfolio
        self.portfolio.append(property_investment)

        # Add to purchased properties set to prevent duplicate purchases
        self.purchased_property_ids.add(property_id)

        # Update monthly financials
        self.monthly_income += cash_flow['effective_rent']
        self.monthly_expenses += cash_flow['expenses']['total']
        self.monthly_cash_flow = self.monthly_income - self.monthly_expenses

        # Update debt and equity
        self.total_debt += property_investment['loan_amount']
        self.total_equity += property_investment['purchase_price'] - property_investment['loan_amount']

        # Update net worth
        self.net_worth = self.available_capital + self.total_equity

        location_str = ""
        if 'address' in property_data and 'city' in property_data and 'state' in property_data:
            location_str = f" at {property_data['address']}, {property_data['city']}, {property_data['state']}"

        print(f"Purchased property{location_str} for ${property_data['price']:,.2f}")

        return True

    def sell_property(self, property_id, sale_price=None):
        """Sell a property from the portfolio"""
        # Find the property in the portfolio
        property_index = None
        for i, prop in enumerate(self.portfolio):
            if prop['id'] == property_id:
                property_index = i
                break

        if property_index is None:
            print(f"Property ID {property_id} not found in portfolio.")
            return False

        property_investment = self.portfolio[property_index]

        # Calculate sale price if not provided
        if sale_price is None:
            # Calculate appreciation based on time owned
            months_owned = (self.current_date - property_investment['purchase_date']).days / 30
            years_owned = months_owned / 12

            # Calculate appreciated value
            sale_price = property_investment['purchase_price'] * (1 + self.property_appreciation) ** years_owned

        # Calculate profit or loss
        loan_payoff = property_investment['current_loan_balance']
        closing_costs = sale_price * 0.06  # Estimated selling costs (agent commission, etc.)

        net_proceeds = sale_price - loan_payoff - closing_costs

        # Update capital
        self.available_capital += net_proceeds

        # Update monthly financials
        self.monthly_income -= property_investment['monthly_rent'] * (1 - self.vacancy_rate)
        self.monthly_expenses -= property_investment['monthly_expenses']
        self.monthly_cash_flow = self.monthly_income - self.monthly_expenses

        # Update debt and equity
        self.total_debt -= property_investment['current_loan_balance']
        self.total_equity -= (property_investment['current_value'] - property_investment['current_loan_balance'])

        # Remove from portfolio
        self.portfolio.pop(property_index)

        # Remove from purchased property IDs (so it can be purchased again if desired)
        self.purchased_property_ids.remove(property_investment['id'])

        # Update net worth
        self.net_worth = self.available_capital + self.total_equity

        location_str = ""
        if 'address' in property_investment and 'city' in property_investment and 'state' in property_investment:
            location_str = f" at {property_investment['address']}, {property_investment['city']}, {property_investment['state']}"

        print(f"Sold property{location_str} for ${sale_price:,.2f}")
        print(f"Net proceeds: ${net_proceeds:,.2f}")

        return True

    def update_portfolio(self, months=1):
        """Update the portfolio over a specified number of months"""
        print(f"Updating portfolio over {months} months...")

        # Update simulation date
        self.current_date = self.current_date + dt.timedelta(days=30 * months)

        # Calculate income from all properties for the period
        period_income = self.monthly_income * months
        period_expenses = self.monthly_expenses * months
        period_cash_flow = period_income - period_expenses

        # Add cash flow to available capital
        self.available_capital += period_cash_flow

        # Update each property
        for prop in self.portfolio:
            # Update loan balance (simplified - just reduce by a portion of the payment)
            mortgage_payment = self.calculate_mortgage_payment(prop['purchase_price'])
            interest_portion = prop['current_loan_balance'] * (self.mortgage_interest_rate / 12)
            principal_reduction = (mortgage_payment - interest_portion) * months
            prop['current_loan_balance'] -= principal_reduction

            # Increase property value due to appreciation
            annual_appreciation = prop['current_value'] * self.property_appreciation
            monthly_appreciation = annual_appreciation / 12
            prop['current_value'] += monthly_appreciation * months

            # Increase rent due to growth
            annual_rent_growth = prop['monthly_rent'] * self.rental_income_growth
            monthly_rent_growth = annual_rent_growth / 12
            prop['monthly_rent'] += monthly_rent_growth * months

            # Update cash flow
            prop['monthly_cash_flow'] = prop['monthly_rent'] * (1 - self.vacancy_rate) - prop['monthly_expenses']

            # Update equity
            prop['equity'] = prop['current_value'] - prop['current_loan_balance']

            # Update ROI
            prop['roi'] = (prop['monthly_cash_flow'] * 12) / prop['initial_investment'] * 100

        # Recalculate portfolio metrics
        self.total_equity = sum(prop['equity'] for prop in self.portfolio)
        self.total_debt = sum(prop['current_loan_balance'] for prop in self.portfolio)

        # Update income and expenses
        self.monthly_income = sum(prop['monthly_rent'] * (1 - self.vacancy_rate) for prop in self.portfolio)
        self.monthly_expenses = sum(prop['monthly_expenses'] for prop in self.portfolio)
        self.monthly_cash_flow = self.monthly_income - self.monthly_expenses

        # Update net worth
        self.net_worth = self.available_capital + self.total_equity

        # Calculate portfolio ROI
        initial_investment = sum(prop['initial_investment'] for prop in self.portfolio)
        if initial_investment > 0:
            portfolio_roi = (self.monthly_cash_flow * 12) / initial_investment * 100
        else:
            portfolio_roi = 0

        # Record history
        self.history['date'].append(self.current_date)
        self.history['capital'].append(float(self.available_capital))  # Convert to Python float
        self.history['net_worth'].append(float(self.net_worth))  # Convert to Python float
        self.history['monthly_cash_flow'].append(float(self.monthly_cash_flow))  # Convert to Python float
        self.history['num_properties'].append(len(self.portfolio))
        self.history['total_equity'].append(float(self.total_equity))  # Convert to Python float
        self.history['total_debt'].append(float(self.total_debt))  # Convert to Python float
        self.history['roi'].append(float(portfolio_roi))  # Convert to Python float

        print(f"Portfolio updated to {self.current_date.strftime('%Y-%m-%d')}")
        print(f"Available Capital: ${self.available_capital:,.2f}")
        print(f"Net Worth: ${self.net_worth:,.2f}")
        print(f"Monthly Cash Flow: ${self.monthly_cash_flow:,.2f}")
        print(f"Portfolio ROI: {portfolio_roi:.2f}%")

        return {
            'date': self.current_date,
            'capital': float(self.available_capital),  # Convert to Python float
            'net_worth': float(self.net_worth),  # Convert to Python float
            'monthly_cash_flow': float(self.monthly_cash_flow),  # Convert to Python float
            'num_properties': len(self.portfolio),
            'total_equity': float(self.total_equity),  # Convert to Python float
            'total_debt': float(self.total_debt),  # Convert to Python float
            'roi': float(portfolio_roi)  # Convert to Python float
        }

    def auto_invest(self, max_properties=5, min_cash_reserve=20000, months_between_purchases=3, min_cash_flow=0):
        """Automatically invest in properties over time"""
        print(f"Starting auto-investment strategy:")
        print(f"- Maximum properties to acquire: {max_properties}")
        print(f"- Minimum cash reserve: ${min_cash_reserve:,.2f}")
        print(f"- Months between purchases: {months_between_purchases}")
        print(f"- Minimum monthly cash flow: ${min_cash_flow:,.2f}")

        properties_acquired = 0
        simulation_months = 0
        max_simulation_months = 60  # 5 years

        # Initialize performance history for plotting
        performance_history = []

        print("\nStarting investment simulation...\n")

        while properties_acquired < max_properties and simulation_months < max_simulation_months:
            # Record current state
            current_state = {
                'month': simulation_months,
                'date': self.current_date,
                'capital': float(self.available_capital),  # Convert to Python float
                'net_worth': float(self.net_worth),  # Convert to Python float
                'monthly_cash_flow': float(self.monthly_cash_flow),  # Convert to Python float
                'num_properties': len(self.portfolio),
                'portfolio_roi': float(self.history['roi'][-1]) if len(self.history['roi']) > 0 else 0
                # Convert to Python float
            }
            performance_history.append(current_state)

            # Check if we should buy a property
            if simulation_months % months_between_purchases == 0:
                print(f"\nMonth {simulation_months}: Evaluating investment opportunities...")

                # Calculate max price based on available capital
                max_price = (self.available_capital - min_cash_reserve) / self.mortgage_down_payment

                if max_price <= 0:
                    print(f"Insufficient capital for new investments. Continuing to build reserves.")
                else:
                    # Find investment opportunities
                    opportunities = self.identify_investment_opportunities(
                        top_n=5,
                        max_price=max_price,
                        min_cash_flow=min_cash_flow
                    )

                    if len(opportunities) > 0:
                        # Select the best opportunity
                        best_property = opportunities.iloc[0]

                        print(f"Best opportunity found:")
                        location_str = ""
                        if all(field in best_property for field in ['address', 'city', 'state']):
                            location_str = f"  Address: {best_property['address']}, {best_property['city']}, {best_property['state']}"
                            print(location_str)
                        print(f"  Price: ${best_property['price']:,.2f}")
                        print(f"  Est. Monthly Cash Flow: ${best_property['monthly_cash_flow']:,.2f}")
                        print(f"  Cash-on-Cash ROI: {best_property['cash_on_cash_roi']:.2f}%")

                        # Purchase the property
                        success = self.purchase_property(best_property['id'])

                        if success:
                            properties_acquired += 1
                            print(f"Successfully acquired property {properties_acquired} of {max_properties}")
                        else:
                            print("Purchase unsuccessful. Will try again later.")
                    else:
                        print("No suitable investment opportunities found this month.")

            # Update portfolio for next month
            self.update_portfolio(months=1)
            simulation_months += 1

            # Print progress
            print(f"\nMonth {simulation_months} summary:")
            print(f"  Properties owned: {len(self.portfolio)}")
            print(f"  Available capital: ${self.available_capital:,.2f}")
            print(f"  Net worth: ${self.net_worth:,.2f}")
            print(f"  Monthly cash flow: ${self.monthly_cash_flow:,.2f}")
            print("-" * 50)

        # Final record
        current_state = {
            'month': simulation_months,
            'date': self.current_date,
            'capital': float(self.available_capital),  # Convert to Python float
            'net_worth': float(self.net_worth),  # Convert to Python float
            'monthly_cash_flow': float(self.monthly_cash_flow),  # Convert to Python float
            'num_properties': len(self.portfolio),
            'portfolio_roi': float(self.history['roi'][-1]) if len(self.history['roi']) > 0 else 0
            # Convert to Python float
        }
        performance_history.append(current_state)

        # Plot performance
        self.plot_investment_performance(performance_history)

        # Print final results
        self.print_investment_summary()

        return performance_history

    def plot_investment_performance(self, performance_history):
        """Plot the investment performance over time"""
        # Convert to DataFrame
        performance_df = pd.DataFrame(performance_history)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Real Estate Investment Performance', fontsize=16)

        # Net Worth Over Time
        axes[0, 0].plot(performance_df['month'], performance_df['net_worth'], marker='o', linestyle='-', color='blue')
        axes[0, 0].set_title('Net Worth Over Time')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Net Worth ($)')
        axes[0, 0].grid(True)

        # Monthly Cash Flow
        axes[0, 1].plot(performance_df['month'], performance_df['monthly_cash_flow'], marker='o', linestyle='-',
                        color='green')
        axes[0, 1].set_title('Monthly Cash Flow')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Cash Flow ($)')
        axes[0, 1].grid(True)

        # Number of Properties
        axes[1, 0].plot(performance_df['month'], performance_df['num_properties'], marker='o', linestyle='-',
                        color='red')
        axes[1, 0].set_title('Number of Properties')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Properties')
        axes[1, 0].grid(True)

        # Portfolio ROI
        axes[1, 1].plot(performance_df['month'], performance_df['portfolio_roi'], marker='o', linestyle='-',
                        color='purple')
        axes[1, 1].set_title('Portfolio ROI')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('ROI (%)')
        axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('real_estate_investment_performance.png')
        plt.show()

    def print_investment_summary(self):
        """Print a summary of the investment results"""
        print("\n" + "=" * 60)
        print("REAL ESTATE INVESTMENT SUMMARY")
        print("=" * 60)

        # Portfolio summary
        print("\nPORTFOLIO SUMMARY:")
        print(f"Starting Capital: ${self.initial_capital:,.2f}")
        print(f"Current Capital: ${self.available_capital:,.2f}")
        print(f"Current Net Worth: ${self.net_worth:,.2f}")
        print(f"Total Equity: ${self.total_equity:,.2f}")
        print(f"Total Debt: ${self.total_debt:,.2f}")
        print(f"Monthly Cash Flow: ${self.monthly_cash_flow:,.2f}")
        print(f"Annual Cash Flow: ${self.monthly_cash_flow * 12:,.2f}")

        # Calculate overall return
        total_gain = self.net_worth - self.initial_capital
        total_return_pct = (total_gain / self.initial_capital) * 100

        simulation_years = (self.current_date - dt.datetime.now()).days / 365
        annualized_return = ((1 + total_return_pct / 100) ** (
                    1 / simulation_years) - 1) * 100 if simulation_years > 0 else 0

        print(f"\nRETURN ON INVESTMENT:")
        print(f"Total Gain: ${total_gain:,.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")
        print(f"Simulation Period: {simulation_years:.2f} years")
        print(f"Annualized Return: {annualized_return:.2f}%")

        # Property listing
        print("\nPROPERTY PORTFOLIO:")
        print("-" * 100)

        if len(self.portfolio) > 0:
            # Check which fields are available in all properties
            properties_df = pd.DataFrame(self.portfolio)

            # Define the columns to include in the report
            display_columns = ['id']

            # Add location information if available
            for field in ['address', 'city', 'state', 'zip']:
                if field in properties_df.columns:
                    display_columns.append(field)

            # Add financial details
            display_columns.extend(['purchase_price', 'current_value', 'equity', 'monthly_cash_flow'])

            # Print headers
            header_str = ""
            for col in display_columns:
                if col in ['id']:
                    header_str += f"{col:<10} "
                elif col in ['address']:
                    header_str += f"{col:<40} "
                else:
                    header_str += f"{col:<15} "
            print(header_str)
            print("-" * 100)

            # Print each property
            for prop in self.portfolio:
                row_str = ""
                for col in display_columns:
                    if col in prop:
                        if col == 'id':
                            row_str += f"{prop[col]:<10} "
                        elif col == 'address':
                            if len(str(prop[col])) > 38:
                                row_str += f"{str(prop[col])[:38]:<40} "
                            else:
                                row_str += f"{str(prop[col]):<40} "
                        elif col in ['purchase_price', 'current_value', 'equity', 'monthly_cash_flow']:
                            row_str += f"${prop[col]:,.2f}".ljust(15) + " "
                        else:
                            row_str += f"{prop[col]!s:<15} "
                    else:
                        if col == 'id':
                            row_str += f"{'N/A':<10} "
                        elif col == 'address':
                            row_str += f"{'N/A':<40} "
                        else:
                            row_str += f"{'N/A':<15} "
                print(row_str)

        print("-" * 100)
        print(f"TOTAL: {len(self.portfolio)} properties")

        # Cash flow breakdown
        print("\nMONTHLY CASH FLOW BREAKDOWN:")
        print(f"Total Rental Income: ${self.monthly_income:,.2f}")
        print(f"Total Expenses: ${self.monthly_expenses:,.2f}")
        print(f"Net Cash Flow: ${self.monthly_cash_flow:,.2f}")

        # Performance metrics
        if len(self.history['date']) > 1:
            start_date = self.history['date'][0]
            end_date = self.history['date'][-1]
            start_networth = self.history['net_worth'][0]
            end_networth = self.history['net_worth'][-1]

            growth = (end_networth - start_networth) / start_networth * 100

            print("\nPERFORMANCE METRICS:")
            print(f"Simulation Start Date: {start_date.strftime('%Y-%m-%d')}")
            print(f"Simulation End Date: {end_date.strftime('%Y-%m-%d')}")
            print(f"Net Worth Growth: {growth:.2f}%")
            print(f"Number of Properties Acquired: {len(self.portfolio)}")

        print("\n" + "=" * 60)

    def save_results(self, filename='real_estate_investment_results.json'):
        """Save the investment results to a file"""
        # Convert datetime objects to strings for JSON serialization
        portfolio_for_json = []
        for prop in self.portfolio:
            prop_copy = prop.copy()
            if 'purchase_date' in prop_copy:
                prop_copy['purchase_date'] = prop_copy['purchase_date'].strftime('%Y-%m-%d')

            # Ensure all numeric values are standard Python floats (not NumPy types)
            for key, value in prop_copy.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    prop_copy[key] = float(value)

            portfolio_for_json.append(prop_copy)

        # Create results dictionary with all values as standard Python types
        results = {
            'initial_capital': float(self.initial_capital),
            'final_capital': float(self.available_capital),
            'net_worth': float(self.net_worth),
            'total_equity': float(self.total_equity),
            'total_debt': float(self.total_debt),
            'monthly_cash_flow': float(self.monthly_cash_flow),
            'num_properties': len(self.portfolio),
            'simulation_end_date': self.current_date.strftime('%Y-%m-%d'),
            'portfolio': portfolio_for_json,
            'history': {
                'date': [d.strftime('%Y-%m-%d') for d in self.history['date']],
                'capital': [float(x) for x in self.history['capital']],
                'net_worth': [float(x) for x in self.history['net_worth']],
                'monthly_cash_flow': [float(x) for x in self.history['monthly_cash_flow']],
                'num_properties': self.history['num_properties'],
                'total_equity': [float(x) for x in self.history['total_equity']],
                'total_debt': [float(x) for x in self.history['total_debt']],
                'roi': [float(x) for x in self.history['roi']]
            }
        }

        # Save to file with explicit conversion of NumPy types to Python native types
        import json

        # Custom encoder to handle NumPy and other non-standard types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        print(f"Results saved to {filename}")

    def save_model(self, filename_prefix='real_estate_ai_model'):
        """Save the trained models to disk"""
        # Save PyTorch models' state dictionaries
        torch.save(self.valuation_nn.state_dict(), f"{filename_prefix}_valuation.pt")
        torch.save(self.rental_nn.state_dict(), f"{filename_prefix}_rental.pt")

        # Save scalers
        joblib.dump(self.valuation_scaler, f"{filename_prefix}_valuation_scaler.pkl")
        joblib.dump(self.rental_scaler, f"{filename_prefix}_rental_scaler.pkl")

        # Save feature names
        joblib.dump({
            'valuation_features': self.valuation_features,
            'rental_features': self.rental_features
        }, f"{filename_prefix}_features.pkl")

        print(f"Models saved with prefix: {filename_prefix}")

    def load_model(self, filename_prefix='real_estate_ai_model'):
        """Load trained models from disk"""
        # Load feature names
        features = joblib.load(f"{filename_prefix}_features.pkl")
        self.valuation_features = features['valuation_features']
        self.rental_features = features['rental_features']

        # Define model architectures first
        class ValuationModel(nn.Module):
            def __init__(self, input_size):
                super(ValuationModel, self).__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                return x

        class RentalModel(nn.Module):
            def __init__(self, input_size):
                super(RentalModel, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 16)
                self.fc4 = nn.Linear(16, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                return x

        # Load scalers
        self.valuation_scaler = joblib.load(f"{filename_prefix}_valuation_scaler.pkl")
        self.rental_scaler = joblib.load(f"{filename_prefix}_rental_scaler.pkl")

        # Recreate models with the right input sizes
        valuation_input_size = len(self.valuation_scaler.mean_)
        rental_input_size = len(self.rental_scaler.mean_)

        # Initialize models
        self.valuation_nn = ValuationModel(valuation_input_size).to(self.device)
        self.rental_nn = RentalModel(rental_input_size).to(self.device)

        # Load saved weights
        self.valuation_nn.load_state_dict(torch.load(f"{filename_prefix}_valuation.pt"))
        self.rental_nn.load_state_dict(torch.load(f"{filename_prefix}_rental.pt"))

        # Set models to evaluation mode
        self.valuation_nn.eval()
        self.rental_nn.eval()

        # Create wrapper classes
        class TorchModelWrapper:
            def __init__(self, model, scaler, device):
                self.model = model
                self.scaler = scaler
                self.device = device

            def predict(self, X):
                # Explicitly set model to eval mode
                self.model.eval()

                # Handle both DataFrame and NumPy array inputs
                if isinstance(X, pd.DataFrame):
                    X_values = X.values
                else:
                    X_values = X

                X_scaled = self.scaler.transform(X_values)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    predictions = self.model(X_tensor).cpu().numpy()

                # Handle scalar results properly and convert to Python float
                if predictions.ndim == 0:  # It's a scalar
                    return np.array([float(predictions)])  # Convert to Python float

                # Convert to Python float to avoid float32 JSON issues
                return np.array([float(x) for x in predictions.reshape(-1)])

        # Create wrappers for both models
        self.valuation_model = TorchModelWrapper(self.valuation_nn, self.valuation_scaler, self.device)
        self.rental_model = TorchModelWrapper(self.rental_nn, self.rental_scaler, self.device)

        print("Models loaded successfully")


# Main simulation function
def run_real_estate_simulation(data_path, initial_capital=200000, num_properties=5, min_cash_flow=100):
    """Run a real estate investment simulation"""
    # Initialize the AI
    ai = RealEstateAI(data_path, initial_capital=initial_capital)

    # Auto-invest
    performance = ai.auto_invest(max_properties=num_properties, min_cash_flow=min_cash_flow)

    # Print final summary
    ai.print_investment_summary()

    # Save results
    ai.save_results()

    # Save trained models
    ai.save_model()

    return ai


# Example usage
if __name__ == "__main__":
    # File path to the property listings dataset
    data_path = "property_listings.csv"

    # Initial investment capital
    initial_capital = 200000

    # Maximum number of properties to acquire
    num_properties = 10

    # Minimum monthly cash flow per property ($)
    min_cash_flow = 500

    # Run the simulation
    ai = run_real_estate_simulation(data_path, initial_capital, num_properties, min_cash_flow)
