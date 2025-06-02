import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import re

class YahooFinanceScraper:
    def __init__(self):
        self.base_url = "https://finance.yahoo.com/quote/{}/history/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def date_to_timestamp(self, date_str):
        """Convert date string (YYYY-MM-DD) to Unix timestamp"""
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return int(dt.timestamp())
        except ValueError:
            raise ValueError("Date format should be YYYY-MM-DD")
    
    def get_historical_data(self, symbol, start_date, end_date, debug=False):
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'TATAMOTORS.BO', 'AAPL')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            debug (bool): Print debug information
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        try:
            # Convert dates to timestamps
            period1 = self.date_to_timestamp(start_date)
            period2 = self.date_to_timestamp(end_date)
            
            # Construct URL
            url = self.base_url.format(symbol)
            params = {
                'period1': period1,
                'period2': period2
            }
            
            if debug:
                print(f"Fetching URL: {url}")
                print(f"Parameters: {params}")
            
            # Make request
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            if debug:
                print(f"Response status: {response.status_code}")
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the data table - try multiple selectors
            table = None
            table_selectors = [
                'table.table',
                'table[data-test="historical-prices"]',
                'table',
                '.table'
            ]
            
            for selector in table_selectors:
                table = soup.select_one(selector)
                if table:
                    if debug:
                        print(f"Found table with selector: {selector}")
                    break
            
            if not table:
                if debug:
                    print("Available tables on page:")
                    tables = soup.find_all('table')
                    for i, t in enumerate(tables):
                        print(f"Table {i}: {t.get('class', 'no-class')}")
                
                raise ValueError("Could not find data table on the page")
            
            # Extract data
            data = self._parse_table(table)
            
            if debug:
                print(f"Extracted {len(data)} rows of data")
                if data:
                    print(f"Sample row: {data[0]}")
            
            if not data:
                raise ValueError("No data found in the table")
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            if debug:
                print(f"DataFrame shape before cleaning: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
            
            # Clean and format data
            df = self._clean_dataframe(df)
            
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Adj_Close'] = pd.to_numeric(df['Adj_Close'], errors='coerce')
            
            if debug:
                print(f"DataFrame shape after cleaning: {df.shape}")
            
            return df
            
        except requests.RequestException as e:
            raise Exception(f"Error fetching data: {e}")
        except Exception as e:
            raise Exception(f"Error processing data: {e}")
    
    def _parse_table(self, table):
        """Parse the HTML table and extract data"""
        data = []
        
        # Get table headers
        headers = []
        header_row = table.find('thead')
        if header_row:
            header_row = header_row.find('tr')
        else:
            # Fallback: try to find headers in the first row
            header_row = table.find('tr')
        
        if header_row:
            for th in header_row.find_all(['th', 'td']):
                # Clean header text (remove extra whitespace and icons)
                header_text = th.get_text().strip()
                header_text = re.sub(r'\s+', ' ', header_text)  # Replace multiple spaces with single space
                # Remove common artifacts
                header_text = header_text.replace('*', '').strip()
                headers.append(header_text)
        
        # Get table rows
        tbody = table.find('tbody')
        if not tbody:
            # If no tbody, get all rows except the first (header)
            all_rows = table.find_all('tr')[1:]
        else:
            all_rows = tbody.find_all('tr')
        
        for row in all_rows:
            row_data = []
            for td in row.find_all(['td', 'th']):
                cell_text = td.get_text().strip()
                # Clean up common formatting issues
                cell_text = cell_text.replace('\n', ' ').replace('\t', ' ')
                cell_text = re.sub(r'\s+', ' ', cell_text).strip()
                row_data.append(cell_text)
            
            # Only add rows that have the right number of columns
            if len(row_data) == len(headers) and len(row_data) > 0:
                # Skip rows that might be empty or contain only dashes
                if not all(cell == '-' or cell == '' for cell in row_data):
                    data.append(dict(zip(headers, row_data)))
        
        return data
    
    def _clean_dataframe(self, df):
        """Clean and format the DataFrame"""
        # Rename columns for consistency
        column_mapping = {
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume'
        }
        
        # Only rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        # Convert Date column to datetime with flexible parsing
        if 'Date' in df.columns:
            # Handle different date formats from Yahoo Finance
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            except:
                # Fallback: try common date formats
                date_formats = ['%b %d, %Y', '%B %d, %Y', '%Y-%m-%d', '%m/%d/%Y']
                for fmt in date_formats:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                        break
                    except:
                        continue
                else:
                    # Last resort - let pandas infer the format
                    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
        for col in numeric_columns:
            if col in df.columns:
                # Handle both comma separators and dash for missing data
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('-', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert Volume (handle commas and different formats)
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(str).str.replace(',', '').str.replace('-', '0')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Remove rows with invalid dates
        if 'Date' in df.columns:
            df = df.dropna(subset=['Date'])
        
        # Sort by date (oldest first)
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def save_to_csv(self, df, filename):
        """Save DataFrame to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

# Example usage and utility functions
def get_last_n_days_data(symbol, days=30, debug=False):
    """Get historical data for the last n days"""
    scraper = YahooFinanceScraper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return scraper.get_historical_data(
        symbol, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d'),
        debug=debug
    )

def get_date_range_data(symbol, start_date, end_date, debug=False):
    """Get historical data for a specific date range"""
    scraper = YahooFinanceScraper()
    return scraper.get_historical_data(symbol, start_date, end_date, debug=debug)


# Set page config
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

class ImprovedStockPredictor:
    def __init__(self, symbol, start_date=None, end_date=None, period="2y"):
        """Initialize the improved stock predictor"""
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.feature_selector = None
        
        if start_date and end_date:
            self.start_date = start_date
            self.end_date = end_date
        else:
            # Convert period to dates
            if period == "1y":
                days = 365
            elif period == "2y":
                days = 730
            elif period == "5y":
                days = 1825
            elif period == "max":
                days = 3650  # 10 years max
            else:
                days = 730
            
            self.start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        
    def fetch_data(self):
        """Fetch stock data"""
        with st.spinner(f"Fetching data for {self.symbol}..."):
            self.data = get_date_range_data(self.symbol, self.start_date, self.end_date)
            
            if self.data is None or self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            st.success(f"Fetched {len(self.data)} days of data")
            return self.data
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        df = self.data.copy()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['HL_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['OC_Pct'] = (df['Open'] - df['Close']) / df['Close']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'SMA_{window}_ratio'] = df['Close'] / df[f'SMA_{window}']
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Momentum
        for period in [1, 5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Price position in recent range
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        df['Price_position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
        
        # Clean infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        self.data = df
        return df
    
    def create_features(self, lookback_days=3):
        """Create features for ML models"""
        df = self.data.copy()
        
        # Select core predictive features
        core_features = [
            'Returns', 'LogReturns', 'HL_Pct', 'Volume_ratio',
            'SMA_5_ratio', 'SMA_20_ratio', 'RSI', 'BB_position',
            'Volatility', 'Price_position', 'Momentum_1', 'Momentum_5'
        ]
        
        # Create lagged features
        feature_list = []
        for col in core_features:
            if col in df.columns:
                feature_list.append(col)
                for lag in range(1, min(3, lookback_days + 1)):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    feature_list.append(f'{col}_lag_{lag}')
        
        # Create target (next day return)
        df['Target_Return'] = df['Returns'].shift(-1)
        
        # Time features
        df.reset_index(inplace=True)
        if 'Date' in df.columns:
            df['DayOfWeek'] = df['Date'].dt.dayofweek / 6.0
            df['Month'] = df['Date'].dt.month / 12.0
            feature_list.extend(['DayOfWeek', 'Month'])
        
        # Drop NaN rows
        df = df.dropna()
        
        # Select features
        available_features = [col for col in feature_list if col in df.columns]
        self.features = df[available_features].copy()
        self.target = df['Target_Return'].copy()
        
        # Final cleanup
        self.features = self.features.fillna(0)
        self.features = self.features.replace([np.inf, -np.inf], 0)
        
        return self.features, self.target
    
    def select_features_with_rfe(self, n_features=10):
        """Feature selection using RFE"""
        if len(self.features.columns) <= n_features:
            return self.features
        
        try:
            estimator = Ridge(alpha=1.0)
            selector = RFE(estimator, n_features_to_select=n_features, step=1)
            
            X = self.features.values.astype(float)
            y = self.target.values.astype(float)
            
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            selector.fit(X, y)
            
            selected_features = self.features.columns[selector.support_].tolist()
            self.features = self.features[selected_features].copy()
            self.feature_selector = selector
            
            return self.features
            
        except Exception as e:
            st.warning(f"Feature selection failed: {str(e)}")
            return self.features
    
    def split_data(self, test_size=0.2):
        """Split data chronologically"""
        n_samples = len(self.features)
        
        if n_samples < 50:
            test_size = min(test_size, 0.3)
        
        split_idx = int(n_samples * (1 - test_size))
        
        X_train = self.features.iloc[:split_idx]
        X_test = self.features.iloc[split_idx:]
        y_train = self.target.iloc[:split_idx]
        y_test = self.target.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self):
        """Train ML models"""
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Define models
        models = {
            'Ridge': Ridge(alpha=10.0),
            'Lasso': Lasso(alpha=0.1, max_iter=5000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            'RandomForest': RandomForestRegressor(
                n_estimators=50, max_depth=3, min_samples_split=20,
                min_samples_leaf=10, random_state=42
            )
        }
        
        # Train models
        progress_bar = st.progress(0)
        total_models = len(models)
        
        for i, (name, model) in enumerate(models.items()):
            try:
                # Cross-validation
                n_splits = min(3, len(X_train) // 20)
                tscv = TimeSeriesSplit(n_splits=n_splits)
                
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=tscv, scoring='r2', n_jobs=-1
                )
                
                # Train on full training set
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                self.models[name] = model
                self.predictions[name] = {
                    'train_pred': train_pred,
                    'test_pred': test_pred,
                    'y_train': y_train.values,
                    'y_test': y_test.values,
                    'cv_score': np.mean(cv_scores)
                }
                
                progress_bar.progress((i + 1) / total_models)
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
                continue
    
    def predict_next_day(self):
        """Make next-day predictions"""
        if not self.models:
            return {}
        
        current_price = float(self.data['Close'].iloc[-1])
        predictions = {}
        
        try:
            # Get latest features
            latest_features = self.features.iloc[-1:].copy()
            latest_features = latest_features.fillna(0)
            latest_features_scaled = self.scalers['main'].transform(latest_features)
            
            # Individual model predictions
            for name, model in self.models.items():
                try:
                    pred_return = model.predict(latest_features_scaled)[0]
                    pred_price = current_price * (1 + pred_return)
                    predictions[name] = pred_price
                except Exception as e:
                    continue
            
            # Ensemble prediction
            if len(predictions) > 1:
                ensemble_price = np.mean(list(predictions.values()))
                predictions['Ensemble'] = ensemble_price
            
            # Apply realistic constraints (max 10% daily change)
            max_change = 0.10
            for name in predictions:
                pred_change = (predictions[name] - current_price) / current_price
                if abs(pred_change) > max_change:
                    capped_change = max_change if pred_change > 0 else -max_change
                    predictions[name] = current_price * (1 + capped_change)
            
            return predictions
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return {'Current': current_price}
    
    def evaluate_models(self):
        """Evaluate model performance"""
        if not self.predictions:
            return pd.DataFrame()
        
        results = {}
        
        for name, pred_data in self.predictions.items():
            test_mse = mean_squared_error(pred_data['y_test'], pred_data['test_pred'])
            test_mae = mean_absolute_error(pred_data['y_test'], pred_data['test_pred'])
            test_r2 = r2_score(pred_data['y_test'], pred_data['test_pred'])
            
            # Directional accuracy
            actual_direction = np.sign(pred_data['y_test'])
            pred_direction = np.sign(pred_data['test_pred'])
            directional_accuracy = np.mean(actual_direction == pred_direction)
            
            # Information Coefficient
            ic = np.corrcoef(pred_data['y_test'], pred_data['test_pred'])[0, 1]
            if np.isnan(ic):
                ic = 0
            
            results[name] = {
                'Test_RMSE': np.sqrt(test_mse),
                'Test_MAE': test_mae,
                'Test_R2': test_r2,
                'Directional_Accuracy': directional_accuracy,
                'Information_Coefficient': ic,
                'CV_Score': pred_data.get('cv_score', 0)
            }
        
        return pd.DataFrame(results).T

def create_price_chart(data, predictions=None):
    """Create interactive price chart"""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # Moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='red', width=1)
        ))
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ))
    
    fig.update_layout(
        title='Stock Price Chart with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        showlegend=True
    )
    
    return fig

def create_indicator_charts(data):
    """Create technical indicator charts"""
    fig = go.Figure()
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ))
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title='RSI Indicator',
        xaxis_title='Date',
        yaxis_title='RSI',
        height=300,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# Streamlit UI
def main():
    st.title("📈 AI Stock Price Predictor")
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)")
    
    # Date selection method
    date_method = st.sidebar.radio("Select Date Range Method", ["Period", "Custom Dates"])
    
    if date_method == "Period":
        period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "max"], index=1)
        start_date = None
        end_date = None
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=730))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        period = None
    
    # Analysis button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    if run_analysis and symbol:
        try:
            # Initialize predictor
            if date_method == "Period":
                predictor = ImprovedStockPredictor(symbol, period=period)
            else:
                predictor = ImprovedStockPredictor(symbol, start_date=start_date, end_date=end_date)
            
            # Run analysis
            st.header(f"Analysis for {symbol.upper()}")
            
            # Fetch and process data
            data = predictor.fetch_data()
            predictor.calculate_technical_indicators()
            predictor.create_features()
            
            # Feature selection
            optimal_features = min(10, len(predictor.features.columns) // 3)
            predictor.select_features_with_rfe(n_features=optimal_features)
            
            # Train models
            st.subheader("🤖 Training ML Models")
            predictor.train_models()
            
            if not predictor.models:
                st.error("No models were successfully trained")
                return
            
            # Display charts
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Price Chart")
                price_chart = create_price_chart(predictor.data)
                st.plotly_chart(price_chart, use_container_width=True)
                
                st.subheader("📈 Technical Indicators")
                indicator_chart = create_indicator_charts(predictor.data)
                st.plotly_chart(indicator_chart, use_container_width=True)
            
            with col2:
                st.subheader("📋 Current Stats")
                current_price = predictor.data['Close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
                
                if 'Returns' in predictor.data.columns:
                    daily_return = predictor.data['Returns'].iloc[-1] * 100
                    st.metric("Daily Return", f"{daily_return:.2f}%")
                
                if 'RSI' in predictor.data.columns:
                    current_rsi = predictor.data['RSI'].iloc[-1]
                    st.metric("RSI", f"{current_rsi:.1f}")
                
                if 'Volatility' in predictor.data.columns:
                    volatility = predictor.data['Volatility'].iloc[-1] * 100
                    st.metric("Volatility", f"{volatility:.2f}%")
            
            # Model evaluation
            st.subheader("🎯 Model Performance")
            results = predictor.evaluate_models()
            
            if not results.empty:
                # Display results table
                st.dataframe(results.round(4), use_container_width=True)
                
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    best_r2 = results['Test_R2'].max()
                    st.metric("Best R²", f"{best_r2:.4f}")
                
                with col2:
                    best_dir_acc = results['Directional_Accuracy'].max()
                    st.metric("Best Dir. Accuracy", f"{best_dir_acc:.4f}")
                
                with col3:
                    avg_ic = results['Information_Coefficient'].mean()
                    st.metric("Avg. Info Coeff.", f"{avg_ic:.4f}")
            
            # Predictions
            st.subheader("🔮 Next Day Predictions")
            predictions = predictor.predict_next_day()
            
            if predictions:
                current_price = predictor.data['Close'].iloc[-1]
                
                # Create predictions dataframe
                pred_df = pd.DataFrame([
                    {
                        'Model': name,
                        'Predicted_Price': price,
                        'Change_$': price - current_price,
                        'Change_%': ((price - current_price) / current_price) * 100
                    }
                    for name, price in predictions.items()
                ])
                
                st.dataframe(pred_df.round(4), use_container_width=True)
                
                # Trading signal
                if 'Ensemble' in predictions:
                    ensemble_price = predictions['Ensemble']
                    expected_return = ((ensemble_price - current_price) / current_price) * 100
                    
                    st.subheader("💡 Trading Signal")
                    
                    if abs(expected_return) < 0.5:
                        signal = "HOLD"
                        signal_color = "gray"
                    elif expected_return > 1.5:
                        signal = "WEAK BUY"
                        signal_color = "green"
                    elif expected_return < -1.5:
                        signal = "WEAK SELL"
                        signal_color = "red"
                    else:
                        signal = "HOLD"
                        signal_color = "gray"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Signal:** :{signal_color}[{signal}]")
                    with col2:
                        st.markdown(f"**Expected Return:** {expected_return:+.2f}%")
                    with col3:
                        st.markdown(f"**Target Price:** ${ensemble_price:.2f}")
                    
                    # Warning for low performance
                    if not results.empty:
                        avg_r2 = results['Test_R2'].mean()
                        avg_dir_acc = results['Directional_Accuracy'].mean()
                        
                        if avg_r2 < 0 or avg_dir_acc < 0.55:
                            st.warning("⚠️ WARNING: Low model performance - Use predictions with extreme caution")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            st.info("""
            - Always use stop-losses and proper position sizing
            - Consider this as one factor among many in your analysis
            - Model predictions are most reliable for very short-term movements
            - Past performance does not guarantee future results
            """)
            
        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    elif run_analysis and not symbol:
        st.warning("Please enter a stock symbol")
    
    # Information section
    if not run_analysis:
        st.markdown("""
        ## Welcome to AI Stock Predictor! 🎯
        
        This application uses machine learning to predict stock prices based on technical indicators and historical data.
        
        ### Features:
        - 📊 Technical analysis with multiple indicators
        - 🤖 Multiple ML models (Ridge, Lasso, ElasticNet, Random Forest)
        - 📈 Interactive charts and visualizations
        - 🎯 Next-day price predictions
        - 💡 Trading signals and recommendations
        
        ### How to use:
        1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
        2. Choose your date range (period or custom dates)
        3. Click "Run Analysis" to start the prediction
        
        ### Disclaimer:
        This tool is for educational purposes only. Always do your own research and consult with financial advisors before making investment decisions.
        """)

if __name__ == "__main__":
    main()
