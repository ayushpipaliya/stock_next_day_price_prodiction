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

# Example usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = YahooFinanceScraper()
    
    # Example 1: Get last 30 days data for Tata Motors
    print("Fetching last 30 days data for TATAMOTORS.BO...")
    try:
        df = get_last_n_days_data('TATAMOTORS.BO', 30)
        print(f"Retrieved {len(df)} rows of data")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Save to CSV
        scraper.save_to_csv(df, 'tatamotors_30days.csv')
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Get data for specific date range
    print("Fetching data for specific date range...")
    try:
        df2 = get_date_range_data('AAPL', '2024-01-01', '2024-01-31')
        print(f"Retrieved {len(df2)} rows of data for AAPL")
        print("\nFirst 5 rows:")
        print(df2.head())
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Multiple stocks
    print("\n" + "="*50 + "\n")
    print("Fetching data for multiple stocks...")
    
    stocks = ['RELIANCE.BO', 'TCS.BO', 'INFY.BO']
    for stock in stocks:
        try:
            print(f"\nFetching data for {stock}...")
            df = get_last_n_days_data(stock, 7)  # Last 7 days
            print(f"Latest price: {df['Close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"Error fetching {stock}: {e}")