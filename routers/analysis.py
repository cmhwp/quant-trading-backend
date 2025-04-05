from fastapi import APIRouter, HTTPException
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

def calculate_ma(data, window):
    """Calculate Moving Average"""
    return data['收盘'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data['收盘'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@router.get("/technical/{stock_code}")
async def get_technical_indicators(stock_code: str, days: int = 90):
    """Get technical indicators for a stock"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for AKShare
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # Fetch stock data
        stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                    start_date=start_date_str, end_date=end_date_str, 
                                    adjust="qfq")
        
        # Calculate technical indicators
        stock_data['MA5'] = calculate_ma(stock_data, 5)
        stock_data['MA10'] = calculate_ma(stock_data, 10)
        stock_data['MA20'] = calculate_ma(stock_data, 20)
        stock_data['MA60'] = calculate_ma(stock_data, 60)
        stock_data['RSI'] = calculate_rsi(stock_data)
        
        # Basic trading signals
        stock_data['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        
        # Simple MA crossover strategy
        stock_data.loc[stock_data['MA5'] > stock_data['MA20'], 'Signal'] = 1
        stock_data.loc[stock_data['MA5'] < stock_data['MA20'], 'Signal'] = -1
        
        # Convert to dictionary format for JSON response
        result = stock_data.fillna("").to_dict(orient='records')
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating technical indicators: {str(e)}")

@router.get("/backtest/{stock_code}")
async def backtest_strategy(stock_code: str, ma_short: int = 5, ma_long: int = 20, days: int = 365):
    """Backtest a simple Moving Average crossover strategy"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for AKShare
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # Fetch stock data
        stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                    start_date=start_date_str, end_date=end_date_str, 
                                    adjust="qfq")
        
        # Calculate Moving Averages
        stock_data['MA_Short'] = calculate_ma(stock_data, ma_short)
        stock_data['MA_Long'] = calculate_ma(stock_data, ma_long)
        
        # Generate signals
        stock_data['Signal'] = 0
        stock_data.loc[stock_data['MA_Short'] > stock_data['MA_Long'], 'Signal'] = 1  # Buy
        stock_data.loc[stock_data['MA_Short'] < stock_data['MA_Long'], 'Signal'] = -1  # Sell
        
        # Calculate daily returns
        stock_data['Returns'] = stock_data['收盘'].pct_change()
        
        # Calculate strategy returns
        stock_data['Strategy_Returns'] = stock_data['Signal'].shift(1) * stock_data['Returns']
        
        # Calculate cumulative returns
        stock_data['Cumulative_Market_Returns'] = (1 + stock_data['Returns']).cumprod() - 1
        stock_data['Cumulative_Strategy_Returns'] = (1 + stock_data['Strategy_Returns']).cumprod() - 1
        
        # Fill NaN values
        stock_data = stock_data.fillna(0)
        
        # Prepare results
        backtest_result = {
            'start_date': start_date_str,
            'end_date': end_date_str,
            'market_return': float(stock_data['Cumulative_Market_Returns'].iloc[-1]),
            'strategy_return': float(stock_data['Cumulative_Strategy_Returns'].iloc[-1]),
            'data': stock_data[['日期', 'Signal', 'Returns', 'Strategy_Returns', 
                               'Cumulative_Market_Returns', 'Cumulative_Strategy_Returns']].to_dict(orient='records')
        }
        
        return backtest_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing backtest: {str(e)}")

@router.get("/correlation")
async def get_stock_correlation(stocks: str, days: int = 90):
    """Get correlation matrix between stocks"""
    try:
        stock_list = stocks.split(',')
        if len(stock_list) < 2 or len(stock_list) > 10:
            raise HTTPException(status_code=400, detail="Please provide between 2 and 10 stock codes")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for AKShare
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # Fetch data for all stocks
        all_data = {}
        for stock in stock_list:
            try:
                stock_data = ak.stock_zh_a_hist(symbol=stock, period="daily", 
                                         start_date=start_date_str, end_date=end_date_str, 
                                         adjust="qfq")
                all_data[stock] = stock_data['收盘']
            except:
                pass
        
        # Create a DataFrame with all stock prices
        df = pd.DataFrame(all_data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr().round(4)
        
        # Convert to records format
        result = []
        for stock1 in corr_matrix.index:
            row = {"stock": stock1}
            for stock2 in corr_matrix.columns:
                row[stock2] = float(corr_matrix.loc[stock1, stock2])
            result.append(row)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating correlation: {str(e)}") 