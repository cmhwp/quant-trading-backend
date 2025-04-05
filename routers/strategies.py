from fastapi import APIRouter, HTTPException, Body
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

class BacktestRequest(BaseModel):
    stock_code: str
    strategy: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    params: Dict[str, Any] = {}

def calculate_ma(data, window):
    """Calculate Moving Average"""
    return data['收盘'].rolling(window=window).mean()

def calculate_ema(data, window):
    """Calculate Exponential Moving Average"""
    return data['收盘'].ewm(span=window, adjust=False).mean()

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

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    ma = data['收盘'].rolling(window=window).mean()
    std = data['收盘'].rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return ma, upper_band, lower_band

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = data['收盘'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['收盘'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def implement_ma_crossover(data, short_window=5, long_window=20):
    """Implement Moving Average Crossover Strategy"""
    # Calculate moving averages
    data['MA_Short'] = calculate_ma(data, short_window)
    data['MA_Long'] = calculate_ma(data, long_window)
    
    # Initialize position column
    data['Position'] = 0
    
    # Generate buy/sell signals
    data['Signal'] = 0
    data.loc[data['MA_Short'] > data['MA_Long'], 'Signal'] = 1  # Buy signal
    data.loc[data['MA_Short'] < data['MA_Long'], 'Signal'] = -1  # Sell signal
    
    # Generate positions
    data['Position'] = data['Signal'].shift(1)
    data['Position'] = data['Position'].fillna(0)
    
    return data

def implement_rsi_strategy(data, rsi_window=14, overbought=70, oversold=30):
    """Implement RSI Strategy"""
    # Calculate RSI
    data['RSI'] = calculate_rsi(data, rsi_window)
    
    # Generate buy/sell signals
    data['Signal'] = 0
    data.loc[data['RSI'] < oversold, 'Signal'] = 1  # Buy signal when RSI below oversold
    data.loc[data['RSI'] > overbought, 'Signal'] = -1  # Sell signal when RSI above overbought
    
    # Generate positions
    data['Position'] = 0
    
    # For RSI strategy, we need to be a bit more careful with signal generation
    # Only take a position when we cross the threshold
    for i in range(1, len(data)):
        if data['RSI'].iloc[i-1] > oversold and data['RSI'].iloc[i] <= oversold:
            data['Position'].iloc[i] = 1  # Buy when crossing below oversold
        elif data['RSI'].iloc[i-1] < overbought and data['RSI'].iloc[i] >= overbought:
            data['Position'].iloc[i] = -1  # Sell when crossing above overbought
        else:
            data['Position'].iloc[i] = data['Position'].iloc[i-1]  # Maintain previous position
    
    return data

def implement_bollinger_strategy(data, window=20, num_std=2):
    """Implement Bollinger Bands Strategy"""
    # Calculate Bollinger Bands
    ma, upper, lower = calculate_bollinger_bands(data, window, num_std)
    data['BB_MA'] = ma
    data['BB_Upper'] = upper
    data['BB_Lower'] = lower
    
    # Generate buy/sell signals
    data['Signal'] = 0
    data.loc[data['收盘'] <= data['BB_Lower'], 'Signal'] = 1  # Buy signal when price touches lower band
    data.loc[data['收盘'] >= data['BB_Upper'], 'Signal'] = -1  # Sell signal when price touches upper band
    
    # Generate positions
    data['Position'] = data['Signal'].shift(1)
    data['Position'] = data['Position'].fillna(0)
    
    return data

def implement_macd_strategy(data, fast=12, slow=26, signal=9):
    """Implement MACD Strategy"""
    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(data, fast, slow, signal)
    data['MACD'] = macd_line
    data['Signal_Line'] = signal_line
    data['Histogram'] = histogram
    
    # Generate buy/sell signals
    data['Signal'] = 0
    data.loc[data['MACD'] > data['Signal_Line'], 'Signal'] = 1  # Buy signal when MACD crosses above signal line
    data.loc[data['MACD'] < data['Signal_Line'], 'Signal'] = -1  # Sell signal when MACD crosses below signal line
    
    # Generate positions
    data['Position'] = 0
    
    # Only take a position on crossover
    for i in range(1, len(data)):
        if data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1] and data['MACD'].iloc[i] > data['Signal_Line'].iloc[i]:
            data['Position'].iloc[i] = 1  # Buy when MACD crosses above signal line
        elif data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1] and data['MACD'].iloc[i] < data['Signal_Line'].iloc[i]:
            data['Position'].iloc[i] = -1  # Sell when MACD crosses below signal line
        else:
            data['Position'].iloc[i] = data['Position'].iloc[i-1]  # Maintain previous position
    
    return data

def implement_dual_ma_strategy(data, fast=5, medium=20, slow=60):
    """Implement Dual Moving Average Strategy with three periods"""
    # Calculate moving averages
    data['MA_Fast'] = calculate_ma(data, fast)
    data['MA_Medium'] = calculate_ma(data, medium)
    data['MA_Slow'] = calculate_ma(data, slow)
    
    # Generate signals
    data['Signal'] = 0
    
    # Buy when fast crosses above medium and both are above slow
    for i in range(1, len(data)):
        if (data['MA_Fast'].iloc[i-1] <= data['MA_Medium'].iloc[i-1] and 
            data['MA_Fast'].iloc[i] > data['MA_Medium'].iloc[i] and 
            data['MA_Medium'].iloc[i] > data['MA_Slow'].iloc[i]):
            data['Signal'].iloc[i] = 1
        # Sell when fast crosses below medium
        elif (data['MA_Fast'].iloc[i-1] >= data['MA_Medium'].iloc[i-1] and 
              data['MA_Fast'].iloc[i] < data['MA_Medium'].iloc[i]):
            data['Signal'].iloc[i] = -1
    
    # Generate positions
    data['Position'] = data['Signal']
    
    # If no signal, maintain previous position
    for i in range(1, len(data)):
        if data['Signal'].iloc[i] == 0:
            data['Position'].iloc[i] = data['Position'].iloc[i-1]
    
    return data

def run_backtest(data):
    """Run backtest on the strategy signals"""
    # Calculate returns
    data['Returns'] = data['收盘'].pct_change()
    
    # Calculate strategy returns
    data['Strategy_Returns'] = data['Position'] * data['Returns']
    
    # Calculate cumulative returns
    data['Cumulative_Market_Returns'] = (1 + data['Returns']).cumprod() - 1
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod() - 1
    
    # Calculate drawdowns
    data['Market_Peak'] = data['Cumulative_Market_Returns'].cummax()
    data['Strategy_Peak'] = data['Cumulative_Strategy_Returns'].cummax()
    data['Market_Drawdown'] = (data['Cumulative_Market_Returns'] - data['Market_Peak']) / (1 + data['Market_Peak'])
    data['Strategy_Drawdown'] = (data['Cumulative_Strategy_Returns'] - data['Strategy_Peak']) / (1 + data['Strategy_Peak'])
    
    # Calculate various metrics
    total_return = data['Cumulative_Strategy_Returns'].iloc[-1]
    market_return = data['Cumulative_Market_Returns'].iloc[-1]
    
    # Annualized returns
    trading_days_per_year = 252
    years = len(data) / trading_days_per_year
    
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
        market_annualized_return = (1 + market_return) ** (1 / years) - 1
    else:
        annualized_return = total_return
        market_annualized_return = market_return
    
    # Volatility and Sharpe ratio
    strategy_volatility = data['Strategy_Returns'].std() * np.sqrt(trading_days_per_year)
    market_volatility = data['Returns'].std() * np.sqrt(trading_days_per_year)
    
    sharpe_ratio = annualized_return / strategy_volatility if strategy_volatility != 0 else 0
    market_sharpe = market_annualized_return / market_volatility if market_volatility != 0 else 0
    
    # Maximum drawdown
    max_drawdown = data['Strategy_Drawdown'].min()
    market_max_drawdown = data['Market_Drawdown'].min()
    
    # Win/loss ratio
    winning_trades = len(data[data['Strategy_Returns'] > 0])
    losing_trades = len(data[data['Strategy_Returns'] < 0])
    win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
    
    # Prepare results
    backtest_results = {
        'total_return': float(total_return),
        'market_return': float(market_return),
        'annualized_return': float(annualized_return),
        'market_annualized_return': float(market_annualized_return),
        'sharpe_ratio': float(sharpe_ratio),
        'market_sharpe': float(market_sharpe),
        'max_drawdown': float(max_drawdown),
        'market_max_drawdown': float(market_max_drawdown),
        'win_rate': float(win_rate),
        'volatility': float(strategy_volatility),
        'market_volatility': float(market_volatility),
        'trading_days': len(data),
        'daily_returns': data[['日期', 'Position', 'Returns', 'Strategy_Returns', 
                           'Cumulative_Market_Returns', 'Cumulative_Strategy_Returns']].to_dict(orient='records')
    }
    
    return backtest_results

@router.post("/backtest")
async def backtest_strategy(request: BacktestRequest):
    """Backtest a trading strategy"""
    try:
        # Set default dates if not provided
        if not request.end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            end_date = request.end_date
            
        if not request.start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        else:
            start_date = request.start_date
        
        # Fetch stock data
        stock_data = ak.stock_zh_a_hist(symbol=request.stock_code, period="daily", 
                                    start_date=start_date, end_date=end_date, 
                                    adjust="qfq")
        
        # Implement the selected strategy
        if request.strategy == "ma_crossover":
            short_window = request.params.get('short_window', 5)
            long_window = request.params.get('long_window', 20)
            stock_data = implement_ma_crossover(stock_data, short_window, long_window)
        elif request.strategy == "rsi":
            rsi_window = request.params.get('rsi_window', 14)
            overbought = request.params.get('overbought', 70)
            oversold = request.params.get('oversold', 30)
            stock_data = implement_rsi_strategy(stock_data, rsi_window, overbought, oversold)
        elif request.strategy == "bollinger":
            window = request.params.get('window', 20)
            num_std = request.params.get('num_std', 2)
            stock_data = implement_bollinger_strategy(stock_data, window, num_std)
        elif request.strategy == "macd":
            fast = request.params.get('fast', 12)
            slow = request.params.get('slow', 26)
            signal = request.params.get('signal', 9)
            stock_data = implement_macd_strategy(stock_data, fast, slow, signal)
        elif request.strategy == "dual_ma":
            fast = request.params.get('fast', 5)
            medium = request.params.get('medium', 20)
            slow = request.params.get('slow', 60)
            stock_data = implement_dual_ma_strategy(stock_data, fast, medium, slow)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
        
        # Run backtest
        backtest_results = run_backtest(stock_data)
        
        # Add strategy info to results
        backtest_results['strategy'] = request.strategy
        backtest_results['stock_code'] = request.stock_code
        backtest_results['params'] = request.params
        backtest_results['start_date'] = start_date
        backtest_results['end_date'] = end_date
        
        return backtest_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting strategy: {str(e)}")

@router.get("/list")
async def list_strategies():
    """List available trading strategies"""
    strategies = [
        {
            "id": "ma_crossover",
            "name": "Moving Average Crossover",
            "description": "A strategy that generates buy/sell signals when a short-term moving average crosses a long-term moving average.",
            "parameters": [
                {"name": "short_window", "default": 5, "description": "Short-term moving average window"},
                {"name": "long_window", "default": 20, "description": "Long-term moving average window"}
            ]
        },
        {
            "id": "rsi",
            "name": "Relative Strength Index",
            "description": "A momentum oscillator that measures the speed and change of price movements, generating signals on overbought/oversold conditions.",
            "parameters": [
                {"name": "rsi_window", "default": 14, "description": "RSI calculation window"},
                {"name": "overbought", "default": 70, "description": "Overbought threshold"},
                {"name": "oversold", "default": 30, "description": "Oversold threshold"}
            ]
        },
        {
            "id": "bollinger",
            "name": "Bollinger Bands",
            "description": "A strategy using Bollinger Bands to identify overbought/oversold conditions when price touches the bands.",
            "parameters": [
                {"name": "window", "default": 20, "description": "Moving average window for band calculation"},
                {"name": "num_std", "default": 2, "description": "Number of standard deviations for band width"}
            ]
        },
        {
            "id": "macd",
            "name": "MACD (Moving Average Convergence Divergence)",
            "description": "A trend-following momentum indicator that shows the relationship between two moving averages of a security's price.",
            "parameters": [
                {"name": "fast", "default": 12, "description": "Fast EMA period"},
                {"name": "slow", "default": 26, "description": "Slow EMA period"},
                {"name": "signal", "default": 9, "description": "Signal line period"}
            ]
        },
        {
            "id": "dual_ma",
            "name": "Dual Moving Average",
            "description": "A strategy using three moving averages to identify trend direction and generate signals.",
            "parameters": [
                {"name": "fast", "default": 5, "description": "Fast MA period"},
                {"name": "medium", "default": 20, "description": "Medium MA period"},
                {"name": "slow", "default": 60, "description": "Slow MA period"}
            ]
        }
    ]
    
    return strategies 