from fastapi import APIRouter, HTTPException, Body
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pydantic import BaseModel

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

class StockWeight(BaseModel):
    code: str
    weight: float

class Portfolio(BaseModel):
    stocks: List[StockWeight]
    start_date: str
    end_date: str = None

@router.post("/performance")
async def analyze_portfolio_performance(portfolio: Portfolio):
    """Analyze the performance of a portfolio over time"""
    try:
        # If end_date is not provided, use current date
        if not portfolio.end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            end_date = portfolio.end_date
            
        # Initialize portfolio performance tracking
        portfolio_data = pd.DataFrame()
        stock_data_dict = {}
        
        # Get data for each stock
        for stock in portfolio.stocks:
            try:
                stock_data = ak.stock_zh_a_hist(symbol=stock.code, period="daily", 
                                          start_date=portfolio.start_date, end_date=end_date, 
                                          adjust="qfq")
                # Calculate daily returns
                stock_data['Returns'] = stock_data['收盘'].pct_change()
                stock_data_dict[stock.code] = stock_data
                
                # Add weighted returns to portfolio
                if portfolio_data.empty:
                    portfolio_data = pd.DataFrame(index=stock_data.index)
                    portfolio_data['日期'] = stock_data['日期']
                
                portfolio_data[f'Returns_{stock.code}'] = stock_data['Returns'] * stock.weight
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing stock {stock.code}: {str(e)}")
        
        # Calculate portfolio returns
        portfolio_data['Portfolio_Returns'] = portfolio_data[[col for col in portfolio_data.columns if col.startswith('Returns_')]].sum(axis=1)
        
        # Calculate cumulative returns
        portfolio_data['Cumulative_Returns'] = (1 + portfolio_data['Portfolio_Returns'].fillna(0)).cumprod() - 1
        
        # Calculate performance metrics
        total_return = portfolio_data['Cumulative_Returns'].iloc[-1]
        annualized_return = ((1 + total_return) ** (252 / len(portfolio_data)) - 1) if len(portfolio_data) > 0 else 0
        volatility = portfolio_data['Portfolio_Returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        max_drawdown = (portfolio_data['Cumulative_Returns'] - portfolio_data['Cumulative_Returns'].cummax()).min()
        
        # Prepare result
        result = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "daily_returns": portfolio_data[['日期', 'Portfolio_Returns', 'Cumulative_Returns']].fillna(0).to_dict(orient='records')
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing portfolio: {str(e)}")

@router.post("/optimize")
async def optimize_portfolio(stock_list: List[str] = Body(...), days: int = 365, optimization_type: str = "sharpe"):
    """Optimize portfolio weights based on historical data"""
    try:
        if len(stock_list) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 stocks for optimization")
            
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # Get data for all stocks
        all_returns = {}
        for stock in stock_list:
            try:
                stock_data = ak.stock_zh_a_hist(symbol=stock, period="daily", 
                                          start_date=start_date_str, end_date=end_date_str, 
                                          adjust="qfq")
                all_returns[stock] = stock_data['收盘'].pct_change().fillna(0)
            except:
                raise HTTPException(status_code=400, detail=f"Cannot fetch data for stock {stock}")
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(all_returns)
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_df.mean() * 252  # Annualized returns
        cov_matrix = returns_df.cov() * 252  # Annualized covariance
        
        # Simple portfolio optimization through Monte Carlo simulation
        num_portfolios = 10000
        results = np.zeros((num_portfolios, len(stock_list) + 2))
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(len(stock_list))
            weights /= np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
            
            # Store results
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_volatility
            results[i, 2:] = weights
        
        # Convert results to DataFrame
        columns = ['Return', 'Volatility'] + stock_list
        portfolios = pd.DataFrame(results, columns=columns)
        
        # Find optimal portfolio based on criterion
        if optimization_type == "sharpe":
            # Find portfolio with highest Sharpe ratio
            portfolios['Sharpe'] = portfolios['Return'] / portfolios['Volatility']
            optimal_idx = portfolios['Sharpe'].idxmax()
        elif optimization_type == "min_volatility":
            # Find portfolio with minimum volatility
            optimal_idx = portfolios['Volatility'].idxmin()
        elif optimization_type == "max_return":
            # Find portfolio with maximum return
            optimal_idx = portfolios['Return'].idxmax()
        else:
            raise HTTPException(status_code=400, detail="Invalid optimization type")
            
        # Get optimal portfolio
        optimal_portfolio = portfolios.iloc[optimal_idx]
        
        # Prepare result
        weights_dict = {stock: float(optimal_portfolio[stock]) for stock in stock_list}
        result = {
            "weights": weights_dict,
            "expected_return": float(optimal_portfolio['Return']),
            "expected_volatility": float(optimal_portfolio['Volatility']),
            "sharpe_ratio": float(optimal_portfolio.get('Sharpe', optimal_portfolio['Return'] / optimal_portfolio['Volatility'])),
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing portfolio: {str(e)}")

@router.get("/risk/{stock_code}")
async def analyze_risk(stock_code: str, days: int = 90):
    """Analyze risk metrics for a specific stock"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # Get stock data
        stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                    start_date=start_date_str, end_date=end_date_str, 
                                    adjust="qfq")
        
        # Calculate returns
        stock_data['Returns'] = stock_data['收盘'].pct_change().fillna(0)
        
        # Calculate various risk metrics
        volatility = stock_data['Returns'].std() * np.sqrt(252)  # Annualized volatility
        negative_returns = stock_data['Returns'][stock_data['Returns'] < 0]
        downside_risk = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # Calculate VaR and CVaR at 95% confidence
        var_95 = np.percentile(stock_data['Returns'], 5)
        cvar_95 = stock_data['Returns'][stock_data['Returns'] <= var_95].mean() if var_95 != 0 else 0
        
        # Calculate max drawdown
        stock_data['Cumulative_Returns'] = (1 + stock_data['Returns']).cumprod() - 1
        drawdown = (stock_data['Cumulative_Returns'] - stock_data['Cumulative_Returns'].cummax())
        max_drawdown = drawdown.min()
        
        # Calculate beta vs market (using Shanghai index as market proxy)
        try:
            market_data = ak.stock_zh_index_daily(symbol="sh000001")
            market_data = market_data[(market_data.index >= stock_data.index[0]) & 
                                      (market_data.index <= stock_data.index[-1])]
            market_returns = market_data['close'].pct_change().fillna(0)
            
            # Align data
            common_dates = set(market_returns.index).intersection(set(stock_data.index))
            stock_returns_aligned = stock_data.set_index('日期')['Returns'].loc[common_dates]
            market_returns_aligned = market_returns.loc[common_dates]
            
            # Calculate beta
            covariance = np.cov(stock_returns_aligned, market_returns_aligned)[0, 1]
            market_variance = np.var(market_returns_aligned)
            beta = covariance / market_variance if market_variance != 0 else 1
        except:
            beta = 1
        
        # Prepare result
        result = {
            "volatility": float(volatility),
            "downside_risk": float(downside_risk),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(max_drawdown),
            "beta": float(beta)
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing risk: {str(e)}") 