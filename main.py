from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import json
from routers import analysis, portfolio, market, strategies

app = FastAPI(title="Quant Trading API", description="API for quantitative trading application")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router)
app.include_router(portfolio.router)
app.include_router(market.router)
app.include_router(strategies.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Quant Trading API"}

@app.get("/api/stock/{stock_code}")
async def get_stock_data(stock_code: str, days: int = 30):
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for AKShare
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # Fetch stock data using AKShare
        if stock_code.startswith('6'):  # Shanghai
            stock_code_full = f"sh{stock_code}"
        else:  # Shenzhen
            stock_code_full = f"sz{stock_code}"
            
        stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                    start_date=start_date_str, end_date=end_date_str, 
                                    adjust="qfq")
        
        # Convert to dictionary format for JSON response
        result = stock_data.to_dict(orient='records')
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@app.get("/api/stocks/list")
async def get_stock_list(limit: int = 50):
    try:
        # Get A-share stock list
        stock_list = ak.stock_zh_a_spot_em()
        
        # Select only needed columns and limit results
        filtered_list = stock_list[['代码', '名称', '最新价', '涨跌幅', '成交量']].head(limit)
        
        # Rename columns to English
        filtered_list.columns = ['code', 'name', 'price', 'change_percent', 'volume']
        
        return filtered_list.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock list: {str(e)}")

@app.get("/api/indices")
async def get_indices():
    try:
        # Get major indices data
        indices = ak.stock_zh_index_spot()
        
        # Select main indices
        main_indices = indices[indices['代码'].isin(['000001', '399001', '000300'])].copy()
        main_indices.columns = ['code', 'name', 'price', 'change_percent', 'volume', 'turnover']
        
        return main_indices.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching indices: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint for the API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 