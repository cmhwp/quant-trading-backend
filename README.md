# Quant Trading Backend API

This is the backend API for the Quant Trading platform, providing data and analysis for quantitative trading.

## Features

- Stock data retrieval and analysis
- Technical indicators calculation
- Portfolio management and optimization
- Market data and economic indicators
- Multiple trading strategies with backtesting

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository
2. Navigate to the backend directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

To start the API server:

```bash
python main.py
```

The API will be available at http://127.0.0.1:8000. You can also access the Swagger UI documentation at http://127.0.0.1:8000/docs.

## API Endpoints

### Core Stock Data

- `GET /api/stock/{stock_code}` - Get historical data for a specific stock
- `GET /api/stocks/list` - Get a list of available stocks
- `GET /api/indices` - Get major market indices

### Technical Analysis

- `GET /api/analysis/technical/{stock_code}` - Get technical indicators for a stock
- `GET /api/analysis/backtest/{stock_code}` - Backtest a simple MA crossover strategy
- `GET /api/analysis/correlation` - Get correlation matrix between stocks

### Portfolio Management

- `POST /api/portfolio/performance` - Analyze portfolio performance
- `POST /api/portfolio/optimize` - Optimize portfolio weights based on historical data
- `GET /api/portfolio/risk/{stock_code}` - Analyze risk metrics for a specific stock

### Market Data

- `GET /api/market/sectors` - Get performance of market sectors
- `GET /api/market/hot_sectors` - Get top performing sectors
- `GET /api/market/concept_stocks` - Get concept stocks and their performance
- `GET /api/market/industry_stocks` - Get industry stocks and their performance
- `GET /api/market/market_breadth` - Get market breadth indicators
- `GET /api/market/news` - Get latest market news
- `GET /api/market/economic/indicators` - Get key economic indicators
- `GET /api/market/institutional/sentiment` - Get institutional investor sentiment
- `GET /api/market/calendar` - Get market events calendar

### Trading Strategies

- `GET /api/strategies/list` - List available trading strategies
- `POST /api/strategies/backtest` - Backtest a trading strategy

## Environment Variables

You can configure the API using environment variables. Create a `.env` file in the backend directory with the following options:

```
DEBUG=False
ALLOWED_ORIGINS=http://localhost:3000,http://example.com
```

## Development

To run the API in development mode with auto-reload:

```bash
uvicorn main:app --reload
``` 