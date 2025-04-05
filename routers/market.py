from fastapi import APIRouter, HTTPException, Query
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

router = APIRouter(prefix="/api/market", tags=["market"])

@router.get("/sectors")
async def get_market_sectors():
    """Get performance of market sectors"""
    try:
        # Get sector performance data
        sector_data = ak.stock_sector_spot()
        
        # Clean and format the data
        sector_data.columns = ['name', 'price', 'change_percent', 'volume', 'turnover', 'amplitude', 'leader_stock', 'leader_stock_price', 'leader_stock_change']
        
        return sector_data.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sector data: {str(e)}")

@router.get("/hot_sectors")
async def get_hot_sectors(limit: int = 10):
    """Get top performing sectors"""
    try:
        sectors = await get_market_sectors()
        
        # Sort by change_percent and get top sectors
        hot_sectors = sorted(sectors, key=lambda x: x['change_percent'], reverse=True)[:limit]
        
        return hot_sectors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching hot sectors: {str(e)}")

@router.get("/concept_stocks")
async def get_concept_stocks(concept: Optional[str] = None):
    """Get concept stocks and their performance"""
    try:
        if concept:
            # Get specific concept stocks
            concept_stocks = ak.stock_board_concept_cons_em(symbol=concept)
            
            # Rename columns for consistency
            concept_stocks.columns = ['code', 'name', 'price', 'change_percent', 'change', 'volume', 'turnover']
            
            return concept_stocks.to_dict(orient='records')
        else:
            # Get all concepts
            concepts = ak.stock_board_concept_name_em()
            
            # Rename columns for consistency
            concepts.columns = ['concept', 'change_percent', 'price', 'volume', 'turnover', 'amplitude', 'leader_stock', 'leader_price', 'leader_change']
            
            # Sort by change_percent
            concepts = concepts.sort_values('change_percent', ascending=False)
            
            return concepts.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching concept stocks: {str(e)}")

@router.get("/industry_stocks")
async def get_industry_stocks(industry: Optional[str] = None):
    """Get industry stocks and their performance"""
    try:
        if industry:
            # Get specific industry stocks
            industry_stocks = ak.stock_board_industry_cons_em(symbol=industry)
            
            # Rename columns for consistency
            industry_stocks.columns = ['code', 'name', 'price', 'change_percent', 'change', 'volume', 'turnover']
            
            return industry_stocks.to_dict(orient='records')
        else:
            # Get all industries
            industries = ak.stock_board_industry_name_em()
            
            # Rename columns for consistency
            industries.columns = ['industry', 'change_percent', 'price', 'volume', 'turnover', 'amplitude', 'leader_stock', 'leader_price', 'leader_change']
            
            # Sort by change_percent
            industries = industries.sort_values('change_percent', ascending=False)
            
            return industries.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching industry stocks: {str(e)}")

@router.get("/market_breadth")
async def get_market_breadth():
    """Get market breadth indicators (advance/decline)"""
    try:
        # Get A-share market data
        market_data = ak.stock_zh_a_spot_em()
        
        # Count advancing and declining stocks
        advancing = len(market_data[market_data['涨跌幅'] > 0])
        declining = len(market_data[market_data['涨跌幅'] < 0])
        unchanged = len(market_data[market_data['涨跌幅'] == 0])
        limit_up = len(market_data[market_data['涨跌幅'] >= 9.9])
        limit_down = len(market_data[market_data['涨跌幅'] <= -9.9])
        
        # Calculate advance-decline ratio
        ad_ratio = advancing / declining if declining > 0 else float('inf')
        
        # Calculate advance-decline line (simplified)
        ad_line = advancing - declining
        
        # Calculate market breadth indicator
        breadth = advancing / (advancing + declining) * 100 if (advancing + declining) > 0 else 50
        
        return {
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "ad_ratio": float(ad_ratio),
            "ad_line": ad_line,
            "breadth": float(breadth)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating market breadth: {str(e)}")

@router.get("/news")
async def get_market_news(category: str = "finance", limit: int = 20):
    """Get latest market news"""
    try:
        # Get market news based on category
        if category == "finance":
            news = ak.stock_zh_a_alerts_cls()
            # Rename columns for consistency
            news.columns = ['title', 'content', 'date', 'url', 'source']
        elif category == "research":
            news = ak.stock_research_report_em()
            # Rename columns for consistency
            news.columns = ['title', 'researcher', 'date', 'url', 'rating', 'type']
        else:
            raise HTTPException(status_code=400, detail=f"Invalid news category: {category}")
        
        # Limit the number of news items
        news = news.head(limit)
        
        return news.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market news: {str(e)}")

@router.get("/economic/indicators")
async def get_economic_indicators():
    """Get key economic indicators"""
    try:
        # Get various economic indicators
        indicators = {}
        
        # GDP growth
        try:
            gdp = ak.macro_china_gdp_yearly()
            indicators["gdp_growth"] = float(gdp.iloc[-1]["gdp"])
        except:
            indicators["gdp_growth"] = None
        
        # CPI
        try:
            cpi = ak.macro_china_cpi_yearly()
            indicators["cpi"] = float(cpi.iloc[-1]["value"])
        except:
            indicators["cpi"] = None
        
        # PPI
        try:
            ppi = ak.macro_china_ppi_yearly()
            indicators["ppi"] = float(ppi.iloc[-1]["value"])
        except:
            indicators["ppi"] = None
        
        # Money Supply (M2)
        try:
            m2 = ak.macro_china_money_supply()
            indicators["m2_growth"] = float(m2.iloc[-1]["m2_yoy"])
        except:
            indicators["m2_growth"] = None
        
        # Interest Rate
        try:
            interest_rate = ak.macro_china_lpr()
            indicators["lpr_1y"] = float(interest_rate.iloc[-1]["1y"])
            indicators["lpr_5y"] = float(interest_rate.iloc[-1]["5y"])
        except:
            indicators["lpr_1y"] = None
            indicators["lpr_5y"] = None
        
        return indicators
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching economic indicators: {str(e)}")

@router.get("/institutional/sentiment")
async def get_institutional_sentiment():
    """Get institutional investor sentiment metrics"""
    try:
        # Get northbound capital flow
        north_flow = ak.stock_em_hsgt_north_net_flow_in()
        
        # Get fund flows
        fund_flow = ak.fund_em_flow_category()
        
        # Process and calculate sentiment metrics
        today_flow = north_flow.iloc[-1]["value"] if not north_flow.empty else 0
        weekly_flow = north_flow.iloc[-5:]["value"].sum() if len(north_flow) >= 5 else north_flow["value"].sum()
        monthly_flow = north_flow.iloc[-20:]["value"].sum() if len(north_flow) >= 20 else north_flow["value"].sum()
        
        # Calculate sentiment indicator (simple version)
        sentiment = 0
        if today_flow > 0:
            sentiment += 1
        if weekly_flow > 0:
            sentiment += 1
        if monthly_flow > 0:
            sentiment += 1
            
        # Determine sentiment label
        sentiment_label = "neutral"
        if sentiment >= 2:
            sentiment_label = "bullish"
        elif sentiment <= 0:
            sentiment_label = "bearish"
        
        return {
            "north_flow_today": float(today_flow),
            "north_flow_weekly": float(weekly_flow),
            "north_flow_monthly": float(monthly_flow),
            "fund_flow": fund_flow.to_dict(orient='records') if not fund_flow.empty else [],
            "sentiment_score": sentiment,
            "sentiment": sentiment_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating institutional sentiment: {str(e)}")

@router.get("/calendar")
async def get_market_calendar(days: int = 7):
    """Get market events calendar for upcoming days"""
    try:
        # Get economic calendar
        calendar = ak.tool_trade_date_hist_sina()
        
        # Filter for the next specified days
        today = datetime.now().date()
        date_range = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        
        # Get IPO calendar
        try:
            ipo_calendar = ak.stock_em_ipo_declare()
            ipo_calendar = ipo_calendar[ipo_calendar['上市日期'].isin(date_range)]
            ipo_events = ipo_calendar.to_dict(orient='records')
        except:
            ipo_events = []
        
        # Get earnings release calendar (mocked data as akshare doesn't directly provide this)
        earnings_events = []
        
        # Combine all calendar events
        result = {
            "trading_days": [day for day in date_range if day in calendar['trade_date'].dt.strftime('%Y-%m-%d').tolist()],
            "ipo_events": ipo_events,
            "earnings_events": earnings_events
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market calendar: {str(e)}") 