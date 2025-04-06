from fastapi import APIRouter, HTTPException, Query
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import asyncio
import concurrent.futures

router = APIRouter(prefix="/api/market", tags=["market"])

@router.get("/sectors")
async def get_market_sectors():
    """Get performance of market sectors"""
    try:
        # Get sector performance data
        sector_data = ak.stock_sector_spot()
        
        # Clean and format the data - update column mapping based on actual data
        # First get original column names
        original_columns = sector_data.columns.tolist()
        
        # Create a mapping from actual columns to expected columns
        column_mapping = {
            'label': 'label',
            '板块': 'name',
            '公司家数': 'companies',
            '平均价格': 'price',
            '涨跌额': 'change',
            '涨跌幅': 'change_percent',
            '总成交量': 'volume',
            '总成交额': 'turnover',
            '股票代码': 'leader_code',
            '个股-涨跌幅': 'leader_change_percent', 
            '个股-当前价': 'leader_price',
            '个股-涨跌额': 'leader_change',
            '股票名称': 'leader_name'
        }
        
        # Rename columns using the mapping
        renamed_columns = {}
        for i, col in enumerate(original_columns):
            if col in column_mapping:
                renamed_columns[col] = column_mapping[col]
            else:
                renamed_columns[col] = f'column_{i}'
        
        # Apply the renaming
        sector_data = sector_data.rename(columns=renamed_columns)
        
        # Select needed columns
        needed_columns = ['name', 'price', 'change_percent', 'volume', 'turnover', 'leader_name', 'leader_price', 'leader_change_percent']
        result_columns = [col for col in needed_columns if col in sector_data.columns]
        
        # Ensure we have at least the most important columns
        if not all(col in sector_data.columns for col in ['name', 'change_percent']):
            # If we're missing critical columns, create a simplified dataframe
            clean_data = pd.DataFrame()
            clean_data['name'] = sector_data['板块'] if '板块' in sector_data.columns else sector_data.iloc[:, 1]
            clean_data['change_percent'] = sector_data['涨跌幅'] if '涨跌幅' in sector_data.columns else None
            return clean_data.to_dict(orient='records')
            
        return sector_data[result_columns].to_dict(orient='records')
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
        # Get A-share market data - 增加超时机制
        # 使用线程池运行同步阻塞操作
        def fetch_market_data():
            return ak.stock_zh_a_spot_em()
        
        # 设置超时为30秒
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            try:
                # 运行在线程中并设置超时
                market_data = await asyncio.wait_for(
                    loop.run_in_executor(executor, fetch_market_data),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                # 超时处理
                raise HTTPException(status_code=504, detail="Market data request timed out")
        
        # 确保数据有效
        if market_data is None or market_data.empty:
            raise HTTPException(status_code=500, detail="Empty market data returned")
        
        # 处理涨跌数据前确保列名正确
        change_column = '涨跌幅'
        if change_column not in market_data.columns:
            # 尝试查找替代列名
            possible_columns = [col for col in market_data.columns if '涨跌' in col or 'change' in col.lower()]
            if possible_columns:
                change_column = possible_columns[0]
            else:
                raise HTTPException(status_code=500, detail="Cannot find price change column in market data")
        
        # Count advancing and declining stocks
        advancing = len(market_data[market_data[change_column] > 0])
        declining = len(market_data[market_data[change_column] < 0])
        unchanged = len(market_data[market_data[change_column] == 0])
        limit_up = len(market_data[market_data[change_column] >= 9.9])
        limit_down = len(market_data[market_data[change_column] <= -9.9])
        
        # 安全计算，避免被零除错误
        ad_ratio = advancing / declining if declining > 0 else float('inf')
        ad_line = advancing - declining
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
        # 更详细的错误信息
        import traceback
        error_msg = f"Error calculating market breadth: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 打印到日志
        raise HTTPException(status_code=500, detail=f"Error calculating market breadth: {str(e)}")

@router.get("/news")
async def get_market_news(category: str = "finance", limit: int = 20):
    """Get latest market news"""
    try:
        # Get market news based on category
        if category == "finance":
            news = ak.stock_news_em()
            # Rename columns for consistency
            news.columns = ['keyword', 'title', 'content', 'date', 'source', 'url']
            # Select needed columns and rename them
            news = news[['title', 'content', 'date', 'url', 'source']]
        elif category == "research":
            # Try to find an alternative for research reports
            try:
                news = ak.stock_research_report_em()
                # Rename columns for consistency
                news.columns = ['title', 'researcher', 'date', 'url', 'rating', 'type']
            except:
                # Fallback to regular news if research specific API is not available
                news = ak.stock_news_em()
                news.columns = ['keyword', 'title', 'content', 'date', 'source', 'url']
                news = news[['title', 'content', 'date', 'url', 'source']]
                news['researcher'] = news['source']
                news['rating'] = 'N/A'
                news['type'] = 'general'
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
        
        # GDP growth - akshare may have changed the API
        try:
            # Try a different approach since macro_china_gdp_yearly might not work
            # Use a placeholder for now (or we could implement a scraper for this data)
            indicators["gdp_growth"] = None
        except Exception as e:
            print(f"GDP error: {str(e)}")
            indicators["gdp_growth"] = None
        
        # CPI
        try:
            cpi = ak.macro_china_cpi_yearly()
            # The column '今值' (current value) contains the CPI value
            if not cpi.empty and '今值' in cpi.columns:
                value = cpi.iloc[-1]["今值"]
                indicators["cpi"] = float(value) if pd.notna(value) else None
            else:
                indicators["cpi"] = None
        except Exception as e:
            print(f"CPI error: {str(e)}")
            indicators["cpi"] = None
        
        # PPI
        try:
            ppi = ak.macro_china_ppi_yearly()
            # The column '今值' (current value) contains the PPI value
            if not ppi.empty and '今值' in ppi.columns:
                value = ppi.iloc[-1]["今值"]
                indicators["ppi"] = float(value) if pd.notna(value) else None
            else:
                indicators["ppi"] = None
        except Exception as e:
            print(f"PPI error: {str(e)}")
            indicators["ppi"] = None
        
        # Money Supply (M2)
        try:
            m2 = ak.macro_china_money_supply()
            # According to our test, the column for M2 YoY growth is '货币和准货币(M2)-同比增长'
            if not m2.empty and '货币和准货币(M2)-同比增长' in m2.columns:
                value = m2.iloc[-1]["货币和准货币(M2)-同比增长"]
                indicators["m2_growth"] = float(value) if pd.notna(value) else None
            else:
                indicators["m2_growth"] = None
        except Exception as e:
            print(f"M2 error: {str(e)}")
            indicators["m2_growth"] = None
        
        # Interest Rate (LPR)
        try:
            interest_rate = ak.macro_china_lpr()
            # According to our test, the columns for LPR are 'LPR1Y' and 'LPR5Y'
            if not interest_rate.empty:
                # Find the last non-null LPR1Y value
                non_null_lpr1y = interest_rate[interest_rate['LPR1Y'].notna()]
                if not non_null_lpr1y.empty:
                    value = non_null_lpr1y.iloc[-1]["LPR1Y"]
                    indicators["lpr_1y"] = float(value) if pd.notna(value) else None
                else:
                    indicators["lpr_1y"] = None
                
                # Find the last non-null LPR5Y value
                non_null_lpr5y = interest_rate[interest_rate['LPR5Y'].notna()]
                if not non_null_lpr5y.empty:
                    value = non_null_lpr5y.iloc[-1]["LPR5Y"]
                    indicators["lpr_5y"] = float(value) if pd.notna(value) else None
                else:
                    indicators["lpr_5y"] = None
            else:
                indicators["lpr_1y"] = None
                indicators["lpr_5y"] = None
        except Exception as e:
            print(f"LPR error: {str(e)}")
            indicators["lpr_1y"] = None
            indicators["lpr_5y"] = None
        
        # Replace any NaN or infinite values with None for JSON compatibility
        for key in indicators:
            if indicators[key] is not None:
                try:
                    # 检查是否为NaN或者无穷大
                    if pd.isna(indicators[key]) or (isinstance(indicators[key], float) and (indicators[key] == float('inf') or indicators[key] == float('-inf'))):
                        indicators[key] = None
                except:
                    # 如果出现任何错误，确保值为JSON兼容
                    try:
                        # 尝试转换为float
                        indicators[key] = float(indicators[key])
                    except:
                        # 如果无法转换，设为None
                        indicators[key] = None
        
        return indicators
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching economic indicators: {str(e)}")

@router.get("/institutional/sentiment")
async def get_institutional_sentiment():
    """Get institutional investor sentiment metrics"""
    try:
        # Use available functions for northbound capital flow
        try:
            # Try using stock_em_hsgt_board_em as alternative
            north_flow_data = ak.stock_em_hsgt_board_em()
            # Create a simple dataframe with value column for northbound flow
            north_flow = pd.DataFrame({
                'date': pd.to_datetime(north_flow_data['日期']),
                'value': north_flow_data['北向资金(亿)'].astype(float)
            })
        except:
            # If not available, create dummy data
            dates = [(datetime.now().date() - timedelta(days=i)) for i in range(30)]
            north_flow = pd.DataFrame({
                'date': dates,
                'value': [0] * len(dates)  # All zeros as placeholder
            })
        
        # Get fund flows - using an alternative or mock data
        try:
            # Try to use fund_em_flow_big_deal if available
            fund_flow = ak.fund_em_flow_big_deal()
        except:
            # Mock data if not available
            fund_flow = pd.DataFrame([
                {'category': 'Stock', 'value': 0, 'percent': 0},
                {'category': 'Bond', 'value': 0, 'percent': 0},
                {'category': 'Hybrid', 'value': 0, 'percent': 0}
            ])
        
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
        
        # Ensure trade_date is datetime type
        if not pd.api.types.is_datetime64_any_dtype(calendar['trade_date']):
            calendar['trade_date'] = pd.to_datetime(calendar['trade_date'])
        
        # Filter for the next specified days
        today = datetime.now().date()
        date_range = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        
        # Get trading days
        trading_days = [day for day in date_range 
                        if day in calendar['trade_date'].dt.strftime('%Y-%m-%d').tolist()]
        
        # Get IPO calendar - using alternative or mock data if not available
        try:
            # Try using stock_ipo_info function as an alternative
            ipo_data = ak.stock_ipo_info()
            ipo_events = ipo_data.to_dict(orient='records')
        except:
            # If no IPO data function is available, provide empty list
            ipo_events = []
        
        # Get earnings release calendar (mocked data as akshare doesn't directly provide this)
        earnings_events = []
        
        # Combine all calendar events
        result = {
            "trading_days": trading_days,
            "ipo_events": ipo_events,
            "earnings_events": earnings_events
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market calendar: {str(e)}") 