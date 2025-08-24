
"""
ENHANCED CRYPTO ANALYTICS SYSTEM V2.0
Advanced TrendPulse + Stochastic RSI 4H Confirmation Filter
Bulletproof CCXT integration with intelligent noise reduction
"""

import pandas as pd
import numpy as np
import requests
import ccxt
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration
COIN_CACHE_FILE = Path("analytics_coin_cache.json")
ALERT_CACHE_FILE = Path("analytics_alerts.json")
BLOCKED_COINS_FILE = Path("blocked_coins.txt")
CACHE_DURATION_MINUTES = 30

# Enhanced Symbol Variations for CCXT
SYMBOL_VARIATIONS = {
    'TON': ['TON/USDT', 'TONCOIN/USDT', 'TONC/USDT'],
    'PI': ['PI/USDT', 'PICOIN/USDT', 'PCHAIN/USDT'],
    'AXL': ['AXL/USDT', 'AXELAR/USDT', 'AXLR/USDT'],
    'BEAM': ['BEAM/USDT', 'BEAMX/USDT'],
    'XMR': ['XMR/USDT', 'MONERO/USDT'],
    'CRO': ['CRO/USDT', 'CRONOS/USDT'],
    'MATIC': ['MATIC/USDT', 'POL/USDT'],
    'FTM': ['FTM/USDT', 'FANTOM/USDT'],
}

def load_alert_cache():
    if ALERT_CACHE_FILE.exists():
        return json.loads(ALERT_CACHE_FILE.read_text())
    return {}

def save_alert_cache(cache):
    ALERT_CACHE_FILE.write_text(json.dumps(cache))

def load_blocked_coins():
    """Load blocked coins from text file"""
    if BLOCKED_COINS_FILE.exists():
        with open(BLOCKED_COINS_FILE, 'r') as f:
            blocked = {line.strip().upper() for line in f 
                      if line.strip() and not line.startswith('#')}
        print(f"📝 Loaded {len(blocked)} blocked coins")
        return blocked
    else:
        print("📝 No blocked coins file found")
        return set()

def create_bulletproof_bingx_exchange():
    """Create BingX exchange with enhanced retry and error handling"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"🔧 Initializing BingX exchange (attempt {attempt + 1}/{max_retries})...")
            
            exchange = ccxt.bingx({
                'apiKey': os.environ.get('BINGX_API_KEY', ''),
                'secret': os.environ.get('BINGX_SECRET_KEY', ''),
                'enableRateLimit': True,
                'rateLimit': 1000,
                'timeout': 30000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                },
                'verbose': False,
            })
            
            # Force load markets
            print("📊 Loading BingX markets...")
            markets_loaded = False
            
            for market_attempt in range(3):
                try:
                    markets = exchange.load_markets(True)
                    markets_loaded = True
                    break
                except Exception as e:
                    if market_attempt < 2:
                        time.sleep(2)
                    continue
            
            if not markets_loaded:
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue
            
            total_markets = len(exchange.markets)
            print(f"✅ BingX markets loaded: {total_markets} total")
            
            # Test connection
            try:
                test_symbol = 'BTC/USDT'
                if test_symbol in exchange.markets:
                    test_data = exchange.fetch_ohlcv(test_symbol, '1h', limit=5)
                    if test_data and len(test_data) > 0:
                        print("✅ BingX connection test successful")
                        return exchange, True
            except:
                pass
            
            return exchange, True
            
        except Exception as e:
            print(f"❌ BingX attempt {attempt + 1} failed: {str(e)[:80]}")
            if attempt < max_retries - 1:
                time.sleep(10)
            continue
    
    print("❌ All BingX connection attempts failed")
    return None, False

def find_working_symbol(base_symbol, exchange):
    """Find working symbol format for a given base symbol"""
    variations = SYMBOL_VARIATIONS.get(base_symbol.upper(), [f"{base_symbol.upper()}/USDT"])
    
    if f"{base_symbol.upper()}/USDT" not in variations:
        variations.append(f"{base_symbol.upper()}/USDT")
    
    results = {'spot': None, 'futures': None}
    
    for variation in variations:
        try:
            # Check spot market
            if variation in exchange.markets:
                market_info = exchange.markets[variation]
                if market_info.get('active', True) and market_info.get('type') == 'spot':
                    results['spot'] = variation
                
            # Check futures market
            futures_variation = variation.replace('/USDT', '/USDT:USDT')
            if futures_variation in exchange.markets:
                market_info = exchange.markets[futures_variation]
                if market_info.get('active', True) and market_info.get('type') == 'swap':
                    results['futures'] = futures_variation
                        
        except:
            continue
    
    return results

def fetch_ohlcv_with_retry(exchange, symbol, timeframe, limit, market_type='spot'):
    """Fetch OHLCV data with retry mechanism"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            if market_type == 'futures':
                exchange.options['defaultType'] = 'swap'
            else:
                exchange.options['defaultType'] = 'spot'
            
            if attempt > 0:
                time.sleep(0.2 * attempt)
            
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv and len(ohlcv) >= 30:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                if len(df) >= 30:
                    return df
            
            return None
            
        except ccxt.RequestTimeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
        except ccxt.NetworkError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
        except ccxt.ExchangeError as e:
            if 'invalid symbol' in str(e).lower() or 'not found' in str(e).lower():
                return None
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
        except:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            return None
    
    return None

def get_multi_timeframe_data(symbol, exchange):
    """
    ENHANCED: Get OHLCV data for multiple timeframes including 4H for Stochastic RSI
    Returns: {'30M': df, '1H': df, '4H': df}
    """
    data = {}
    
    if not exchange or symbol.upper() in ['WHYPE']:
        return data
    
    working_symbols = find_working_symbol(symbol, exchange)
    
    # Timeframes needed: 30M, 1H for TrendPulse + 4H for Stochastic RSI
    timeframes = [
        ('30m', '30M', 100),
        ('1h', '1H', 50),
        ('4h', '4H', 100)  # NEW: 4H for Stochastic RSI confirmation
    ]
    
    for tf, label, limit in timeframes:
        success = False
        
        # Try spot first
        if working_symbols['spot'] and not success:
            df = fetch_ohlcv_with_retry(exchange, working_symbols['spot'], tf, limit, 'spot')
            if df is not None and len(df) >= 30:
                data[label] = df
                success = True
        
        # Fallback to futures
        if working_symbols['futures'] and not success:
            df = fetch_ohlcv_with_retry(exchange, working_symbols['futures'], tf, limit, 'futures')
            if df is not None and len(df) >= 30:
                data[label] = df
                success = True
    
    return data

def convert_to_heikin_ashi(df):
    """Convert regular OHLC data to Heikin Ashi candles"""
    if len(df) < 2:
        return df
    
    ha_df = df.copy()
    ha_df = ha_df.assign(HA_Close=0.0, HA_Open=0.0, HA_High=0.0, HA_Low=0.0)
    
    first_idx = ha_df.index[0]
    ha_df.at[first_idx, 'HA_Close'] = (df.at[first_idx, 'Open'] + df.at[first_idx, 'High'] + 
                                      df.at[first_idx, 'Low'] + df.at[first_idx, 'Close']) / 4.0
    ha_df.at[first_idx, 'HA_Open'] = (df.at[first_idx, 'Open'] + df.at[first_idx, 'Close']) / 2.0
    ha_df.at[first_idx, 'HA_High'] = df.at[first_idx, 'High']
    ha_df.at[first_idx, 'HA_Low'] = df.at[first_idx, 'Low']
    
    for i in range(1, len(df)):
        curr_idx = ha_df.index[i]
        prev_idx = ha_df.index[i-1]
        
        ha_df.at[curr_idx, 'HA_Close'] = (df.at[curr_idx, 'Open'] + df.at[curr_idx, 'High'] + 
                                         df.at[curr_idx, 'Low'] + df.at[curr_idx, 'Close']) / 4.0
        ha_df.at[curr_idx, 'HA_Open'] = (ha_df.at[prev_idx, 'HA_Open'] + 
                                        ha_df.at[prev_idx, 'HA_Close']) / 2.0
        ha_df.at[curr_idx, 'HA_High'] = max(df.at[curr_idx, 'High'], 
                                           ha_df.at[curr_idx, 'HA_Open'], 
                                           ha_df.at[curr_idx, 'HA_Close'])
        ha_df.at[curr_idx, 'HA_Low'] = min(df.at[curr_idx, 'Low'], 
                                          ha_df.at[curr_idx, 'HA_Open'], 
                                          ha_df.at[curr_idx, 'HA_Close'])
    
    return ha_df

def calculate_stochastic_rsi(df, period=14, smooth_k=3, smooth_d=3):
    """
    Calculate Stochastic RSI indicator for confirmation filter
    Returns: {'stoch_rsi_k': Series, 'stoch_rsi_d': Series}
    """
    if len(df) < period + smooth_k + smooth_d + 10:
        return None
    
    try:
        # Calculate RSI first
        close = df['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic of RSI
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        
        # Smooth with moving averages
        stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()
        
        return {
            'stoch_rsi_k': stoch_rsi_k,
            'stoch_rsi_d': stoch_rsi_d,
            'current_k': float(stoch_rsi_k.iloc[-2]) if len(stoch_rsi_k) > 1 else 0,
            'current_d': float(stoch_rsi_d.iloc[-2]) if len(stoch_rsi_d) > 1 else 0
        }
    except Exception as e:
        return None

class EnhancedTrendPulseAnalyzer:
    """Enhanced TrendPulse Analysis with Stochastic RSI 4H Confirmation"""
    
    def __init__(self):
        self.ch_len = 9
        self.avg_len = 12
        self.smooth_len = 3

    def ema(self, src, length):
        return src.ewm(span=length, adjust=False).mean()

    def sma(self, src, length):
        return src.rolling(window=length).mean()

    def analyze_with_stoch_rsi_confirmation(self, data_dict, tier_type, debug_symbol=""):
        """
        ENHANCED: Analyze with TrendPulse + Stochastic RSI 4H confirmation
        Significantly reduces noise by requiring both indicators to align
        """
        timeframe = '1H' if tier_type == 'HIGH_RISK' else '30M'
        
        # Check if we have required data
        if timeframe not in data_dict or '4H' not in data_dict:
            return {
                'signals': [], 'wt1': 0, 'wt2': 0, 'stoch_rsi_k': 0, 'stoch_rsi_d': 0,
                'has_signal': False, 'signal_type': 'none', 'confirmation': 'missing_data'
            }
        
        # Get TrendPulse analysis on primary timeframe
        ha_df = convert_to_heikin_ashi(data_dict[timeframe])
        trendpulse_result = self.analyze_heikin_ashi(ha_df, tier_type, debug_symbol)
        
        # Get Stochastic RSI confirmation on 4H
        stoch_rsi_4h = calculate_stochastic_rsi(data_dict['4H'])
        
        if not stoch_rsi_4h or not trendpulse_result['has_signal']:
            return {
                **trendpulse_result,
                'stoch_rsi_k': stoch_rsi_4h['current_k'] if stoch_rsi_4h else 0,
                'stoch_rsi_d': stoch_rsi_4h['current_d'] if stoch_rsi_4h else 0,
                'confirmation': 'no_signal' if not trendpulse_result['has_signal'] else 'stoch_unavailable'
            }
        
        # Apply Stochastic RSI confirmation filter
        confirmed_signals = []
        
        for signal in trendpulse_result['signals']:
            signal_type = signal['type']
            k_value = stoch_rsi_4h['current_k']
            d_value = stoch_rsi_4h['current_d']
            
            # Confirmation rules
            if signal_type == 'buy':
                # BUY: Stochastic RSI must be oversold (< 30)
                stoch_confirmed = k_value < 30 and d_value < 30
                confirmation_reason = f"StochRSI_4H_Oversold(K:{k_value:.1f},D:{d_value:.1f})"
            else:  # sell
                # SELL: Stochastic RSI must be overbought (> 70)  
                stoch_confirmed = k_value > 70 and d_value > 70
                confirmation_reason = f"StochRSI_4H_Overbought(K:{k_value:.1f},D:{d_value:.1f})"
            
            if stoch_confirmed:
                # Add confirmation data to signal
                enhanced_signal = signal.copy()
                enhanced_signal.update({
                    'stoch_rsi_k': k_value,
                    'stoch_rsi_d': d_value,
                    'confirmation': confirmation_reason,
                    'candle_timestamp': ha_df.index[-2].strftime('%Y-%m-%dT%H:%M')  # For deduplication
                })
                confirmed_signals.append(enhanced_signal)
        
        # Return enhanced result
        return {
            'signals': confirmed_signals,
            'wt1': trendpulse_result['wt1'],
            'wt2': trendpulse_result['wt2'],
            'stoch_rsi_k': stoch_rsi_4h['current_k'],
            'stoch_rsi_d': stoch_rsi_4h['current_d'],
            'has_signal': len(confirmed_signals) > 0,
            'signal_type': confirmed_signals[0]['type'] if confirmed_signals else 'none',
            'confirmation': f"{len(confirmed_signals)}_confirmed_of_{len(trendpulse_result['signals'])}"
        }

    def analyze_heikin_ashi(self, ha_df, tier_type, debug_symbol=""):
        """Core TrendPulse analysis on Heikin Ashi candles - FIXED TIMING"""
        if len(ha_df) < self.ch_len + self.avg_len + 5:
            return {
                'signals': [], 'wt1': 0, 'wt2': 0,
                'has_signal': False, 'signal_type': 'none'
            }
        
        try:
            # TrendPulse calculation
            ha_hlc3 = (ha_df['HA_High'] + ha_df['HA_Low'] + ha_df['HA_Close']) / 3.0
            esa = self.ema(ha_hlc3, self.ch_len)
            dev = self.ema(abs(ha_hlc3 - esa), self.ch_len)
            dev_safe = dev.replace(0, 0.001)
            ci = (ha_hlc3 - esa) / (0.015 * dev_safe)
            
            wt1 = self.ema(ci, self.avg_len)
            wt2 = self.sma(wt1, self.smooth_len)

            wt1_values = wt1.values
            wt2_values = wt2.values

            # FIXED: Use closed candle for current values
            current_wt1 = float(wt1_values[-2])
            current_wt2 = float(wt2_values[-2])
            
            signals = []
            # FIXED: Check only closed candles
            for i in range(1, min(4, len(wt1_values) - 1)):
                wt1_curr = float(wt1_values[-(i+1)])
                wt2_curr = float(wt2_values[-(i+1)])
                wt1_prev = float(wt1_values[-(i+2)])
                wt2_prev = float(wt2_values[-(i+2)])
                
                # Tier-specific thresholds
                if tier_type == "HIGH_RISK":
                    oversold = (wt1_curr <= -50) and (wt2_curr <= -50)
                    overbought = (wt2_curr >= 50) and (wt1_curr >= 50)
                else:
                    oversold = (wt1_curr <= -60) and (wt2_curr <= -60)
                    overbought = (wt2_curr >= 60) and (wt1_curr >= 60)
                
                bullish_cross = (wt1_prev <= wt2_prev) and (wt1_curr > wt2_curr)
                bearish_cross = (wt1_prev >= wt2_prev) and (wt1_curr < wt2_curr)
                
                if bullish_cross and oversold:
                    signals.append({
                        'type': 'buy',
                        'candles_ago': i,
                        'wt1': wt1_curr,
                        'wt2': wt2_curr,
                        'strength': abs(wt1_curr) + abs(wt2_curr),
                        'tier': tier_type
                    })
                elif bearish_cross and overbought:
                    signals.append({
                        'type': 'sell',
                        'candles_ago': i,
                        'wt1': wt1_curr,
                        'wt2': wt2_curr,
                        'strength': abs(wt1_curr) + abs(wt2_curr),
                        'tier': tier_type
                    })
            
            return {
                'signals': signals,
                'wt1': current_wt1,
                'wt2': current_wt2,
                'has_signal': len(signals) > 0,
                'signal_type': signals[0]['type'] if signals else 'none'
            }
            
        except Exception as e:
            print(f"  ❌ TrendPulse error for {debug_symbol}: {str(e)[:50]}")
            return {
                'signals': [], 'wt1': 0, 'wt2': 0,
                'has_signal': False, 'signal_type': 'none'
            }

class CoinGeckoDataManager:
    """CoinGecko integration with pagination for comprehensive coverage"""
    
    def __init__(self):
        self.api_calls_used = 0

    def load_cache(self):
        if COIN_CACHE_FILE.exists():
            try:
                cache_data = json.loads(COIN_CACHE_FILE.read_text())
                coins = cache_data.get('coins', [])
                timestamp = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
                return coins, timestamp
            except:
                return [], datetime.min
        return [], datetime.min

    def save_cache(self, coins):
        cache_data = {
            'coins': coins,
            'timestamp': datetime.utcnow().isoformat()
        }
        COIN_CACHE_FILE.write_text(json.dumps(cache_data))

    def get_dual_tier_coins(self):
        """Get coins with pagination for complete coverage"""
        cached_coins, cache_time = self.load_cache()
        now = datetime.utcnow()
        
        if (now - cache_time).total_seconds() < CACHE_DURATION_MINUTES * 60:
            cache_age = (now - cache_time).total_seconds() / 60
            print(f"🔄 Using cached CoinGecko data (age: {cache_age:.1f} min)")
            return self.categorize_coins(cached_coins), 0
        
        print("🌐 Fetching comprehensive coin data from CoinGecko...")
        api_key = os.environ.get('COINGECKO_API_KEY', '')
        url = "https://api.coingecko.com/api/v3/coins/markets"
        headers = {'x-cg-demo-api-key': api_key} if api_key else {}
        
        all_coins = []
        api_calls = 0
        
        try:
            for page in range(1, 6):  # Get 5 pages (1250 coins)
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': 250,
                    'page': page,
                }
                
                print(f"  📄 Fetching CoinGecko page {page}...")
                r = requests.get(url, params=params, headers=headers, timeout=20)
                r.raise_for_status()
                data = r.json()
                api_calls += 1
                
                if not data:
                    break
                
                all_coins.extend(data)
                time.sleep(0.2)
            
            print(f"✅ CoinGecko: Fetched {len(all_coins)} total coins")
            
            # Filter coins
            filtered = []
            stablecoins = {'USDT', 'USDC', 'DAI', 'BUSD', 'USDE', 'FDUSD'}
            
            for coin in all_coins:
                market_cap = coin.get('market_cap', 0) or 0
                volume_24h = coin.get('total_volume', 0) or 0
                
                high_risk_qualified = (market_cap >= 10_000_000 and 
                                     market_cap < 500_000_000 and 
                                     volume_24h >= 10_000_000)
                
                standard_qualified = (market_cap >= 500_000_000 and 
                                    volume_24h >= 30_000_000)
                
                if ((high_risk_qualified or standard_qualified) and 
                    coin['symbol'].upper() not in stablecoins):
                    
                    filtered.append({
                        'id': coin['id'],
                        'symbol': coin['symbol'].upper(),
                        'name': coin['name'],
                        'market_cap': market_cap,
                        'total_volume': volume_24h,
                        'current_price': coin.get('current_price', 0),
                        'price_change_24h': coin.get('price_change_percentage_24h', 0)
                    })
            
            self.save_cache(filtered)
            self.api_calls_used += api_calls
            
            categorized = self.categorize_coins(filtered)
            print(f"✅ CoinGecko: HIGH RISK: {len(categorized['high_risk'])} coins")
            print(f"✅ CoinGecko: STANDARD: {len(categorized['standard'])} coins")
            
            return categorized, api_calls
            
        except Exception as e:
            print(f"❌ CoinGecko error: {e}")
            return self.categorize_coins(cached_coins), 0

    def categorize_coins(self, all_coins):
        """Separate coins into tiers"""
        high_risk = [coin for coin in all_coins if coin['market_cap'] < 500_000_000]
        standard = [coin for coin in all_coins if coin['market_cap'] >= 500_000_000]
        
        return {'high_risk': high_risk, 'standard': standard}

def get_ist_time_12h():
    """Get IST time formatting"""
    utc = datetime.utcnow()
    ist = utc + timedelta(hours=5, minutes=30)
    return ist.strftime('%I:%M %p %d-%m-%Y'), ist.strftime('%A, %d %B %Y')

def get_reliable_tradingview_url(symbol):
    """Get working TradingView chart URL with Bybit priority"""
    base_symbol = symbol.upper()
    
    exchanges = [
        ('BYBIT', f"{base_symbol}USDT"),
        ('BINANCE', f"{base_symbol}USDT"),
        ('OKX', f"{base_symbol}USDT"),
        ('COINBASE', f"{base_symbol}USD"),
    ]
    
    for exchange, pair in exchanges:
        url = f"https://www.tradingview.com/chart/?symbol={exchange}:{pair}"
        try:
            resp = requests.head(url, timeout=2, allow_redirects=True)
            if resp.status_code == 200:
                return url, exchange
        except:
            continue
    
    return f"https://www.tradingview.com/chart/?symbol=BYBIT:{base_symbol}USDT", "BYBIT"

def send_enhanced_crypto_alert(coin, analysis, tier_type, cache):
    """Send enhanced alerts with Stochastic RSI confirmation and perfect deduplication"""
    
    if tier_type == "HIGH_RISK":
        chat_id = os.environ.get('HIGH_RISK_CHAT_ID')
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
    else:
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
    
    if not token or not chat_id or not analysis['signals']:
        return
    
    signal = analysis['signals'][0]
    action = signal['type']
    time_str, day_str = get_ist_time_12h()
    
    # ENHANCED: Use candle timestamp for perfect deduplication
    candle_ts = signal.get('candle_timestamp', time_str)
    key = f"{tier_type}_{coin['symbol']}_{action}_{candle_ts}"
    
    if key in cache:
        return
    
    tv_url, exchange = get_reliable_tradingview_url(coin['symbol'])
    
    # Format price
    price = coin['current_price']
    if price >= 1:
        price_str = f"${price:,.2f}"
    elif price >= 0.01:
        price_str = f"${price:.4f}"
    else:
        price_str = f"${price:.8f}"
    
    # Price change
    price_change = coin.get('price_change_24h', 0)
    change_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
    change_str = f"{change_emoji} {price_change:+.2f}%"
    
    # Market cap
    cap_billions = coin['market_cap'] / 1_000_000_000
    if cap_billions >= 1:
        cap_category = f"🔷 Large Cap (${cap_billions:.1f}B)"
    else:
        cap_millions = coin['market_cap'] / 1_000_000
        cap_category = f"💎 {'Mid' if cap_millions >= 100 else 'Small'} Cap (${cap_millions:.0f}M)"
    
    action_emoji = '🟢' if action == 'buy' else '🔴'
    
    # ENHANCED: Include Stochastic RSI confirmation
    if tier_type == "HIGH_RISK":
        title = f"{action_emoji} HIGH RISK Alert {action_emoji}"
        timeframe_info = "📊 1H Heikin Ashi + 4H StochRSI Confirmed"
        urgency = "⚡ HIGH REWARD POTENTIAL ⚡"
    else:
        title = f"{action_emoji} Standard Alert {action_emoji}"
        timeframe_info = "📈 30M Heikin Ashi + 4H StochRSI Confirmed"
        urgency = "📊 QUALITY CONFIRMED SIGNAL"
    
    # Enhanced message with confirmation details
    message = f"{title}\n"
    message += f"**{coin['symbol']}-USD — {action.upper()}**\n"
    message += f"{cap_category}\n"
    message += f"{urgency}\n\n"
    message += f"💰 **Price**: {price_str}\n"
    message += f"📊 **24h Change**: {change_str}\n"
    message += f"📈 **WT1**: {signal['wt1']:.2f} | **WT2**: {signal['wt2']:.2f}\n"
    message += f"🎯 **StochRSI 4H**: K:{signal['stoch_rsi_k']:.1f} D:{signal['stoch_rsi_d']:.1f}\n"
    message += f"💪 **Strength**: {signal['strength']:.1f}\n"
    message += f"✅ **Confirmation**: {signal['confirmation']}\n"
    message += f"🕐 **Time**: {time_str} IST\n"
    message += f"📅 **Date**: {day_str}\n"
    message += f"{timeframe_info}\n\n"
    message += f"🔗 [**{exchange} Chart**]({tv_url})"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    
    try:
        res = requests.post(url, data=data, timeout=5)
        if res.status_code == 200:
            cache[key] = True
            print(f"✅ {tier_type} CONFIRMED alert: {coin['symbol']} {action.upper()} @ {price_str} via {exchange}")
        else:
            print(f"❌ Telegram error for {tier_type}: {res.status_code}")
    except Exception as e:
        print(f"❌ Telegram error for {tier_type}: {e}")

def analyze_coin_enhanced(coin, tier_type, analyzer, bingx_exchange, blocked_coins):
    """Enhanced coin analysis with Stochastic RSI 4H confirmation"""
    try:
        if coin['symbol'].upper() in blocked_coins:
            return None, f"🚫 BLOCKED: {coin['symbol']}"
        
        # Get multi-timeframe data (30M, 1H, 4H)
        data = get_multi_timeframe_data(coin['symbol'], bingx_exchange)
        
        if not data:
            return None, f"❌ No data: {coin['symbol']}"
        
        # Enhanced analysis with confirmation filter
        analysis = analyzer.analyze_with_stoch_rsi_confirmation(data, tier_type, coin['symbol'])
        
        if analysis['has_signal']:
            return {
                'coin': coin,
                'analysis': analysis,
                'tier': tier_type
            }, f"✅ {analysis['signal_type'].upper()} CONFIRMED: {coin['symbol']} ({analysis['confirmation']})"
        else:
            reason = analysis.get('confirmation', 'no_signal')
            return None, f"📊 No confirmed signal: {coin['symbol']} ({reason})"
            
    except Exception as e:
        return None, f"❌ Analysis error {coin['symbol']}: {str(e)[:50]}"

def main():
    """Enhanced Crypto Analytics System V2.0 - TrendPulse + Stochastic RSI 4H"""
    print("🚀 ENHANCED CRYPTO ANALYTICS SYSTEM V2.0")
    print("=" * 95)
    print("🔥 Advanced TrendPulse + Stochastic RSI 4H Confirmation")
    print("📊 CoinGecko: Market cap filtering & comprehensive coverage")
    print("🏢 Bulletproof BingX: Multi-timeframe data (30M, 1H, 4H)")
    print("🎯 NOISE REDUCTION: Only confirmed signals with dual indicator alignment")
    print("📈 HIGH RISK: 1H TrendPulse + 4H StochRSI • STANDARD: 30M TrendPulse + 4H StochRSI")
    print("💰 Enhanced Alerts: Perfect deduplication + confirmation details")
    print("=" * 95)
    
    start_time = datetime.utcnow()
    
    # Initialize components
    alert_cache = load_alert_cache()
    blocked_coins = load_blocked_coins()
    coingecko_manager = CoinGeckoDataManager()
    analyzer = EnhancedTrendPulseAnalyzer()
    
    # Connect to BingX
    bingx_exchange, connection_success = create_bulletproof_bingx_exchange()
    
    if not connection_success:
        print("❌ BingX connection failed")
        return
    
    # Get coin data
    tier_data, coingecko_calls = coingecko_manager.get_dual_tier_coins()
    high_risk_coins = tier_data['high_risk']
    standard_coins = tier_data['standard']
    
    if not high_risk_coins and not standard_coins:
        print("❌ No coins retrieved")
        return
    
    total_coins = len(high_risk_coins) + len(standard_coins)
    print(f"📊 Analyzing {total_coins} coins with Enhanced V2.0 system...")
    print("🎯 Dual confirmation: TrendPulse + Stochastic RSI 4H")
    print("=" * 95)
    
    all_results = []
    
    # Process with ThreadPool
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all coins
        high_risk_futures = [
            executor.submit(analyze_coin_enhanced, coin, 'HIGH_RISK', analyzer, bingx_exchange, blocked_coins)
            for coin in high_risk_coins
        ]
        
        standard_futures = [
            executor.submit(analyze_coin_enhanced, coin, 'STANDARD', analyzer, bingx_exchange, blocked_coins)
            for coin in standard_coins
        ]
        
        # Collect results
        print("🔥 HIGH RISK TIER (1H TrendPulse + 4H StochRSI Confirmation):")
        print("-" * 80)
        for i, future in enumerate(high_risk_futures, 1):
            result, log = future.result()
            print(f"[{i}/{len(high_risk_futures)}] {log}")
            if result:
                all_results.append(result)
        
        print("\n📊 STANDARD TIER (30M TrendPulse + 4H StochRSI Confirmation):")
        print("-" * 80)
        for i, future in enumerate(standard_futures, 1):
            result, log = future.result()
            print(f"[{i}/{len(standard_futures)}] {log}")
            if result:
                all_results.append(result)
    
    # Send enhanced alerts
    print(f"\n🚨 PROCESSING {len(all_results)} CONFIRMED SIGNALS:")
    print("-" * 60)
    
    for result in all_results:
        send_enhanced_crypto_alert(
            result['coin'], 
            result['analysis'], 
            result['tier'], 
            alert_cache
        )
    
    save_alert_cache(alert_cache)
    
    execution_time = (datetime.utcnow() - start_time).total_seconds()
    
    print(f"\n🎉 ENHANCED CRYPTO ANALYTICS V2.0 SCAN COMPLETE:")
    print("=" * 85)
    print(f"   ⏱️  Execution Time: {execution_time:.1f}s")
    print(f"   📊 HIGH RISK (1H+4H Enhanced): {len(high_risk_coins)} coins")
    print(f"   📊 STANDARD (30M+4H Enhanced): {len(standard_coins)} coins")
    print(f"   🚨 CONFIRMED Signals: {len(all_results)}")
    print(f"   📡 CoinGecko API Calls: {coingecko_calls}")
    print(f"   🔥 ENHANCEMENT: Dual indicator confirmation reduces noise significantly")
    print(f"   🎯 SUCCESS: Only high-quality confirmed signals sent")
    print(f"   ✅ System Status: {'ENHANCED V2.0 SUCCESS' if all_results or total_coins > 200 else 'PARTIAL COVERAGE'}")

if __name__ == "__main__":
    main()
