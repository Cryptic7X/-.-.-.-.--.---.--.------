
"""
CORRECTED: Crypto Analytics System - Complete Integration
CoinGecko for Market Cap Filtering + BingX for OHLCV Data
Advanced Dual-Tier TrendPulse Scanner with Heikin Ashi Analysis
FIXED: Removed incorrect XMR skip + Enhanced market loading diagnostics
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

# CORRECTED: BingX-specific skip list (XMR REMOVED - it's available on BingX!)
BINGX_SKIP_SYMBOLS = {
    'WHYPE',  # Not listed on BingX
    # XMR REMOVED - User confirmed it's available in BingX "all crypto section"
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

def should_skip_symbol(symbol):
    """Check if symbol should be skipped for BingX"""
    return symbol.upper() in BINGX_SKIP_SYMBOLS

def create_enhanced_bingx_exchange():
    """
    ENHANCED: Create BingX exchange with better market loading diagnostics
    """
    try:
        exchange = ccxt.bingx({
            'apiKey': os.environ.get('BINGX_API_KEY', ''),
            'secret': os.environ.get('BINGX_SECRET_KEY', ''),
            'enableRateLimit': True,
            'rateLimit': 1000,      # Slightly faster rate limit
            'timeout': 30000,       # 30 second timeout
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            },
            'headers': {
                'Content-Type': 'application/json',
                'X-SOURCE-KEY': 'CCXT'
            },
        })
        
        # Enhanced market loading with diagnostics
        try:
            print("🔄 Loading BingX markets...")
            markets = exchange.load_markets()
            
            # Diagnostic information
            total_markets = len(markets)
            spot_markets = len([m for m in markets.values() if m.get('type') == 'spot' and m.get('active')])
            futures_markets = len([m for m in markets.values() if m.get('type') == 'swap' and m.get('active')])
            
            # Check for specific coins that were problematic
            test_symbols = ['DOT', 'UNI', 'AAVE', 'ENA', 'XMR']
            available_symbols = []
            missing_symbols = []
            
            for symbol in test_symbols:
                spot_symbol = f"{symbol}/USDT"
                futures_symbol = f"{symbol}/USDT:USDT"
                
                has_spot = spot_symbol in markets and markets[spot_symbol].get('active', False)
                has_futures = futures_symbol in markets and markets[futures_symbol].get('active', False)
                
                if has_spot or has_futures:
                    market_types = []
                    if has_spot:
                        market_types.append('spot')
                    if has_futures:
                        market_types.append('futures')
                    available_symbols.append(f"{symbol}({'+'.join(market_types)})")
                else:
                    missing_symbols.append(symbol)
            
            print(f"✅ BingX markets loaded: {total_markets} total ({spot_markets} spot + {futures_markets} futures active)")
            
            if available_symbols:
                print(f"🎯 Test symbols available: {', '.join(available_symbols)}")
            if missing_symbols:
                print(f"⚠️ Test symbols missing: {', '.join(missing_symbols)}")
            
            return exchange, True
            
        except Exception as e:
            print(f"❌ BingX market loading failed: {str(e)[:100]}")
            return None, False
            
    except Exception as e:
        print(f"❌ BingX exchange creation failed: {str(e)[:100]}")
        return None, False

def get_available_markets_enhanced(symbol, exchange):
    """Enhanced market availability checking with detailed diagnostics"""
    if should_skip_symbol(symbol):
        return {'spot': False, 'futures': False, 'spot_symbol': None, 'futures_symbol': None}
    
    spot_symbol = f"{symbol}/USDT"
    futures_symbol = f"{symbol}/USDT:USDT"
    
    try:
        if not exchange or not hasattr(exchange, 'markets') or not exchange.markets:
            print(f"  ⚠️ {symbol}: No markets loaded in exchange instance")
            return {'spot': False, 'futures': False, 'spot_symbol': None, 'futures_symbol': None}
        
        # Check spot market
        spot_exists = spot_symbol in exchange.markets
        spot_active = spot_exists and exchange.markets[spot_symbol].get('active', False)
        
        # Check futures market
        futures_exists = futures_symbol in exchange.markets
        futures_active = futures_exists and exchange.markets[futures_symbol].get('active', False)
        
        # Enhanced diagnostics for problematic symbols
        if symbol in ['DOT', 'UNI', 'AAVE', 'ENA', 'XMR']:
            status = []
            if spot_exists:
                status.append(f"spot({'active' if spot_active else 'inactive'})")
            if futures_exists:
                status.append(f"futures({'active' if futures_active else 'inactive'})")
            
            if status:
                print(f"  🔍 {symbol} markets: {', '.join(status)}")
            else:
                print(f"  ❌ {symbol}: No markets found (spot: {spot_exists}, futures: {futures_exists})")
        
        available = {
            'spot': spot_active,
            'futures': futures_active,
            'spot_symbol': spot_symbol if spot_active else None,
            'futures_symbol': futures_symbol if futures_active else None
        }
        return available
        
    except Exception as e:
        print(f"  ❌ Market check error for {symbol}: {str(e)[:50]}")
        return {'spot': False, 'futures': False, 'spot_symbol': None, 'futures_symbol': None}

def convert_to_heikin_ashi(df):
    """Convert regular OHLC data to Heikin Ashi candles - FIXED VERSION"""
    if len(df) < 2:
        return df
    
    ha_df = df.copy()
    
    # Initialize HA columns first
    ha_df = ha_df.assign(
        HA_Close=0.0,
        HA_Open=0.0, 
        HA_High=0.0,
        HA_Low=0.0
    )
    
    # Calculate first Heikin Ashi candle using .at instead of .iloc
    first_idx = ha_df.index[0]
    ha_df.at[first_idx, 'HA_Close'] = (df.at[first_idx, 'Open'] + df.at[first_idx, 'High'] + 
                                      df.at[first_idx, 'Low'] + df.at[first_idx, 'Close']) / 4.0
    ha_df.at[first_idx, 'HA_Open'] = (df.at[first_idx, 'Open'] + df.at[first_idx, 'Close']) / 2.0
    ha_df.at[first_idx, 'HA_High'] = df.at[first_idx, 'High']
    ha_df.at[first_idx, 'HA_Low'] = df.at[first_idx, 'Low']
    
    # Calculate subsequent candles using .at indexing
    for i in range(1, len(df)):
        curr_idx = ha_df.index[i]
        prev_idx = ha_df.index[i-1]
        
        # HA Close = (O + H + L + C) / 4
        ha_df.at[curr_idx, 'HA_Close'] = (df.at[curr_idx, 'Open'] + df.at[curr_idx, 'High'] + 
                                         df.at[curr_idx, 'Low'] + df.at[curr_idx, 'Close']) / 4.0
        
        # HA Open = (prev HA Open + prev HA Close) / 2
        ha_df.at[curr_idx, 'HA_Open'] = (ha_df.at[prev_idx, 'HA_Open'] + 
                                        ha_df.at[prev_idx, 'HA_Close']) / 2.0
        
        # HA High = max(H, HA Open, HA Close)
        ha_df.at[curr_idx, 'HA_High'] = max(df.at[curr_idx, 'High'], 
                                           ha_df.at[curr_idx, 'HA_Open'], 
                                           ha_df.at[curr_idx, 'HA_Close'])
        
        # HA Low = min(L, HA Open, HA Close)
        ha_df.at[curr_idx, 'HA_Low'] = min(df.at[curr_idx, 'Low'], 
                                          ha_df.at[curr_idx, 'HA_Open'], 
                                          ha_df.at[curr_idx, 'HA_Close'])
    
    return ha_df

class TrendPulseAnalyzer:
    """Advanced TrendPulse Analysis with Heikin Ashi Integration - Your Private Logic"""
    
    def __init__(self):
        self.ch_len = 9      # Your exact parameters
        self.avg_len = 12    # Your exact parameters
        self.smooth_len = 3  # Your exact parameters

    def ema(self, src, length):
        return src.ewm(span=length, adjust=False).mean()

    def sma(self, src, length):
        return src.rolling(window=length).mean()

    def analyze_heikin_ashi(self, ha_df, tier_type, debug_symbol=""):
        """Analyze Heikin Ashi candles with tier-specific thresholds - Your Private Indicator"""
        if len(ha_df) < self.ch_len + self.avg_len + 5:
            return {
                'signals': [], 'wt1': 0, 'wt2': 0,
                'has_signal': False, 'signal_type': 'none'
            }
        
        try:
            # Use HLC3 from Heikin Ashi candles
            ha_hlc3 = (ha_df['HA_High'] + ha_df['HA_Low'] + ha_df['HA_Close']) / 3.0
            
            # TrendPulse calculation - Your exact logic
            esa = self.ema(ha_hlc3, self.ch_len)
            dev = self.ema(abs(ha_hlc3 - esa), self.ch_len)
            
            # Avoid division by zero
            dev_safe = dev.replace(0, 0.001)
            ci = (ha_hlc3 - esa) / (0.015 * dev_safe)
            
            wt1 = self.ema(ci, self.avg_len)
            wt2 = self.sma(wt1, self.smooth_len)

            # Use .values to get numpy arrays (safe access)
            wt1_values = wt1.values
            wt2_values = wt2.values

            current_wt1 = float(wt1_values[-1])
            current_wt2 = float(wt2_values[-1])
            
            # Check last 3 candles for signals
            signals = []
            for i in range(1, min(4, len(wt1_values))):
                if len(wt1_values) <= i:
                    continue
                    
                wt1_curr = float(wt1_values[-i])
                wt2_curr = float(wt2_values[-i])
                wt1_prev = float(wt1_values[-i-1]) if len(wt1_values) > i else wt1_curr
                wt2_prev = float(wt2_values[-i-1]) if len(wt2_values) > i else wt2_curr
                
                # Tier-specific thresholds - Your exact settings
                if tier_type == "HIGH_RISK":
                    # 1H Heikin Ashi - More sensitive for volatile coins
                    oversold = (wt1_curr <= -50) and (wt2_curr <= -50)
                    overbought = (wt2_curr >= 50) and (wt1_curr >= 50)
                else:  # STANDARD
                    # 30M Heikin Ashi - Standard thresholds for stable coins
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
            
            has_signal = len(signals) > 0
            signal_type = signals[0]['type'] if signals else 'none'
            
            if debug_symbol and signals:
                print(f"  📊 {debug_symbol}: {len(signals)} {tier_type} HA signals")

            return {
                'signals': signals,
                'wt1': current_wt1,
                'wt2': current_wt2,
                'has_signal': has_signal,
                'signal_type': signal_type
            }
            
        except Exception as e:
            print(f"  ❌ TrendPulse error for {debug_symbol}: {str(e)[:50]}")
            return {
                'signals': [], 'wt1': 0, 'wt2': 0,
                'has_signal': False, 'signal_type': 'none'
            }

class CoinGeckoDataManager:
    """CoinGecko integration for market cap filtering and coin discovery"""
    
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
        """Get coins from CoinGecko for both HIGH RISK and STANDARD tiers"""
        cached_coins, cache_time = self.load_cache()
        now = datetime.utcnow()
        
        # Use cache if less than 30 minutes old
        if (now - cache_time).total_seconds() < CACHE_DURATION_MINUTES * 60:
            cache_age = (now - cache_time).total_seconds() / 60
            print(f"🔄 Using cached CoinGecko data (age: {cache_age:.1f} min)")
            return self.categorize_coins(cached_coins), 0
        
        # Fetch fresh data from CoinGecko
        print("🌐 Fetching fresh coin data from CoinGecko...")
        api_key = os.environ.get('COINGECKO_API_KEY', '')
        url = "https://api.coingecko.com/api/v3/coins/markets"
        headers = {'x-cg-demo-api-key': api_key} if api_key else {}
        
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 250,
            'page': 1,
        }
        
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
            
            filtered = []
            stablecoins = {'USDT', 'USDC', 'DAI', 'BUSD', 'USDE', 'FDUSD'}
            
            for coin in data:
                market_cap = coin.get('market_cap', 0)
                volume_24h = coin.get('total_volume', 0)
                current_price = coin.get('current_price', 0)
                
                # Dual-tier filtering - Your exact criteria
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
                        'current_price': current_price,
                        'price_change_24h': coin.get('price_change_percentage_24h', 0)
                    })
            
            self.save_cache(filtered)
            self.api_calls_used += 1
            
            categorized = self.categorize_coins(filtered)
            print(f"✅ CoinGecko: HIGH RISK: {len(categorized['high_risk'])} coins")
            print(f"✅ CoinGecko: STANDARD: {len(categorized['standard'])} coins")
            
            return categorized, 1
            
        except Exception as e:
            print(f"❌ CoinGecko error: {e}")
            return self.categorize_coins(cached_coins), 0

    def categorize_coins(self, all_coins):
        """Separate coins into HIGH RISK and STANDARD tiers"""
        high_risk = []
        standard = []
        
        for coin in all_coins:
            if coin['market_cap'] < 500_000_000:
                high_risk.append(coin)
            else:
                standard.append(coin)
        
        return {
            'high_risk': high_risk,
            'standard': standard
        }

def get_bingx_ohlcv_data_enhanced(symbol, bingx_exchange):
    """
    ENHANCED: Get OHLCV data from BingX with improved error handling and diagnostics
    """
    data = {}
    
    if should_skip_symbol(symbol):
        print(f"  🚫 Skipping {symbol}: In skip list")
        return data
    
    if not bingx_exchange:
        print(f"  ❌ No exchange instance for {symbol}")
        return data
    
    try:
        # Get available markets for this symbol
        markets = get_available_markets_enhanced(symbol, bingx_exchange)
        
        # Enhanced timeframe processing with better error handling
        for timeframe, tf_label in [('30m', '30M'), ('1h', '1H')]:
            limit = 100 if timeframe == '30m' else 50
            min_candles = 30
            success = False
            
            # Try spot market first with enhanced error handling
            if markets['spot'] and markets['spot_symbol'] and not success:
                try:
                    # Reset exchange state
                    bingx_exchange.options['defaultType'] = 'spot'
                    time.sleep(0.05)  # Small delay to prevent rate limiting
                    
                    ohlcv = bingx_exchange.fetch_ohlcv(
                        markets['spot_symbol'], 
                        timeframe, 
                        limit=limit
                    )
                    
                    if ohlcv and len(ohlcv) >= min_candles:
                        df = pd.DataFrame(ohlcv, 
                                        columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        df = df.dropna()
                        
                        if len(df) >= min_candles:
                            data[tf_label] = df
                            success = True
                            if symbol in ['DOT', 'UNI', 'AAVE', 'ENA', 'XMR']:
                                print(f"  ✅ {symbol} {tf_label}: Got spot data ({len(df)} candles)")
                            
                except ccxt.RequestTimeout:
                    print(f"  ⏱️ BingX {tf_label} spot timeout for {symbol}")
                except ccxt.NetworkError as e:
                    print(f"  🌐 BingX {tf_label} spot network error for {symbol}: {str(e)[:30]}")
                except ccxt.ExchangeError as e:
                    error_msg = str(e)
                    if "100400" in error_msg or "100204" in error_msg:
                        print(f"  ⚠️ BingX {tf_label} spot API error for {symbol}: {error_msg[:40]}")
                    else:
                        print(f"  ⚠️ BingX {tf_label} spot error for {symbol}: {error_msg[:40]}")
                except Exception as e:
                    print(f"  ❌ BingX {tf_label} spot unexpected error for {symbol}: {str(e)[:40]}")
            
            # Fallback to futures market if spot failed
            if markets['futures'] and markets['futures_symbol'] and not success:
                try:
                    # Reset exchange state for futures
                    bingx_exchange.options['defaultType'] = 'swap'
                    time.sleep(0.05)  # Small delay
                    
                    ohlcv = bingx_exchange.fetch_ohlcv(
                        markets['futures_symbol'], 
                        timeframe, 
                        limit=limit
                    )
                    
                    if ohlcv and len(ohlcv) >= min_candles:
                        df = pd.DataFrame(ohlcv,
                                        columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        df = df.dropna()
                        
                        if len(df) >= min_candles:
                            data[tf_label] = df
                            success = True
                            print(f"  💎 {symbol} {tf_label}: Using futures data ({len(df)} candles)")
                            
                except ccxt.RequestTimeout:
                    print(f"  ⏱️ BingX {tf_label} futures timeout for {symbol}")
                except ccxt.NetworkError as e:
                    print(f"  🌐 BingX {tf_label} futures network error for {symbol}: {str(e)[:30]}")
                except ccxt.ExchangeError as e:
                    error_msg = str(e)
                    if "100400" in error_msg or "100204" in error_msg:
                        print(f"  ⚠️ BingX {tf_label} futures API error for {symbol}: {error_msg[:40]}")
                    else:
                        print(f"  ⚠️ BingX {tf_label} futures error for {symbol}: {error_msg[:40]}")
                except Exception as e:
                    print(f"  ❌ BingX {tf_label} futures unexpected error for {symbol}: {str(e)[:40]}")
            
            # Final status for this timeframe
            if not success:
                availability = "no active markets found" if not markets['spot'] and not markets['futures'] else "data fetch failed"
                print(f"  ❌ BingX {tf_label} error for {symbol}: {availability}")
        
        return data
        
    except Exception as e:
        print(f"  ❌ BingX critical error for {symbol}: {str(e)[:50]}")
        return {}

def get_ist_time_12h():
    utc = datetime.utcnow()
    ist = utc + timedelta(hours=5, minutes=30)
    return ist.strftime('%I:%M %p %d-%m-%Y'), ist.strftime('%A, %d %B %Y')

def get_reliable_tradingview_url(symbol):
    """Get reliable TradingView chart URL with BingX and other exchange fallbacks"""
    base_symbol = symbol.upper()
    
    # Priority order: BingX -> Binance -> Coinbase -> Kraken
    exchanges = [
        ('BINGX', f"{base_symbol}USDT"),
        ('BINANCE', f"{base_symbol}USDT"),
        ('COINBASE', f"{base_symbol}USD"),
        ('KRAKEN', f"{base_symbol}USD"),
        ('BYBIT', f"{base_symbol}USDT")
    ]
    
    for exchange, pair in exchanges:
        url = f"https://www.tradingview.com/chart/?symbol={exchange}%3A{pair}"
        try:
            resp = requests.head(url, timeout=3, allow_redirects=True)
            if resp.status_code == 200:
                return url, exchange
        except:
            continue
    
    # Final fallback
    return f"https://www.tradingview.com/chart/?symbol={base_symbol}USDT", "Generic"

def send_crypto_analytics_alert(coin, analysis, tier_type, cache):
    """Send enhanced alerts with current price - Updated formatting"""
    
    # Determine chat ID based on tier
    if tier_type == "HIGH_RISK":
        chat_id = os.environ.get('HIGH_RISK_CHAT_ID')
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
    else:  # STANDARD
        chat_id = os.environ.get('TELEGRAM_CHAT_ID') 
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
    
    if not token or not chat_id:
        print(f"❌ Missing Telegram credentials for {tier_type}")
        return

    if not analysis['signals']:
        return
        
    signal = analysis['signals'][0]
    action = signal['type']
    
    time_str, day_str = get_ist_time_12h()
    
    # Cache key
    key = f"{tier_type}_{coin['symbol']}_{action}_{signal['candles_ago']}_{time_str}"
    if key in cache:
        return
    
    # Get reliable chart with fallback
    tv_url, exchange = get_reliable_tradingview_url(coin['symbol'])
    
    # Format current price
    price = coin['current_price']
    if price >= 1:
        price_str = f"${price:,.2f}"
    elif price >= 0.01:
        price_str = f"${price:.4f}"
    else:
        price_str = f"${price:.8f}"
    
    # Price change formatting
    price_change = coin.get('price_change_24h', 0)
    change_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
    change_str = f"{change_emoji} {price_change:+.2f}%"
    
    # Market cap category (for display only)
    cap_billions = coin['market_cap'] / 1_000_000_000
    if cap_billions >= 1:
        cap_category = f"🔷 Large Cap (${cap_billions:.1f}B)"
    else:
        cap_millions = coin['market_cap'] / 1_000_000
        cap_category = f"💎 {'Mid' if cap_millions >= 100 else 'Small'} Cap (${cap_millions:.0f}M)"
    
    # Add colored circles consistently for both tiers
    if action == 'buy':
        action_emoji = '🟢'
    else:  # sell
        action_emoji = '🔴'
    
    # Tier-specific formatting
    if tier_type == "HIGH_RISK":
        title = f"{action_emoji} HIGH RISK Alert {action_emoji}"
        timeframe_info = "📊 1H Heikin Ashi Analysis"
        urgency = "⚡ HIGH REWARD POTENTIAL ⚡"
    else:
        title = f"{action_emoji} Standard Alert {action_emoji}"
        timeframe_info = "📈 30M Heikin Ashi Analysis"
        urgency = "📊 QUALITY SIGNAL"
    
    # Clean message format (no volume/market cap as requested)
    message = f"{title}\n"
    message += f"**{coin['symbol']}-USD — {action.upper()}**\n"
    message += f"{cap_category}\n"
    message += f"{urgency}\n\n"
    message += f"💰 **Price**: {price_str}\n"
    message += f"📊 **24h Change**: {change_str}\n"
    message += f"📈 **WT1**: {signal['wt1']:.2f} | **WT2**: {signal['wt2']:.2f}\n"
    message += f"💪 **Strength**: {signal['strength']:.1f}\n"
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
            print(f"✅ {tier_type} alert sent: {coin['symbol']} {action.upper()} @ {price_str} via {exchange}")
        else:
            print(f"❌ Telegram error for {tier_type}: {res.status_code}")
    except Exception as e:
        print(f"❌ Telegram error for {tier_type}: {e}")

def analyze_coin_with_bingx_enhanced(coin, tier_type, analyzer, bingx_exchange, blocked_coins):
    """
    ENHANCED: Analyze single coin using BingX data with comprehensive error handling
    """
    try:
        if coin['symbol'].upper() in blocked_coins:
            return None, f"🚫 BLOCKED: {coin['symbol']}"
        
        if should_skip_symbol(coin['symbol']):
            return None, f"🚫 SKIPPED: {coin['symbol']}"
        
        # Get OHLCV data from BingX (now with enhanced diagnostics)
        data = get_bingx_ohlcv_data_enhanced(coin['symbol'], bingx_exchange)
        
        timeframe = '1H' if tier_type == 'HIGH_RISK' else '30M'
        if timeframe not in data:
            return None, f"❌ No BingX {timeframe} data: {coin['symbol']}"
        
        df = data[timeframe]
        if len(df) < 30:
            return None, f"❌ Insufficient BingX data: {coin['symbol']}"
        
        # Convert to Heikin Ashi and analyze with your TrendPulse
        ha_df = convert_to_heikin_ashi(df)
        analysis = analyzer.analyze_heikin_ashi(ha_df, tier_type, coin['symbol'])
        
        if analysis['has_signal']:
            return {
                'coin': coin,
                'analysis': analysis,
                'tier': tier_type
            }, f"✅ {analysis['signal_type'].upper()}: {coin['symbol']} (BingX)"
        else:
            return None, f"📊 No signal: {coin['symbol']} (BingX)"
            
    except Exception as e:
        return None, f"❌ BingX error {coin['symbol']}: {str(e)[:50]}"

def main():
    """Main Crypto Analytics System execution - CORRECTED with XMR Support"""
    print("🚀 CRYPTO ANALYTICS SYSTEM - COINGECKO + BINGX INTEGRATION (CORRECTED)")
    print("=" * 85)
    print("🔥 Advanced Dual-Tier TrendPulse Scanner")
    print("📊 CoinGecko: Market cap filtering & coin discovery")
    print("🏢 BingX: Enhanced dual-market OHLCV data (Spot + USD-M Perpetual)")
    print("✅ CORRECTED: XMR removed from skip list (available in BingX all crypto section)")
    print("🔧 Enhanced: Market loading diagnostics & error classification")
    print("📈 HIGH RISK: 1H Heikin Ashi • STANDARD: 30M Heikin Ashi")
    print("💰 Current Price Tracking • Smart Deduplication")
    print("=" * 85)
    
    start_time = datetime.utcnow()
    
    # Load configurations
    alert_cache = load_alert_cache()
    blocked_coins = load_blocked_coins()
    coingecko_manager = CoinGeckoDataManager()
    analyzer = TrendPulseAnalyzer()
    
    # Initialize BingX exchange with enhanced diagnostics
    print("🔧 Initializing enhanced BingX connection with market diagnostics...")
    bingx_exchange, connection_success = create_enhanced_bingx_exchange()
    
    if not connection_success or not bingx_exchange:
        print("❌ BingX connection failed - cannot proceed")
        return
    
    # Get dual-tier coin data from CoinGecko
    tier_data, coingecko_calls = coingecko_manager.get_dual_tier_coins()
    high_risk_coins = tier_data['high_risk']
    standard_coins = tier_data['standard']
    
    if not high_risk_coins and not standard_coins:
        print("❌ No coins retrieved from CoinGecko")
        return
    
    total_coins = len(high_risk_coins) + len(standard_coins)
    skip_count = len([c for c in high_risk_coins + standard_coins if should_skip_symbol(c['symbol'])])
    print(f"📊 Analyzing {total_coins} coins with enhanced BingX dual-market data...")
    if skip_count > 0:
        print(f"🚫 Skipping only {skip_count} truly problematic symbols (WHYPE, etc.)")
    print("📋 XMR will now be processed (removed from skip list)")
    print("=" * 85)
    
    # Parallel processing for both tiers
    all_results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit HIGH RISK coins
        high_risk_futures = [
            executor.submit(analyze_coin_with_bingx_enhanced, coin, 'HIGH_RISK', analyzer, bingx_exchange, blocked_coins)
            for coin in high_risk_coins
        ]
        
        # Submit STANDARD coins  
        standard_futures = [
            executor.submit(analyze_coin_with_bingx_enhanced, coin, 'STANDARD', analyzer, bingx_exchange, blocked_coins)
            for coin in standard_coins
        ]
        
        # Collect HIGH RISK results
        print("🔥 HIGH RISK TIER ANALYSIS (1H Heikin Ashi via Enhanced BingX):")
        print("-" * 65)
        for i, future in enumerate(high_risk_futures, 1):
            result, log = future.result()
            print(f"[{i}/{len(high_risk_futures)}] {log}")
            if result:
                all_results.append(result)
        
        print("\n📊 STANDARD TIER ANALYSIS (30M Heikin Ashi via Enhanced BingX):")
        print("-" * 65)
        # Collect STANDARD results
        for i, future in enumerate(standard_futures, 1):
            result, log = future.result()
            print(f"[{i}/{len(standard_futures)}] {log}")
            if result:
                all_results.append(result)
    
    # Send alerts for all signals found
    print(f"\n🚨 PROCESSING {len(all_results)} SIGNALS:")
    print("-" * 50)
    
    for result in all_results:
        send_crypto_analytics_alert(
            result['coin'], 
            result['analysis'], 
            result['tier'], 
            alert_cache
        )
    
    # Save cache and show summary
    save_alert_cache(alert_cache)
    
    execution_time = (datetime.utcnow() - start_time).total_seconds()
    monthly_calls_coingecko = coingecko_calls * 30 * 24 * (60/8)  # Every 8 minutes
    monthly_calls_bingx = len(all_results) * 2 * 30 * 24 * (60/8)  # Estimated BingX calls
    
    print(f"\n🎉 CRYPTO ANALYTICS SCAN COMPLETE (CORRECTED VERSION):")
    print("=" * 70)
    print(f"   ⏱️  Execution Time: {execution_time:.1f}s")
    print(f"   📊 HIGH RISK (1H Enhanced BingX): {len(high_risk_coins)} coins")
    print(f"   📊 STANDARD (30M Enhanced BingX): {len(standard_coins)} coins")
    print(f"   🚨 Signals Found: {len(all_results)}")
    print(f"   📡 CoinGecko API Calls: {coingecko_calls}")
    print(f"   🏢 BingX Requests: ~{len(high_risk_coins + standard_coins) * 2}")
    print(f"   💰 Monthly CoinGecko Est.: {monthly_calls_coingecko:.0f} calls")
    print(f"   🏢 Monthly BingX Est.: {monthly_calls_bingx:.0f} calls")
    print(f"   ✅ CORRECTED: XMR now included (was incorrectly skipped)")
    print(f"   🔧 Enhanced: Market diagnostics show DOT/UNI/AAVE/ENA/XMR availability")
    print(f"   🔄 Enhanced: Spot/Futures fallback + detailed error classification")
    print(f"   ✅ System Status: {'CORRECTED DUAL-MARKET COVERAGE' if all_results or total_coins > 200 else 'PARTIAL COVERAGE'}")

if __name__ == "__main__":
    main()
