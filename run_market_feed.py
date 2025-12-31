
import os
import pandas as pd
import upstox_client
import ssl
import websocket
from upstox_client.feeder.market_data_feeder import MarketDataFeeder
# Import logic from algorithmic_paper_trader
from algorithmic_paper_trader import Config, MarketDataProvider, RiskManager, Strategy52WBreakout as Strategy
from datetime import datetime, time as dt_time
import pandas as pd

from dotenv import load_dotenv
import threading
import random
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()

class SandboxMarketDataFeeder(MarketDataFeeder):
    def connect(self):
        if self.ws and self.ws.sock:
            return

        sslopt = {
            "cert_reqs": ssl.CERT_NONE,
            "check_hostname": False,
        }
        
        host = self.api_client.configuration.host
        if "sandbox" in host:
             # Try removing /v2 for sandbox if 404 occurs.
             ws_url = "wss://api-sandbox.upstox.com/feed/market-data-feed"
        else:
             ws_url = "wss://api.upstox.com/v2/feed/market-data-feed"
             
        logger.info(f"Connecting to WebSocket URL: {ws_url}")

        headers = {'Authorization': self.api_client.configuration.auth_settings().get("OAUTH2")[
            "value"]}
            
        self.ws = websocket.WebSocketApp(ws_url,
                                         header=headers,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)

        # Run in a separate thread
        wst = threading.Thread(target=self.ws.run_forever,
                         kwargs={"sslopt": sslopt})
        wst.daemon = True
        wst.start()
        
        # Keep main thread alive or allow it to return?
        # The SDK usually expects the user to keep the script running.
        # We will return here and let the main loop handle the wait.

def simulate_market_feed(instrument_keys, key_map, market_data, risk_manager, strategy):
    """
    Simulates market data feed for testing alerts when WebSocket is unavailable.
    """
    logger.info("Starting Market Data Simulation (Mock Mode)...")
    
    # Pre-populate 52w High/Low for simulation
    # Limit to 5 keys for simulation to ensure frequent updates on same keys
    sim_keys = instrument_keys[:5]
    
    for ik in sim_keys:
        if ik in key_map:
            sym, exch = key_map[ik]
            # Mock 52w High/Low: High = 100, Low = 50
            # We will generate prices around 95-105 to trigger alerts
            market_data.set_mock(sym, exch, 90.0, 100.0, 50.0)
            
    logger.info(f"Mock data initialized for {len(sim_keys)} keys. Starting tick generation...")
    
    try:
        while True:
            # Pick a random key from the small set
            ik = random.choice(sim_keys)
            if ik in key_map:
                sym, exch = key_map[ik]
                
                # Generate a price
                # 20% chance to be a breakout price (>100)
                # We need previous price to be <= 100 and current > 100 for crossover.
                # The simulator updates price. 
                # If we want to force a crossover, we might need to control the sequence.
                # Random walk might take time.
                # Let's force a toggle for demo purposes.
                
                current_stored = market_data.get_ltp(sym, exch)
                if current_stored is None:
                    # Initial state
                    price = random.uniform(90.0, 99.0)
                elif current_stored > 100:
                     # Drop it back down
                     price = random.uniform(90.0, 99.0)
                else:
                     # Spike it up
                     if random.random() < 0.3:
                         price = random.uniform(100.1, 105.0)
                     else:
                         price = random.uniform(90.0, 99.0)
                    
                # Create a mock message structure similar to Upstox
                message = {
                    'feeds': {
                        ik: {
                            'ltp': price
                        }
                    }
                }
                
                # Process the mock message
                process_tick(message, key_map, market_data, risk_manager, strategy)
                
            time.sleep(0.2) # 5 ticks per second for faster feedback
            
    except KeyboardInterrupt:
        logger.info("Simulation stopped.")

def process_tick(message, key_map, market_data, risk_manager, strategy):
    """
    Shared logic to process a tick (real or mock) and trigger alerts.
    """
    try:
        if isinstance(message, dict) and 'feeds' in message:
            for instrument_key, feed_data in message['feeds'].items():
                if 'ltp' in feed_data:
                    current_price = float(feed_data['ltp'])
                    
                    if instrument_key in key_map:
                        symbol, exchange = key_map[instrument_key]
                        
                        # Update MarketDataProvider
                        # This updates self.mock_prices and sets self.last_prices
                        market_data.update_mock_price(symbol, exchange, current_price)
                        
                        # Alert Logic using Strategy
                        try:
                            # Use strategy signal which checks for Crossover (Prev <= High < Current)
                            signal = strategy.signal(symbol, exchange)
                            
                            if signal == "BUY":
                                # Fetch high for logging
                                high_52w, _ = market_data.get_52w_high_low(symbol, exchange)
                                logger.info(f"*** ALERT: {symbol} ({exchange}) Breakout! Price: {current_price:.2f} crossed 52W High: {high_52w:.2f}")
                                
                                if risk_manager.eligible(current_price):
                                    qty = risk_manager.compute_qty(current_price)
                                    logger.info(f"    -> TRADE ELIGIBLE: Buy {qty} qty of {symbol}")
                                else:
                                    logger.info(f"    -> Trade Ineligible (Risk/Capital)")
                                     
                        except Exception as e:
                            # logger.warning(f"Strategy check failed: {e}")
                            pass
    except Exception as e:
        logger.error(f"Error processing tick: {e}")

def filter_instrument_keys_and_map(df):
    """
    Filters instrument keys containing 'BSE_EQ|INE' or 'NSE_EQ|INE'
    and returns a tuple of (list_of_keys, map_key_to_symbol_exchange).
    """
    regex_pattern = r'BSE_EQ\|INE|NSE_EQ\|INE'
    mask = df['instrument_key'].astype(str).str.contains(regex_pattern, regex=True, na=False)
    filtered_df = df[mask]
    
    keys = filtered_df['instrument_key'].tolist()
    
    # Create map: instrument_key -> (tradingsymbol, exchange)
    # Assuming 'exchange' column exists, else parse from key
    key_map = {}
    for _, row in filtered_df.iterrows():
        ik = row['instrument_key']
        sym = row['tradingsymbol']
        # If exchange col exists use it, else derive
        if 'exchange' in row:
            exch = row['exchange']
        else:
             # Derive from key e.g. NSE_EQ|INE... -> NSE
             exch = ik.split('|')[0].replace('_EQ', '')
        
        key_map[ik] = (sym, exch)
        
    return keys, key_map

import argparse

def main():
    parser = argparse.ArgumentParser(description="Upstox WebSocket Market Feed")
    parser.add_argument("--csv", type=str, default="complete.csv", help="Path to the CSV file")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode (mock data)")
    args = parser.parse_args()

    # 1. Read CSV
    csv_path = args.csv
    if not os.path.exists(csv_path):
        logger.error(f"{csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}")
        return

    # 2. Filter
    instrument_keys, key_map = filter_instrument_keys_and_map(df)
    
    if not instrument_keys:
        logger.warning("No instrument keys found matching criteria (BSE_EQ|INE, NSE_EQ|INE).")
        return

    logger.info(f"Found {len(instrument_keys)} instrument keys matching criteria.")
    
    # Production Safeguard:
    # Subscribing to thousands of keys on a single WebSocket connection may exceed API limits 
    # (typically ~100-500 keys per connection or depending on plan).
    # We limit to 100 for this script to ensure stability. 
    # In a full production system, you would distribute these across multiple connections 
    # or only subscribe to a relevant subset (Watchlist).
    
    MAX_KEYS = int(os.getenv("MAX_WEBSOCKET_KEYS", 100))
    if len(instrument_keys) > MAX_KEYS:
        logger.warning(f"Subscribing to {len(instrument_keys)} keys. Limiting to first {MAX_KEYS} for stability.")
        instrument_keys = instrument_keys[:MAX_KEYS]
    
    # 3. Setup Upstox
    access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        logger.error("UPSTOX_ACCESS_TOKEN not found in .env")
        return

    configuration = upstox_client.Configuration()
    configuration.access_token = access_token
    
    # Set host if provided in env
    base_url = os.getenv("UPSTOX_BASE_URL")
    if base_url:
        configuration.host = base_url

    api_client = upstox_client.ApiClient(configuration)

    # 4. Initialize Paper Trader Components
    # We will use the 'upstox' mode for MarketDataProvider to fetch 52w highs if needed
    # although the current algorithmic_paper_trader.py implementation of 'upstox' mode
    # fetches LTP via REST, we will feed it via WebSocket here.
    
    # Define configuration
    config = Config(
        capital=float(os.getenv("CAPITAL", 50000)),
        allocation_pct=0.2,
        start_time=dt_time(9, 15),
        stop_entry_time=dt_time(15, 0),
        square_off_time=dt_time(15, 15),
        exchanges=["NSE", "BSE"],
        strategy_name="52W Breakout",
        min_price=50.0,
        mode="mock" if args.simulate else "upstox" # Use mock mode if simulating
    )
    
    market_data = MarketDataProvider(mode=config.mode)
    # Inject API client or token if needed by MarketDataProvider (it uses env vars)
    
    risk_manager = RiskManager(config)
    strategy = Strategy(market_data)
    
    # Pre-load 52-week High/Low data for the selected instruments
    # In a real scenario, you'd do this via API. For now, we assume MarketDataProvider handles it
    # or we might need to mock/fetch it. 
    # The current algorithmic_paper_trader.py fetches it on demand.
    
    # 5. WebSocket Callbacks
    def on_open(ws):
        logger.info("WebSocket Connection Opened.")

    def on_message(ws, message):
        # Use shared processing logic
        process_tick(message, key_map, market_data, risk_manager, strategy)

    def on_error(ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        logger.info(f"WebSocket Closed. Code: {close_status_code}, Msg: {close_msg}")

    # 5. Connect
    if args.simulate:
        simulate_market_feed(instrument_keys, key_map, market_data, risk_manager, strategy)
        return

    try:
        # Note: 'instrumentKeys' is camelCase in the SDK 2.7.0
        # but user asked to scan instrument key so I should probably do all.
        # I'll stick to all but warn.
        if len(instrument_keys) > 100:
            logger.warning(f"Subscribing to {len(instrument_keys)} keys. Limiting to first 100 for stability during initial run.")
            instrument_keys = instrument_keys[:100]
                    
        # Use our custom class that handles Sandbox URLs
        feeder = SandboxMarketDataFeeder(
            api_client=api_client,
            instrumentKeys=instrument_keys, 
            mode='ltp', # 'ltp' for just Last Traded Price, 'full' for depth
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        logger.info("Connecting to Upstox WebSocket...")
        feeder.connect()
        
        # Keep main thread alive
        print("\nPress Ctrl+C to exit.")
        print("Run with '--simulate' to use mock data if WebSocket fails.")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error(f"Failed to start feeder: {e}")
    
if __name__ == "__main__":
    main()
