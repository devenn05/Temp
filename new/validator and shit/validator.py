# unified_validator.py (v5.0 Final - The Adaptive Trading Partner)

import pandas as pd
import sys
import api_client
import indicators

# --- CONFIGURATION ---
TRADING_PROFILES = {
    "1": {
        "name": "Scalper",
        "timeframes": {'ltf': '1m', 'entry': '5m', 'htf': '15m'} # ltf = lower timeframe for damage control
    },
    "2": {
        "name": "Day Trader",
        "timeframes": {'ltf': '5m', 'entry': '15m', 'htf': '1h'}
    },
    "3": {
        "name": "Swing Trader",
        "timeframes": {'ltf': '1h', 'entry': '4h', 'htf': '1d'}
    }
}

# --- COLOR CODES ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- DATA & INPUT FUNCTIONS ---
def get_user_scenario():
    print(f"{bcolors.HEADER}{bcolors.BOLD}--- Unified Validator v5.0 ---{bcolors.ENDC}")
    
    # 1. Get Trading Profile
    print("First, select your Trading Profile:")
    for key, profile in TRADING_PROFILES.items():
        print(f"{key}. {profile['name']} (Analyzes {profile['timeframes']['entry']} chart)")
    while True:
        choice = input(f"{bcolors.OKCYAN}Enter your profile choice (1, 2, or 3): {bcolors.ENDC}")
        if choice in TRADING_PROFILES:
            profile = TRADING_PROFILES[choice]
            break
        print(f"{bcolors.FAIL}Invalid choice.{bcolors.ENDC}")

    # 2. Get Scenario
    print("\nNext, select your scenario:")
    print("1. Pre-Trade Validation (Should I enter?)")
    print("2. In-Flight Trade Management (I am already in a trade)")
    while True:
        choice = input(f"{bcolors.OKCYAN}Enter your scenario choice (1 or 2): {bcolors.ENDC}")
        if choice in ['1', '2']:
            scenario = int(choice)
            break

    # 3. Get Trade Details
    plan = {'profile': profile}
    plan['symbol'] = input(f"\nEnter Trading Symbol (e.g., BTCUSDT): {bcolors.ENDC}").upper()
    while True:
        trade_type = input(f"What is your trade's direction ('long' or 'short')?: {bcolors.ENDC}").lower()
        if trade_type in ['long', 'short']: break
    plan['type'] = trade_type
    
    def get_price(prompt):
        while True:
            try: return float(input(prompt))
            except ValueError: print(f"{bcolors.FAIL}Invalid number.{bcolors.ENDC}")
    
    plan['entry'] = get_price(f"Your Entry Price: {bcolors.ENDC}")
    if scenario == 1:
        plan['tp'] = get_price(f"Your Planned Take-Profit: {bcolors.ENDC}")
        plan['sl'] = get_price(f"Your Planned Stop-Loss: {bcolors.ENDC}")

    return scenario, plan

def fetch_all_data(symbol, timeframes):
    print(f"\n{bcolors.OKBLUE}Fetching market data for {symbol}...{bcolors.ENDC}")
    data = {}
    for key, tf in timeframes.items():
        df = api_client.fetch_ohlc_data(symbol, tf, 'futures', limit=300)
        if df is None: return None
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
        data[tf] = df
    return data

# --- ANALYSIS ENGINE ---
def run_adaptive_analysis(data, plan):
    profile_tfs = plan['profile']['timeframes']
    trade_type = plan['type']
    
    # Define which dataframes to use based on the profile
    df_entry = data[profile_tfs['entry']]
    df_htf = data[profile_tfs['htf']]

    analysis = {'thesis': {'checklist': {}, 'is_counter_trend': False}}

    # 1. HTF Context
    _, direction_htf = indicators.get_market_regime_advanced(df_htf)
    is_aligned = (trade_type == 'long' and direction_htf == 'Bullish') or \
                 (trade_type == 'short' and direction_htf == 'Bearish')
    analysis['thesis']['checklist']['htf_context'] = f"Analyzing {profile_tfs['entry']} for a {trade_type} within a {direction_htf} {profile_tfs['htf']} trend."
    if not is_aligned: analysis['thesis']['is_counter_trend'] = True
        
    # 2. Liquidity Grab (on entry timeframe)
    entry_swings = indicators.find_swing_points(df_entry, order=8)
    grab_price, grab_time = indicators.get_liquidity_grab(df_entry, entry_swings, trade_type)
    if not grab_price: return analysis

    analysis['thesis']['checklist']['liquidity_grab'] = f"Swept {profile_tfs['entry']} level @ {indicators.format_price(grab_price)}"
    analysis['thesis']['invalidation_price'] = grab_price

    # 3. Break of Structure (on entry timeframe)
    bos_price = indicators.analyze_break_of_structure(df_entry, grab_time, trade_type)
    if not bos_price: return analysis
    analysis['thesis']['checklist']['structure_break'] = f"Confirmed @ {indicators.format_price(bos_price)}"
    
    # 4. Point of Interest (on entry timeframe)
    fvg = indicators.find_latest_fvg(df_entry.tail(50))
    poi_key = 'bullish' if trade_type == 'long' else 'bearish'
    if fvg[poi_key]:
        analysis['thesis']['poi'] = {'type': f"{profile_tfs['entry']} FVG", 'zone': fvg[poi_key]}
        analysis['thesis']['checklist']['poi_identified'] = f"Found a {analysis['thesis']['poi']['type']}"

    return analysis

# --- REPORTING ---
def print_report(scenario, plan, analysis):
    profile_name = plan['profile']['name']
    trade_type = plan['type']
    is_thesis_confirmed = len(analysis['thesis']['checklist']) >= 4

    print("\n\n" + "="*70)
    print(f"|{bcolors.BOLD}{bcolors.HEADER}              The Adaptive Trade Validator v5.0                 {bcolors.ENDC}|")
    print("="*70 + "\n")
    print(f"{bcolors.OKBLUE}[INFO] Analysis for a {bcolors.BOLD}{trade_type.upper()}{bcolors.ENDC} on {bcolors.BOLD}{plan['symbol']}{bcolors.ENDC}. Profile: {bcolors.BOLD}{profile_name}{bcolors.ENDC}\n")

    # --- Print Correct Report based on Scenario ---
    if scenario == 1: # Pre-Trade
        print(f"{bcolors.BOLD}{bcolors.UNDERLINE}Scenario: Pre-Trade Validation{bcolors.ENDC}")
        if is_thesis_confirmed:
            print(f"{bcolors.OKGREEN}Verdict: High-Confidence Setup. The institutional thesis is fully formed.{bcolors.ENDC}")
            if analysis['thesis']['is_counter_trend']:
                print(f"{bcolors.WARNING}Note: This is a high-risk counter-trend setup. The HTF trend is against you.{bcolors.ENDC}")
            
            # Actionable Plan
            poi_high, poi_low = max(analysis['thesis']['poi']['zone']), min(analysis['thesis']['poi']['zone'])
            sl = analysis['thesis']['invalidation_price'] * (1.001 if trade_type == 'short' else 0.999)
            htf_swings = indicators.find_swing_points(analysis['data'][plan['profile']['timeframes']['htf']], order=8)
            tp_key = 'lows' if trade_type == 'short' else 'highs'
            primary_target = htf_swings[tp_key].iloc[-1][tp_key.rstrip('s')]

            print(f"\n{bcolors.HEADER}Optimal Action Plan:{bcolors.ENDC}")
            print(f"  - Entry Zone (POI): {bcolors.OKCYAN}{indicators.format_price(poi_low)} - {indicators.format_price(poi_high)}{bcolors.ENDC}")
            print(f"  - Stop-Loss:      {bcolors.FAIL}{indicators.format_price(sl)}{bcolors.ENDC}")
            print(f"  - Primary Target: {bcolors.OKGREEN}{indicators.format_price(primary_target)}{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}Verdict: Low-Confidence Setup. The institutional thesis is incomplete.{bcolors.ENDC}")
            # Show what's missing
            print("  - " + analysis['thesis']['checklist'].get('htf_context', "HTF Context check..."))
            if 'htf_context' in analysis['thesis']['checklist']: print("  - " + analysis['thesis']['checklist'].get('liquidity_grab', "Waiting for a liquidity grab..."))
            if 'liquidity_grab' in analysis['thesis']['checklist']: print("  - " + analysis['thesis']['checklist'].get('structure_break', "Waiting for a break of structure..."))
            if 'structure_break' in analysis['thesis']['checklist']: print("  - " + analysis['thesis']['checklist'].get('poi_identified', "Waiting for a clear POI to form..."))
    
    else: # In-Flight or Damage Control
        # Assess if thesis is still valid from CURRENT price perspective
        htf_ok = not analysis['thesis']['is_counter_trend']
        # The true invalidation for an existing trade is a break of structure against us
        bos_against, bos_against_time = indicators.get_liquidity_grab(analysis['data'][plan['profile']['timeframes']['entry']], indicators.find_swing_points(analysis['data'][plan['profile']['timeframes']['entry']]), 'long' if trade_type=='short' else 'short')
        is_still_valid = htf_ok and (bos_against is None)

        if is_still_valid:
            print_in_flight_report(plan, analysis)
        else:
            print_damage_control_report(plan, analysis)


def print_in_flight_report(plan, analysis):
    """Prints the management plan for a still-valid trade."""
    print(f"{bcolors.BOLD}{bcolors.UNDERLINE}Scenario: In-Flight Trade Management{bcolors.ENDC}")
    print(f"\n{bcolors.HEADER}Situation Assessment:{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}[✅] Thesis Remains Valid. The market structure continues to support your position.{bcolors.ENDC}")
    
    print(f"\n{bcolors.HEADER}Dynamic Management Plan:{bcolors.ENDC}")
    entry_tf = plan['profile']['timeframes']['entry']
    htf_tf = plan['profile']['timeframes']['htf']
    swings_entry = indicators.find_swing_points(analysis['data'][entry_tf])
    swings_htf = indicators.find_swing_points(analysis['data'][htf_tf])

    # Invalidation Level
    sl_key, sl_price_key = ('lows', 'low') if plan['type'] == 'long' else ('highs', 'high')
    invalidation_level = swings_entry[sl_key].iloc[-1][sl_price_key]
    print(f"[🛑] 1. Protective Stop-Loss: {bcolors.FAIL}{indicators.format_price(invalidation_level)}{bcolors.ENDC}")
    print("    -- This is the current hard invalidation level based on entry timeframe structure.")

def print_damage_control_report(plan, analysis):
    """Prints the optimal exit strategy for a failed trade."""
    profile = plan['profile']
    trade_type = plan['type']
    
    print(f"\n{bcolors.FAIL}{bcolors.BOLD}{'='*60}\n|             !! TRADE INVALIDATED - DAMAGE CONTROL !!         |\n{'='*60}{bcolors.ENDC}")
    print(f"\n{bcolors.HEADER}Situation Assessment:{bcolors.ENDC}")
    print(f"{bcolors.FAIL}[❌] Thesis Invalidated. The market structure no longer supports your {trade_type} position.{bcolors.ENDC}")
    if analysis['thesis']['is_counter_trend']: print(f"{bcolors.WARNING}Reason: The trade was counter-trend, and HTF momentum has taken over.{bcolors.ENDC}")
    else: print(f"{bcolors.WARNING}Reason: A break of structure against your position was detected.{bcolors.ENDC}")

    # Calculate optimal exits using the designated lower timeframe
    df_ltf = analysis['data'][profile['timeframes']['ltf']]
    fvg = indicators.find_latest_fvg(df_ltf.tail(30))
    optimal_exit_fvg = fvg['bullish' if trade_type == 'long' else 'bearish']
    
    print(f"\n{bcolors.HEADER}Recommended Exit Strategy:{bcolors.ENDC}")
    if optimal_exit_fvg:
        fvg_high, fvg_low = max(optimal_exit_fvg), min(optimal_exit_fvg)
        print(f"[🎯] 1. Optimal Exit Target: {bcolors.OKCYAN}{indicators.format_price(fvg_low)} - {indicators.format_price(fvg_high)}{bcolors.ENDC}")
        print(f"    -- High probability of a brief relief bounce into this {profile['timeframes']['ltf']} FVG. Aim to exit here.")
    else:
        print(f"[🎯] 1. Optimal Exit Target: {bcolors.WARNING}No clean FVG found for a bounce. Consider exiting on any sign of strength.{bcolors.ENDC}")

    # Final Hard Stop
    breakdown_candle = analysis['data'][profile['timeframes']['entry']].iloc[-2]
    hard_stop = breakdown_candle['low'] if trade_type == 'long' else breakdown_candle['high']
    print(f"\n[🛑] 2. Final Hard Stop: {bcolors.FAIL}{indicators.format_price(hard_stop)}{bcolors.ENDC}")
    print("    -- If price moves beyond this level, the chance of a bounce is minimal. You MUST exit.")


if __name__ == '__main__':
    scenario, user_plan = get_user_scenario()
    profile = user_plan['profile']
    
    all_data = fetch_all_data(user_plan['symbol'], profile['timeframes'])
    if not all_data:
        sys.exit(f"\n{bcolors.FAIL}Could not fetch all required market data. Exiting.{bcolors.ENDC}")
        
    full_analysis = run_adaptive_analysis(all_data, user_plan)
    full_analysis['data'] = all_data # Attach data for reporting functions
    
    print_report(scenario, user_plan, full_analysis)
    
    print("\n" + "="*70)
    print("Disclaimer: This tool provides a technical analysis model. Not financial advice.")
    print("="*70)