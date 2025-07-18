# project_beta.py (v2.0 - Direction-Agnostic Opportunity Generator)

import pandas as pd
import sys
import api_client
import indicators

# --- CONFIGURATION ---
MIN_THESIS_SCORE = 75  # The minimum score needed to be considered a valid thesis

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
def get_user_input():
    """Gets only the symbol from the user."""
    print(f"{bcolors.HEADER}{bcolors.BOLD}--- Project Beta: The Opportunity Generator ---{bcolors.ENDC}")
    symbol = input(f"{bcolors.OKCYAN}Enter Trading Symbol to analyze (e.g., BTCUSDT): {bcolors.ENDC}").upper()
    return symbol

def fetch_all_data(symbol):
    print(f"\n{bcolors.OKBLUE}Fetching market data for {symbol}...{bcolors.ENDC}")
    data = {}
    timeframes = ['15m', '1h', '4h', '1d']
    for tf in timeframes:
        df = api_client.fetch_ohlc_data(symbol, tf, 'futures', limit=500)
        if df is None: return None
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)
        data[tf] = df
    return data

# --- ANALYSIS ENGINE ---
def calculate_thesis_confidence(data, trade_type):
    """
    Analyzes a single direction (long or short) and returns a confidence score
    and the full thesis data if confirmed.
    """
    score = 0
    thesis = {'checklist': {}}
    
    # Define which dataframes to use for this analysis
    df_1h, df_4h = data['1h'], data['4h']

    # 1. HTF Context (+50 points)
    _, direction_4h = indicators.get_market_regime_advanced(df_4h)
    if (trade_type == 'long' and direction_4h == 'Bullish') or \
       (trade_type == 'short' and direction_4h == 'Bearish'):
        score += 50
        thesis['checklist']['htf_context'] = f"Aligned with {direction_4h} 4H Trend."
    else: return 0, None # Fails immediately if against HTF trend
        
    # 2. Liquidity Grab of HTF swing point (+25 points)
    htf_swings = indicators.find_swing_points(df_4h, order=8)
    grab_price, grab_time = indicators.get_liquidity_grab(df_1h, htf_swings, trade_type)
    if not grab_price: return 0, None
    score += 25
    thesis['checklist']['liquidity_grab'] = f"Swept major 4H level @ {indicators.format_price(grab_price)}"
    thesis['invalidation_price'] = grab_price

    # 3. Break of Structure (+25 points)
    bos_price = indicators.analyze_break_of_structure(df_1h, grab_time, trade_type)
    if not bos_price: return 0, None
    score += 25
    thesis['checklist']['structure_break'] = f"Confirmed with a 1H BoS @ {indicators.format_price(bos_price)}"

    # 4. Find POI (no points, just adds data)
    fvg = indicators.find_latest_fvg(df_1h.tail(100))
    poi_key = 'bullish' if trade_type == 'long' else 'bearish'
    if fvg[poi_key]:
        thesis['poi'] = {'type': '1H FVG', 'zone': fvg[poi_key]}
        thesis['checklist']['poi_identified'] = f"A reactive {thesis['poi']['type']} has formed."
        
    return score, thesis

def generate_plan(thesis, trade_type):
    """Generates one plan based on a confirmed thesis."""
    if not thesis or not 'poi' in thesis: return None
    plan = {}
    poi_high, poi_low = max(thesis['poi']['zone']), min(thesis['poi']['zone'])
    plan['entry_zone'] = (poi_low, poi_high)
    plan['sl'] = thesis['invalidation_price'] * (0.9985 if trade_type == 'long' else 1.0015)
    
    # Targets based on opposing HTF swing points
    opposing_trade_type = 'short' if trade_type == 'long' else 'long'
    # To get a target, we look for the next major high/low to be taken
    htf_swings = indicators.find_swing_points(analysis['data']['4h'], order=8)
    tp_key = 'highs' if trade_type == 'long' else 'lows'
    price_key = 'high' if trade_type == 'long' else 'low'
    
    if not htf_swings[tp_key].empty:
        plan['tp'] = htf_swings[tp_key].iloc[-1][price_key]

    return plan

# --- REPORTING ---
def print_market_briefing(symbol, long_analysis, short_analysis):
    current_price = long_analysis['data']['1h'].iloc[-1]['close'] # Price is same for both
    
    print("\n\n" + "="*70)
    print(f"|{bcolors.BOLD}{bcolors.HEADER}           Market Briefing for {symbol.ljust(10)}                 {bcolors.ENDC}|")
    print("="*70)
    print(f"Current Price: {indicators.format_price(current_price)}")

    long_score, long_thesis = long_analysis['score'], long_analysis['thesis']
    short_score, short_thesis = short_analysis['score'], short_analysis['thesis']

    # Adjudicator Logic
    if long_score < MIN_THESIS_SCORE and short_score < MIN_THESIS_SCORE:
        print(f"\n{bcolors.WARNING}{bcolors.BOLD}VERDICT: MARKET IN CONSOLIDATION / CHOP{bcolors.ENDC}")
        print("Neither a Long nor a Short institutional thesis is confirmed.")
        print("It is advised to wait for a clearer market structure to form before trading.")
        
    elif long_score >= short_score:
        print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}VERDICT: PRIMARY OPPORTUNITY IS LONG (Score: {long_score}/100){bcolors.ENDC}")
        print_thesis_details(long_thesis, 'long')
        plan = generate_plan(long_thesis, 'long')
        if plan: print_plan_details(plan)
            
    else: # short_score > long_score
        print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}VERDICT: PRIMARY OPPORTUNITY IS SHORT (Score: {short_score}/100){bcolors.ENDC}")
        print_thesis_details(short_thesis, 'short')
        plan = generate_plan(short_thesis, 'short')
        if plan: print_plan_details(plan)

    print("\n" + "="*70)
    print("Disclaimer: This tool provides a technical analysis model. Not financial advice.")
    print("="*70)


def print_thesis_details(thesis, trade_type):
    """A helper function to print the checklist details."""
    print(f"\n{bcolors.UNDERLINE}Thesis Checklist ({trade_type.upper()}):{bcolors.ENDC}")
    if not thesis: return
    for key, value in thesis.get('checklist', {}).items():
        print(f"  [✅] {key.replace('_',' ').title()}: {value}")

def print_plan_details(plan):
    """A helper function to print the actionable plan."""
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}Actionable Trade Plan:{bcolors.ENDC}")
    print(f"  - Optimal Entry Zone (POI): {bcolors.OKCYAN}{indicators.format_price(plan['entry_zone'][0])} - {indicators.format_price(plan['entry_zone'][1])}{bcolors.ENDC}")
    print(f"  - Structural Stop-Loss:     {bcolors.FAIL}{indicators.format_price(plan['sl'])}{bcolors.ENDC}")
    if 'tp' in plan:
        # Calculate R:R
        entry_mid = (plan['entry_zone'][0] + plan['entry_zone'][1]) / 2
        risk = abs(entry_mid - plan['sl'])
        reward = abs(plan['tp'] - entry_mid)
        rr = reward/risk if risk > 0 else 0
        print(f"  - Primary HTF Target:     {bcolors.OKGREEN}{indicators.format_price(plan['tp'])} (R:R Approx: {rr:.2f}:1){bcolors.ENDC}")


if __name__ == '__main__':
    symbol = get_user_input()
    all_market_data = fetch_all_data(symbol)

    if not all_market_data:
        sys.exit(f"\n{bcolors.FAIL}Could not fetch market data. Exiting.{bcolors.ENDC}")

    # --- Dual Analysis ---
    long_score, long_thesis = calculate_thesis_confidence(all_market_data, 'long')
    short_score, short_thesis = calculate_thesis_confidence(all_market_data, 'short')
    
    # Bundle for reporting
    long_analysis_package = {'score': long_score, 'thesis': long_thesis, 'data': all_market_data}
    short_analysis_package = {'score': short_score, 'thesis': short_thesis, 'data': all_market_data}
    
    # The Adjudicator passes both results to the final reporter
    analysis = all_market_data # a placeholder since we pass both long and short analysis
    print_market_briefing(symbol, long_analysis_package, short_analysis_package)