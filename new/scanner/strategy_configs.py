# strategy_configs.py

# This file holds the "golden parameters" discovered during optimization.
# The scanner will load these configurations for each asset.
# The optimizer will write its best results to this file.

OPTIMIZED_PARAMETERS = {
    # This dictionary will be populated by the optimizer.py script.
    # It will look like this after running:
    #
    # 'BTCUSDT': {
    #     'risk': {
    #         'min_rr': 3.0,
    #         'atr_buffer_multiplier': 1.5,
    #         'sweep_lookback': 20
    #     },
    #     'costs': {
    #         'trading_fee_percent': 0.04,
    #         'slippage_percent': 0.01
    #     }
    # },
    # 'ETHUSDT': {
    #     ...etc...
    # }
}

# --- Default settings to use if an asset hasn't been optimized yet ---
DEFAULT_SETTINGS = {
    "risk": {
        "min_rr": 3.0,
        "risk_per_trade_percent": 1.0, # Standard risk per trade
        "atr_buffer_multiplier": 1.5,
        "partial_tp_rr": 1.5,
        "adx_threshold": 20, # Default value
        "sweep_lookback": 20, # Default value
    },
    "costs": {
        "trading_fee_percent": 0.04,
        "slippage_percent": 0.01
    }
}