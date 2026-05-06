from __future__ import annotations

from typing import Any

from almanak.framework.dashboard.templates import get_rsi_config, render_ta_dashboard


def _build_dashboard_config(strategy_config: dict[str, Any]):
    period = int(strategy_config.get("rsi_period", 14))
    overbought = float(strategy_config.get("rsi_upper", 55))
    oversold = float(strategy_config.get("rsi_lower", 45))
    regime_mode = str(strategy_config.get("regime_mode", "momentum_flip")).lower()

    config = get_rsi_config(period=period, overbought=overbought, oversold=oversold)
    config.signal_type = "momentum" if regime_mode == "momentum_flip" else "reversion"
    return config


def render_custom_dashboard(
    strategy_id: str,
    strategy_config: dict[str, Any],
    api_client: Any,
    session_state: dict[str, Any],
) -> None:
    config = _build_dashboard_config(strategy_config)
    render_ta_dashboard(strategy_id, strategy_config, session_state, config)
