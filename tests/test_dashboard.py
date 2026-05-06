from __future__ import annotations

from unittest.mock import patch

from dashboard.ui import _build_dashboard_config, render_custom_dashboard


def test_build_dashboard_config_uses_strategy_settings() -> None:
    config = _build_dashboard_config(
        {
            "rsi_period": 14,
            "rsi_upper": 55,
            "rsi_lower": 45,
            "regime_mode": "momentum_flip",
        }
    )

    assert config.indicator_name == "RSI"
    assert config.indicator_period == 14
    assert config.upper_threshold == 55
    assert config.lower_threshold == 45
    assert config.signal_type == "momentum"


def test_render_custom_dashboard_calls_ta_template() -> None:
    strategy_id = "strategy-1"
    strategy_config = {
        "rsi_period": 14,
        "rsi_upper": 55,
        "rsi_lower": 45,
        "regime_mode": "momentum_flip",
    }
    session_state = {"rsi_value": 52}

    with patch("dashboard.ui.render_ta_dashboard") as render_mock:
        render_custom_dashboard(strategy_id, strategy_config, api_client=None, session_state=session_state)

    render_mock.assert_called_once()
    call_strategy_id, call_config, call_session_state, dashboard_config = render_mock.call_args.args

    assert call_strategy_id == strategy_id
    assert call_config == strategy_config
    assert call_session_state == session_state
    assert dashboard_config.indicator_name == "RSI"
    assert dashboard_config.upper_threshold == 55
    assert dashboard_config.lower_threshold == 45
    assert dashboard_config.signal_type == "momentum"
