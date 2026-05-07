from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from almanak.framework.strategies import RSIData, TokenBalance
from almanak.framework.teardown import TeardownMode
from strategy import LONG_USDC, LONG_WMATIC, NEUTRAL, WMATICTASwapRSIStrategy


@pytest.fixture
def config() -> dict:
    with (Path(__file__).parent.parent / "config.json").open() as f:
        return json.load(f)


@pytest.fixture
def strategy(config: dict) -> WMATICTASwapRSIStrategy:
    return WMATICTASwapRSIStrategy(
        config=config,
        chain="polygon",
        wallet_address="0x" + "1" * 40,
    )


def _ohlcv_from_rsi(rsi_value: Decimal, period: int = 14) -> pd.DataFrame:
    rsi_float = float(rsi_value)
    ratio = rsi_float / (100.0 - rsi_float)
    diffs = [ratio, -1.0] + [0.0] * (period - 2)
    closes = [100.0]
    for diff in diffs:
        closes.append(closes[-1] + diff)

    rows = [{"close": close} for close in closes]
    return pd.DataFrame(rows)


def _market(
    *,
    timestamp: datetime,
    rsi_value: Decimal,
    usdc_balance: Decimal = Decimal("1000"),
    usdc_usd: Decimal = Decimal("1000"),
    wmatic_balance: Decimal = Decimal("100"),
    wmatic_usd: Decimal = Decimal("100"),
    worthwhile: bool = True,
    slippage_bps: int = 10,
    liquidity: int = 10_000,
    fee_tier: int = 500,
) -> MagicMock:
    market = MagicMock()
    market.timestamp = timestamp
    market.chain = "polygon"

    usdc = TokenBalance(symbol="USDC", balance=usdc_balance, balance_usd=usdc_usd, address="0xusdc")
    wmatic = TokenBalance(symbol="WMATIC", balance=wmatic_balance, balance_usd=wmatic_usd, address="0xwmatic")

    def _balance(token: str):
        if token == "USDC":
            return usdc
        if token == "WMATIC":
            return wmatic
        raise ValueError("unsupported token")

    market.balance.side_effect = _balance
    market.rsi.return_value = RSIData(value=rsi_value, period=14)
    market.ohlcv.return_value = _ohlcv_from_rsi(rsi_value)
    market.pool_price_by_pair.return_value = SimpleNamespace(value=SimpleNamespace(price=Decimal("1")))
    market.pool_reserves.return_value = SimpleNamespace(fee_tier=fee_tier, liquidity=liquidity)
    market.estimate_slippage.return_value = SimpleNamespace(effective_slippage_bps=slippage_bps)
    market.is_trade_worthwhile.return_value = worthwhile
    return market


def test_cross_above_upper_flips_to_wmatic(strategy: WMATICTASwapRSIStrategy):
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    assert strategy.decide(_market(timestamp=t0, rsi_value=Decimal("54"))).intent_type.value == "HOLD"

    t1 = t0 + timedelta(minutes=5)
    intent = strategy.decide(_market(timestamp=t1, rsi_value=Decimal("56")))

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "USDC"
    assert intent.to_token == "WMATIC"


def test_rsi_uses_configured_indicator_token(strategy: WMATICTASwapRSIStrategy):
    market = _market(timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=UTC), rsi_value=Decimal("50"))

    strategy.decide(market)

    market.ohlcv.assert_called_once_with(
        token=strategy.rsi_token,
        timeframe="5m",
        limit=strategy.rsi_period + 20,
        quote=strategy.quote_token,
        pool_address=strategy.target_pool_address,
    )


def test_cross_below_lower_flips_to_usdc(strategy: WMATICTASwapRSIStrategy):
    strategy.position_state = LONG_WMATIC
    strategy.prev_rsi = Decimal("46")

    t1 = datetime(2026, 1, 1, 0, 10, tzinfo=UTC)
    intent = strategy.decide(_market(timestamp=t1, rsi_value=Decimal("44"), wmatic_usd=Decimal("500")))

    assert intent.intent_type.value == "SWAP"
    assert intent.from_token == "WMATIC"
    assert intent.to_token == "USDC"


def test_neutral_band_holds(strategy: WMATICTASwapRSIStrategy):
    strategy.prev_rsi = Decimal("50")
    intent = strategy.decide(_market(timestamp=datetime(2026, 1, 1, 0, 15, tzinfo=UTC), rsi_value=Decimal("52")))
    assert intent.intent_type.value == "HOLD"
    assert strategy.signal_state == NEUTRAL


def test_same_candle_dedupes(strategy: WMATICTASwapRSIStrategy):
    ts = datetime(2026, 1, 1, 0, 20, tzinfo=UTC)
    strategy.prev_rsi = Decimal("50")
    strategy.decide(_market(timestamp=ts, rsi_value=Decimal("56")))
    second = strategy.decide(_market(timestamp=ts, rsi_value=Decimal("56")))
    assert second.intent_type.value == "HOLD"


def test_cooldown_blocks_follow_up_flip(strategy: WMATICTASwapRSIStrategy):
    strategy.prev_rsi = Decimal("54")
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    first = strategy.decide(_market(timestamp=ts, rsi_value=Decimal("56")))
    strategy.on_intent_executed(first, success=True, result=SimpleNamespace())

    second_ts = ts + timedelta(minutes=5)
    second = strategy.decide(_market(timestamp=second_ts, rsi_value=Decimal("44"), wmatic_usd=Decimal("1000")))
    assert second.intent_type.value == "HOLD"


def test_min_notional_guard(strategy: WMATICTASwapRSIStrategy):
    strategy.prev_rsi = Decimal("54")
    ts = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    hold = strategy.decide(
        _market(
            timestamp=ts,
            rsi_value=Decimal("56"),
            usdc_balance=Decimal("1"),
            usdc_usd=Decimal("1"),
        )
    )
    assert hold.intent_type.value == "HOLD"


def test_gas_gate_blocks_trade(strategy: WMATICTASwapRSIStrategy):
    strategy.prev_rsi = Decimal("54")
    ts = datetime(2026, 1, 1, 0, 35, tzinfo=UTC)
    hold = strategy.decide(_market(timestamp=ts, rsi_value=Decimal("56"), worthwhile=False))
    assert hold.intent_type.value == "HOLD"


def test_slippage_gate_blocks_trade(strategy: WMATICTASwapRSIStrategy):
    strategy.prev_rsi = Decimal("54")
    ts = datetime(2026, 1, 1, 0, 40, tzinfo=UTC)
    hold = strategy.decide(_market(timestamp=ts, rsi_value=Decimal("56"), slippage_bps=100))
    assert hold.intent_type.value == "HOLD"


def test_max_trades_per_day_blocks(strategy: WMATICTASwapRSIStrategy):
    strategy.max_trades_per_day = 1
    strategy.trades_today = 1
    strategy.trade_day_utc = "2026-01-01"
    strategy.prev_rsi = Decimal("54")

    hold = strategy.decide(_market(timestamp=datetime(2026, 1, 1, 0, 45, tzinfo=UTC), rsi_value=Decimal("56")))
    assert hold.intent_type.value == "HOLD"


def test_breaker_blocks_after_failures(strategy: WMATICTASwapRSIStrategy):
    strategy.prev_rsi = Decimal("54")
    strategy.consecutive_failed_swaps = strategy.max_consecutive_failures
    hold = strategy.decide(_market(timestamp=datetime(2026, 1, 1, 0, 50, tzinfo=UTC), rsi_value=Decimal("56")))
    assert hold.intent_type.value == "HOLD"
    assert strategy.breaker_until_candle_id is not None


def test_force_action_buy_sell(strategy: WMATICTASwapRSIStrategy):
    ts = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
    strategy.force_action = "buy"
    buy = strategy.decide(_market(timestamp=ts, rsi_value=Decimal("50")))
    assert buy.intent_type.value == "SWAP"
    assert buy.from_token == "USDC"

    strategy.force_action = "sell"
    sell = strategy.decide(_market(timestamp=ts, rsi_value=Decimal("50")))
    assert sell.intent_type.value == "SWAP"
    assert sell.from_token == "WMATIC"


def test_persistence_round_trip(config: dict):
    s1 = WMATICTASwapRSIStrategy(config=config, chain="polygon", wallet_address="0x" + "2" * 40)
    s1.signal_state = LONG_WMATIC
    s1.position_state = LONG_WMATIC
    s1.prev_rsi = Decimal("57")
    s1.trades_today = 3
    s1.trade_day_utc = "2026-01-01"
    s1.consecutive_failed_swaps = 1

    saved = s1.get_persistent_state()

    s2 = WMATICTASwapRSIStrategy(config=config, chain="polygon", wallet_address="0x" + "3" * 40)
    s2.load_persistent_state(saved)

    assert s2.signal_state == LONG_WMATIC
    assert s2.position_state == LONG_WMATIC
    assert s2.prev_rsi == Decimal("57")
    assert s2.trades_today == 3


def test_teardown_unwinds_wmatic(strategy: WMATICTASwapRSIStrategy):
    market = _market(
        timestamp=datetime(2026, 1, 1, 1, 5, tzinfo=UTC),
        rsi_value=Decimal("50"),
        wmatic_balance=Decimal("10"),
        wmatic_usd=Decimal("10"),
    )

    intents = strategy.generate_teardown_intents(mode=TeardownMode.SOFT, market=market)
    assert len(intents) == 1
    assert intents[0].intent_type.value == "SWAP"
    assert intents[0].from_token == "WMATIC"
    assert intents[0].to_token == "USDC"


def test_get_open_positions_empty_when_no_balance(strategy: WMATICTASwapRSIStrategy):
    market = _market(
        timestamp=datetime(2026, 1, 1, 1, 10, tzinfo=UTC),
        rsi_value=Decimal("50"),
        wmatic_balance=Decimal("0"),
        wmatic_usd=Decimal("0"),
    )
    strategy.create_market_snapshot = MagicMock(return_value=market)

    summary = strategy.get_open_positions()
    assert summary.positions == []


def test_on_intent_executed_success_updates_position(strategy: WMATICTASwapRSIStrategy):
    strategy._pending_target_position = LONG_USDC
    strategy._pending_candle_id = 123
    intent = SimpleNamespace(intent_type=SimpleNamespace(value="SWAP"))

    strategy.on_intent_executed(intent, success=True, result=SimpleNamespace())

    assert strategy.position_state == LONG_USDC
    assert strategy.last_flip_candle_id == 123
    assert strategy.consecutive_failed_swaps == 0
