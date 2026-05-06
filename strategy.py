from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from almanak.framework.data.market_snapshot import (
    BalanceUnavailableError,
    GasUnavailableError,
    PoolPriceUnavailableError,
    PoolReservesUnavailableError,
    PriceUnavailableError,
    RSIUnavailableError,
    SlippageEstimateUnavailableError,
)
from almanak.framework.intents import Intent
from almanak.framework.strategies import IntentStrategy, MarketSnapshot, almanak_strategy

logger = logging.getLogger(__name__)


LONG_WMATIC = "LONG_WMATIC"
LONG_USDC = "LONG_USDC"
NEUTRAL = "NEUTRAL"


@dataclass
class RegimeSignal:
    target: str
    reason: str


@almanak_strategy(
    name="w_m_a_t_i_c_t_a_swap_r_s_i",
    description="RSI regime-flip WMATIC/USDC swap strategy on Polygon Uniswap V3",
    version="1.0.0",
    author="Generated",
    tags=["ta_swap", "rsi", "polygon", "uniswap_v3"],
    supported_chains=["polygon"],
    supported_protocols=["uniswap_v3"],
    intent_types=["SWAP", "HOLD"],
    default_chain="polygon",
)
class WMATICTASwapRSIStrategy(IntentStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.protocol = str(self.get_config("protocol", "uniswap_v3"))
        self.base_token = str(self.get_config("base_token", "WMATIC"))
        self.quote_token = str(self.get_config("quote_token", "USDC"))
        self.rsi_token = str(self.get_config("rsi_token", self.base_token))

        self.rsi_period = int(self.get_config("rsi_period", 14))
        self.rsi_timeframe = str(self.get_config("rsi_timeframe", "5m"))
        self.rsi_lower = Decimal(str(self.get_config("rsi_lower", "45")))
        self.rsi_upper = Decimal(str(self.get_config("rsi_upper", "55")))
        self.regime_mode = str(self.get_config("regime_mode", "momentum_flip")).lower()

        self.allocation_pct = Decimal(str(self.get_config("allocation_pct", "0.95")))
        self.min_notional_usd = Decimal(str(self.get_config("min_notional_usd", "25")))
        self.max_slippage = Decimal(str(self.get_config("max_slippage", "0.003")))
        self.max_price_impact = Decimal(str(self.get_config("max_price_impact", "0.003")))
        self.max_estimated_slippage = Decimal(str(self.get_config("max_estimated_slippage", "0.003")))
        self.max_gas_ratio = Decimal(str(self.get_config("max_gas_ratio", "0.05")))

        self.cooldown_candles = int(self.get_config("cooldown_candles", 1))
        raw_cap = self.get_config("max_trades_per_day", None)
        self.max_trades_per_day = int(raw_cap) if raw_cap is not None else None

        self.max_consecutive_failures = int(self.get_config("max_consecutive_failures", 3))
        self.failure_cooldown_candles = int(self.get_config("failure_cooldown_candles", 12))

        self.target_fee_tier = int(self.get_config("target_fee_tier", 500))
        self.target_pool_address = self.get_config("target_pool_address", None)
        self.min_pool_liquidity = int(self.get_config("min_pool_liquidity", 0))

        self.force_action = str(self.get_config("force_action", "") or "").lower()

        self.signal_state = NEUTRAL
        self.position_state = NEUTRAL
        self.prev_rsi: Decimal | None = None
        self.last_processed_candle_id: int | None = None
        self.last_flip_candle_id: int | None = None

        self.trade_day_utc: str = ""
        self.trades_today = 0
        self.consecutive_failed_swaps = 0
        self.breaker_until_candle_id: int | None = None

        self._pending_target_position: str | None = None
        self._pending_candle_id: int | None = None

    def decide(self, market: MarketSnapshot) -> Intent:
        if self.force_action:
            return self._forced_intent(market)

        timestamp = market.timestamp
        candle_id = int(timestamp.timestamp()) // 300
        self._roll_daily_counter(timestamp.date())

        if self.breaker_until_candle_id is not None and candle_id < self.breaker_until_candle_id:
            return Intent.hold(reason="Circuit breaker active after repeated failed swaps")

        if self.consecutive_failed_swaps >= self.max_consecutive_failures:
            self.breaker_until_candle_id = candle_id + self.failure_cooldown_candles
            return Intent.hold(reason="Circuit breaker armed due to failed swaps")

        if self.last_processed_candle_id == candle_id:
            return Intent.hold(reason="Waiting for confirmed 5m candle close")

        try:
            rsi = market.rsi(self.rsi_token, period=self.rsi_period, timeframe=self.rsi_timeframe)
        except (RSIUnavailableError, ValueError):
            self.last_processed_candle_id = candle_id
            return Intent.hold(reason="RSI data unavailable")

        current_rsi = Decimal(str(rsi.value))
        signal = self._compute_signal(current_rsi)
        self.last_processed_candle_id = candle_id

        if signal.target == NEUTRAL:
            self.prev_rsi = current_rsi
            self.signal_state = NEUTRAL
            return Intent.hold(reason=signal.reason)

        if signal.target == self.position_state:
            self.prev_rsi = current_rsi
            self.signal_state = signal.target
            return Intent.hold(reason=f"Already in regime {signal.target}")

        if self.last_flip_candle_id is not None and candle_id - self.last_flip_candle_id <= self.cooldown_candles:
            self.prev_rsi = current_rsi
            self.signal_state = signal.target
            return Intent.hold(reason="Cooldown active")

        if self.max_trades_per_day is not None and self.trades_today >= self.max_trades_per_day:
            self.prev_rsi = current_rsi
            self.signal_state = signal.target
            return Intent.hold(reason="Max trades per day reached")

        if signal.target == LONG_WMATIC:
            from_token = self.quote_token
            to_token = self.base_token
        else:
            from_token = self.base_token
            to_token = self.quote_token

        try:
            source_balance = market.balance(from_token)
        except (BalanceUnavailableError, ValueError):
            self.prev_rsi = current_rsi
            self.signal_state = signal.target
            return Intent.hold(reason=f"Balance unavailable for {from_token}")

        trade_amount = (Decimal(str(source_balance.balance)) * self.allocation_pct).quantize(Decimal("0.00000001"))
        trade_notional_usd = Decimal(str(source_balance.balance_usd)) * self.allocation_pct
        if trade_notional_usd < self.min_notional_usd or trade_amount <= Decimal("0"):
            self.prev_rsi = current_rsi
            self.signal_state = signal.target
            return Intent.hold(reason="Trade notional below minimum threshold")

        sanity_error = self._run_sanity_gates(market, from_token, to_token, trade_amount, trade_notional_usd)
        if sanity_error is not None:
            self.prev_rsi = current_rsi
            self.signal_state = signal.target
            return Intent.hold(reason=sanity_error)

        self.prev_rsi = current_rsi
        self.signal_state = signal.target
        self._pending_target_position = signal.target
        self._pending_candle_id = candle_id

        return Intent.swap(
            from_token=from_token,
            to_token=to_token,
            amount=trade_amount,
            max_slippage=self.max_slippage,
            max_price_impact=self.max_price_impact,
            protocol=self.protocol,
            chain=self.chain,
        )

    def _compute_signal(self, current_rsi: Decimal) -> RegimeSignal:
        if self.prev_rsi is None:
            return RegimeSignal(NEUTRAL, "Priming crossover state")

        crossed_up = self.prev_rsi <= self.rsi_upper and current_rsi > self.rsi_upper
        crossed_down = self.prev_rsi >= self.rsi_lower and current_rsi < self.rsi_lower

        if self.regime_mode == "reversal":
            if crossed_up:
                return RegimeSignal(LONG_USDC, f"Reversal: RSI crossed above {self.rsi_upper}")
            if crossed_down:
                return RegimeSignal(LONG_WMATIC, f"Reversal: RSI crossed below {self.rsi_lower}")
        else:
            if crossed_up:
                return RegimeSignal(LONG_WMATIC, f"Momentum: RSI crossed above {self.rsi_upper}")
            if crossed_down:
                return RegimeSignal(LONG_USDC, f"Momentum: RSI crossed below {self.rsi_lower}")

        return RegimeSignal(NEUTRAL, "RSI in neutral band")

    def _run_sanity_gates(
        self,
        market: MarketSnapshot,
        from_token: str,
        to_token: str,
        trade_amount: Decimal,
        trade_notional_usd: Decimal,
    ) -> str | None:
        try:
            market.pool_price_by_pair(
                token_a=self.base_token,
                token_b=self.quote_token,
                protocol=self.protocol,
                fee_tier=self.target_fee_tier,
                chain=self.chain,
            )
        except (PoolPriceUnavailableError, ValueError):
            return f"Target {self.target_fee_tier} fee tier pool unavailable"

        if self.target_pool_address:
            try:
                pool = market.pool_reserves(self.target_pool_address, chain=self.chain)
            except (PoolReservesUnavailableError, ValueError):
                return "Pool reserves unavailable"

            fee_tier = getattr(pool, "fee_tier", None)
            liquidity = getattr(pool, "liquidity", None)
            if fee_tier is not None and int(fee_tier) != self.target_fee_tier:
                return "Configured pool fee tier mismatch"
            if liquidity is not None and int(liquidity) < self.min_pool_liquidity:
                return "Pool liquidity below configured minimum"

        try:
            slippage = market.estimate_slippage(
                token_in=from_token,
                token_out=to_token,
                amount=trade_amount,
                chain=self.chain,
                protocol=self.protocol,
            )
            slippage_pct = Decimal(str(slippage.effective_slippage_bps)) / Decimal("10000")
            if slippage_pct > self.max_estimated_slippage:
                return "Estimated slippage above configured limit"
        except (SlippageEstimateUnavailableError, ValueError):
            return "Slippage estimate unavailable"

        try:
            if not market.is_trade_worthwhile(
                amount_usd=trade_notional_usd,
                chain=self.chain,
                max_gas_ratio=self.max_gas_ratio,
            ):
                return "Gas cost too high for trade size"
        except (GasUnavailableError, ValueError):
            return "Gas data unavailable"

        return None

    def _forced_intent(self, market: MarketSnapshot) -> Intent:
        if self.force_action == "buy":
            try:
                quote_balance = market.balance(self.quote_token)
            except (BalanceUnavailableError, ValueError):
                raise ValueError("force_action=buy requires quote balance")

            amount = (Decimal(str(quote_balance.balance)) * self.allocation_pct).quantize(Decimal("0.00000001"))
            if amount <= Decimal("0"):
                raise ValueError("force_action=buy has zero source amount")
            return Intent.swap(
                from_token=self.quote_token,
                to_token=self.base_token,
                amount=amount,
                max_slippage=self.max_slippage,
                max_price_impact=self.max_price_impact,
                protocol=self.protocol,
                chain=self.chain,
            )

        if self.force_action == "sell":
            try:
                base_balance = market.balance(self.base_token)
            except (BalanceUnavailableError, ValueError):
                raise ValueError("force_action=sell requires base balance")

            amount = (Decimal(str(base_balance.balance)) * self.allocation_pct).quantize(Decimal("0.00000001"))
            if amount <= Decimal("0"):
                raise ValueError("force_action=sell has zero source amount")
            return Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount=amount,
                max_slippage=self.max_slippage,
                max_price_impact=self.max_price_impact,
                protocol=self.protocol,
                chain=self.chain,
            )

        raise ValueError(f"Unknown force_action: {self.force_action!r}")

    def _roll_daily_counter(self, current_day: date) -> None:
        day_str = current_day.isoformat()
        if self.trade_day_utc != day_str:
            self.trade_day_utc = day_str
            self.trades_today = 0

    def on_intent_executed(self, intent, success: bool, result) -> None:
        if getattr(intent, "intent_type", None) is None:
            return

        if getattr(intent.intent_type, "value", "") != "SWAP":
            return

        if success:
            if self._pending_target_position is not None:
                self.position_state = self._pending_target_position
            if self._pending_candle_id is not None:
                self.last_flip_candle_id = self._pending_candle_id
            self.trades_today += 1
            self.consecutive_failed_swaps = 0
            self.breaker_until_candle_id = None
            self._pending_target_position = None
            self._pending_candle_id = None
            return

        self.consecutive_failed_swaps += 1
        self._pending_target_position = None
        self._pending_candle_id = None

    def get_persistent_state(self) -> dict[str, Any]:
        return {
            "signal_state": self.signal_state,
            "position_state": self.position_state,
            "prev_rsi": str(self.prev_rsi) if self.prev_rsi is not None else None,
            "last_processed_candle_id": self.last_processed_candle_id,
            "last_flip_candle_id": self.last_flip_candle_id,
            "trade_day_utc": self.trade_day_utc,
            "trades_today": self.trades_today,
            "consecutive_failed_swaps": self.consecutive_failed_swaps,
            "breaker_until_candle_id": self.breaker_until_candle_id,
        }

    def load_persistent_state(self, state: dict[str, Any]) -> None:
        if not state:
            return

        self.signal_state = str(state.get("signal_state", NEUTRAL))
        self.position_state = str(state.get("position_state", NEUTRAL))
        prev_rsi_raw = state.get("prev_rsi")
        self.prev_rsi = Decimal(str(prev_rsi_raw)) if prev_rsi_raw is not None else None
        self.last_processed_candle_id = state.get("last_processed_candle_id")
        self.last_flip_candle_id = state.get("last_flip_candle_id")
        self.trade_day_utc = str(state.get("trade_day_utc", ""))
        self.trades_today = int(state.get("trades_today", 0))
        self.consecutive_failed_swaps = int(state.get("consecutive_failed_swaps", 0))
        breaker_raw = state.get("breaker_until_candle_id")
        self.breaker_until_candle_id = int(breaker_raw) if breaker_raw is not None else None

    def supports_teardown(self) -> bool:
        return True

    def get_open_positions(self):
        from almanak.framework.teardown import PositionInfo, PositionType, TeardownPositionSummary

        try:
            market = self.create_market_snapshot()
            balance = market.balance(self.base_token)
        except (BalanceUnavailableError, ValueError):
            return TeardownPositionSummary.empty(getattr(self, "strategy_id", self.STRATEGY_NAME))

        if balance.balance <= Decimal("0"):
            return TeardownPositionSummary.empty(getattr(self, "strategy_id", self.STRATEGY_NAME))

        return TeardownPositionSummary(
            strategy_id=getattr(self, "strategy_id", self.STRATEGY_NAME),
            timestamp=datetime.now(UTC),
            positions=[
                PositionInfo(
                    position_type=PositionType.TOKEN,
                    position_id=f"{self.STRATEGY_NAME}_wmatic",
                    chain=self.chain,
                    protocol=self.protocol,
                    value_usd=Decimal(str(balance.balance_usd)),
                    details={
                        "asset": self.base_token,
                        "target_exit_asset": self.quote_token,
                        "amount": str(balance.balance),
                    },
                )
            ],
        )

    def generate_teardown_intents(self, mode, market=None) -> list[Intent]:
        from almanak.framework.teardown import TeardownMode

        snapshot = market
        if snapshot is None:
            snapshot = self.create_market_snapshot()

        try:
            base_balance = snapshot.balance(self.base_token)
        except (BalanceUnavailableError, ValueError):
            return []

        if base_balance.balance <= Decimal("0"):
            return []

        slippage = Decimal("0.03") if mode == TeardownMode.HARD else self.max_slippage
        return [
            Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount="all",
                max_slippage=slippage,
                protocol=self.protocol,
                chain=self.chain,
            )
        ]

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": self.STRATEGY_NAME,
            "chain": self.chain,
            "signal_state": self.signal_state,
            "position_state": self.position_state,
            "trades_today": self.trades_today,
            "consecutive_failed_swaps": self.consecutive_failed_swaps,
            "breaker_until_candle_id": self.breaker_until_candle_id,
        }
