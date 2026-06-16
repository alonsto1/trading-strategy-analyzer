# optimized_backtester.py


from __future__ import annotations
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

import math

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    SESSION_GAP: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    BAR_INTERVAL: timedelta = field(default_factory=lambda: timedelta(seconds=5))
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    ROC_PERIOD: int = 9
    EMA_TREND_PERIOD: int = 35
    EMA_MAJOR_PERIOD: int = 140
    EMA_BIG_PERIOD: int = 1000
    ROC_THRESHOLD: float = 0.00  # relaxed from 0.03
    VOLUME_PERIOD: int = 20
    ATR_SL_MULTIPLE: float = 1.0
    ATR_TP_MULTIPLE: float = 7.0
    RISK_PERCENT: float = 0.0001
    ACCOUNT_EQUITY: float = 100_000.0
    MIN_CANDLES: int = 300
    DEBUG: bool = True
    HOLD_TIMEOUT_MINUTES: float = 60.0
    ALLOW_SHORT: bool = True
    MIN_ATR_THRESHOLD: float = 0.68 # relaxed from 5.0
    MAX_ATR_THRESHOLD: float = 3.47
    SLOPE_THRESHOLD: float = 110.45
    MAX_LOSS: float = 6.0



    DOLLAR_PER_POINT: float = 2.0
    FEE_PER_CON: float = 0.74

    enter_over_risk: bool = True


class Backtester:
    """Vectorised re‑implementation of the original back‑testing engine."""

    def __init__(self, cfg: Optional[StrategyConfig] = None) -> None:
        self.cfg = cfg or StrategyConfig()

    # ---------------------------------------------------------------------
    # Indicator helpers – unchanged except for minor NumPy micro‑optimisations
    # ---------------------------------------------------------------------
    def _ema_np(self, arr: np.ndarray, period: int) -> np.ndarray:
        if arr.size < period:
            return np.full_like(arr, np.nan)
        return pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()

    def _macd_np(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fast = self._ema_np(arr, self.cfg.MACD_FAST)
        slow = self._ema_np(arr, self.cfg.MACD_SLOW)
        macd = fast - slow
        signal = pd.Series(macd).ewm(span=self.cfg.MACD_SIGNAL, adjust=False).mean().to_numpy()
        return macd, signal, macd - signal

    def _roc_np(self, arr: np.ndarray, period: int) -> np.ndarray:
        out = np.full_like(arr, np.nan)
        if arr.size > period:
            prev = arr[:-period]
            curr = arr[period:]
            out[period:] = (curr - prev) / prev * 100.0
        return out

    def _atr_np(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        prev_close = closes[:-1]
        tr = np.maximum.reduce([
            highs[1:] - lows[1:],
            np.abs(highs[1:] - prev_close),
            np.abs(lows[1:] - prev_close),
        ])
        tr = np.insert(tr, 0, np.nan)
        return pd.Series(tr).rolling(period).mean().to_numpy()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["c"].to_numpy()
        high = df["h"].to_numpy()
        low = df["l"].to_numpy()

        df["ema_50"] = self._ema_np(close, self.cfg.EMA_TREND_PERIOD)
        df["ema_200"] = self._ema_np(close, self.cfg.EMA_MAJOR_PERIOD)
        df["ema_500"] = self._ema_np(close, self.cfg.EMA_BIG_PERIOD)

        macd, signal, _ = self._macd_np(close)
        df["macd_line"], df["macd_signal"] = macd, signal
        df["roc"] = self._roc_np(close, self.cfg.ROC_PERIOD)
        df["atr"] = self._atr_np(high, low, close)

        df["ema_50_slope"] = (df["ema_50"].diff())/df["atr"]
        df["ema_200_slope"] = (df["ema_200"].diff(periods=3))/df["atr"]
        df["ema_500_slope"] = (df['ema_500'].diff(periods=25))/df["atr"]
        df["macd_slope"] = df["macd_line"].diff(periods=7)/df["atr"]
        df["atr_slope"] = (df["atr"].diff(periods=3))/df["atr"]

        df["ema_50_diff_slope"] = abs(df["ema_50"].diff(5) / df["atr"] - df["ema_50_slope"])

        df["ema_50_slopes_diff"] = abs(df["ema_50"].diff() / df["atr"] - df["ema_50_slope"].diff())


        df["avg_vol_15"] = df["v"].rolling(window=25).mean()
        df["vol_diff_15"] = (df["v"] - df["avg_vol_15"])/df["avg_vol_15"]
        df["avg_vol_slope"] = df["avg_vol_15"].diff(periods=6)/df["avg_vol_15"]

        df['checks'] = (df["ema_200"].diff(periods=80))/df["atr"]

        return df

    # ------------------------------------------------------------------
    # Position sizing helper (unchanged)
    # ------------------------------------------------------------------
    def _position_size(self, atr: float) -> int:
        dollar_risk = self.cfg.ACCOUNT_EQUITY * self.cfg.RISK_PERCENT
        cons = dollar_risk / (atr * self.cfg.ATR_SL_MULTIPLE * self.cfg.DOLLAR_PER_POINT)
        cons = min(int(cons), 20)

        if self.cfg.enter_over_risk:
            if cons < 1:
                cons = 1


        return cons

    # ------------------------------------------------------------------
    # Main back‑test routine – rewritten to iterate over NumPy arrays
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Union[List, float]]:
        # ------------------------------------------------------------------
        # 1.  Load and pre‑process candles & indicators
        # ------------------------------------------------------------------
        df = pd.read_csv("data2.csv", parse_dates=["t"])
        df = df[df["t"].dt.weekday < 5].reset_index(drop=True)
        df = self._compute_indicators(df)

        # Extract the required columns to NumPy ndarrays **once**.
        cols = [
            "t", "c", "h", "l", "ema_50", "ema_200", "ema_500", "ema_50_slope",
            "ema_200_slope", "ema_500_slope", "macd_line", "macd_signal", "macd_slope", "roc", "atr",
            "atr_slope", "ema_50_diff_slope", "ema_50_slopes_diff","checks", "avg_vol_slope", "avg_vol_15", "vol_diff_15"
        ]
        arrays = {name: df[name].to_numpy() for name in cols}
        t_arr = arrays["t"]  # original Timestamp objects for logging / output
        hour_arr = df["t"].dt.hour.to_numpy()
        minute_arr = df["t"].dt.minute.to_numpy()

        # Short aliases for config constants used *inside* the hot loop – avoids
        # repeated attribute look‑ups.
        cfg = self.cfg
        SLOPE_TH = cfg.SLOPE_THRESHOLD
        ROC_TH = cfg.ROC_THRESHOLD
        ATR_MIN, ATR_MAX = cfg.MIN_ATR_THRESHOLD, cfg.MAX_ATR_THRESHOLD
        MAX_LOSS = cfg.MAX_LOSS
        DOLLAR_PP = cfg.DOLLAR_PER_POINT
        FEE = cfg.FEE_PER_CON

        # ------------------------------------------------------------------
        # 2.  Runtime state
        # ------------------------------------------------------------------
        trades: List[Dict[str, Union[str, float, datetime]]] = []
        profit = 0.0


        pos_open = False
        side = ""
        entry_price = stop_price = take_profit = big_stop_price = 0.0
        entry_time: Optional[pd.Timestamp] = None
        position_size = 0
        fifteen_sec_snapshot: Optional[float] = None
        fifteen_sec_recorded: bool = False
        entry_atr: Optional[float] = None

        # Indicator snapshots at entry (for later diagnostics)
        entry_snapshot: Dict[str, Optional[float]] = {
            "macd_line": None,
            "macd_signal": None,
            "macd_slope": None,
            "roc": None,
            "ema_50": None,
            "ema_200": None,
            "ema_50_slope": None,
            "ema_200_slope": None,
            "ema_500_slope": None,
            "atr": None,
            "atr_slope": None,
            "checks": None,
            "ema_50_diff_slope": None,
            "ema_50_slopes_diff": None,
            "ema200_dist_atr": None,
            "ema200_ema500_dist_atr": None,
            "ema50_ema200_dist_atr": None,
            "avg_vol_15": None,
            "vol_diff_15": None,
            "avg_vol_slope": None,
        }

        # ------------------------------------------------------------------
        # 3.  Main loop – iterate using **raw indices** into the ndarrays
        # ------------------------------------------------------------------
        for i in range(cfg.MIN_CANDLES, len(df)):
            # Fast array access
            atr = arrays["atr"][i]
            if np.isnan(atr):
                continue



            # Vital fields for this bar
            t = t_arr[i]
            hour = hour_arr[i]
            minute = minute_arr[i]
            c = arrays["c"][i]
            h_price = arrays["h"][i]
            l_price = arrays["l"][i]
            macd_line = arrays["macd_line"][i]
            macd_sig = arrays["macd_signal"][i]
            roc = arrays["roc"][i]
            ema_50 = arrays["ema_50"][i]
            ema_200 = arrays["ema_200"][i]
            ema_500 = arrays["ema_500"][i]
            ema_50_slope = arrays["ema_50_slope"][i]
            ema_200_slope = arrays["ema_200_slope"][i]
            ema_500_slope = arrays["ema_500_slope"][i]
            macd_slope = arrays["macd_slope"][i]
            atr_slope = arrays["atr_slope"][i]
            checks = arrays["checks"][i]
            avg_vol_slope = arrays["avg_vol_slope"][i]
            ema_50_diff_slope = arrays["ema_50_diff_slope"][i]
            ema_50_slopes_diff = arrays["ema_50_slopes_diff"][i]
            avg_vol = arrays["avg_vol_15"][i]
            vol_diff = arrays["vol_diff_15"][i]

            # Previous bar (index i‑1) values that are needed exactly once
            prev_macd_line = arrays["macd_line"][i - 1]
            prev_macd_sig = arrays["macd_signal"][i - 1]


            #DATA CORRUPTION
            if t == "2025-03-21 13:29:55+00:00":
                i += 1006
                continue

            if pos_open and not fifteen_sec_recorded and (t - entry_time) >= timedelta(seconds=15):
                # deviation in ATR units:
                if side == "long":
                    diff = (c - entry_price) / entry_atr
                else:  # short
                    diff = (entry_price - c) / entry_atr

                # store it (long above=positive, below negative; short reversed)
                fifteen_sec_snapshot = diff
                fifteen_sec_recorded = True





            # --------------------------------------------------------------
            # A. Handle **open position** exits first
            # --------------------------------------------------------------
            if pos_open:
                # Hard stop (absolute loss cap)
                if (side == "long" and l_price <= big_stop_price) or (
                    side == "short" and h_price >= big_stop_price
                ):
                    exit_price = (big_stop_price - 0.25) if side == "long" else (big_stop_price + 0.25)
                    exit_reason = "MAX loss"
                # Timeout / take‑profit / session timeout
                elif (
                    t - entry_time > timedelta(minutes=cfg.HOLD_TIMEOUT_MINUTES)
                    or (side == "long" and h_price > take_profit)
                    or (side == "short" and l_price < take_profit)
                ):
                    if (side == "long" and h_price > take_profit) or (
                        side == "short" and l_price < take_profit
                    ):
                        exit_price = take_profit
                        exit_reason = "TP"
                    else:
                        exit_price = c
                        exit_reason = "TIMEOUT"
                # In‑bar SL/TP logic
                elif (
                    side == "long" and h_price <= stop_price and ema_50_slope > SLOPE_TH
                ) or (
                    side == "short" and l_price >= stop_price and ema_50_slope < -SLOPE_TH
                ):
                    exit_price = c
                    exit_reason = "STOP"
                elif (
                    side == "long" and h_price >= take_profit
                ) or (
                    side == "short" and l_price <= take_profit
                ):
                    exit_price = take_profit
                    exit_reason = "TAKE_PROFIT"
                # End‑of‑day forced exit
                elif hour == 20 and minute == 9:
                    exit_price = c
                    exit_reason = "MARKET_CLOSE"
                else:
                    exit_price = None  # stay in trade

                if exit_price is not None:
                    pnl = (
                        (exit_price - entry_price) * position_size * DOLLAR_PP
                        if side == "long"
                        else (entry_price - exit_price) * position_size * DOLLAR_PP
                    ) - (FEE * position_size)

                    if cfg.DEBUG:
                        print(f"{t} EXIT {side.upper()} @ {exit_price:.2f} [{exit_reason}] PnL={pnl:.2f}")

                    trades.append(
                        {
                            "side": side,
                            "entry_time": entry_time,
                            "exit_time": t,
                            "entry_px": entry_price,
                            "exit_px": exit_price,
                            "pnl": pnl,
                            "15s_atr_dev": fifteen_sec_snapshot,
                            "fee": position_size * FEE,
                            **{k: v for k, v in entry_snapshot.items()},
                            "position_size": position_size,
                            "exit_reason": exit_reason,
                        }
                    )
                    profit += pnl
                    pos_open = False
                    fifteen_sec_snapshot = None
                    fifteen_sec_recorded = False
                    entry_atr = None
                    # Clear snapshot
                    for k in entry_snapshot:
                        entry_snapshot[k] = None


            # --------------------------------------------------------------
            # B. Skip bars outside trading window when **no position**
            # --------------------------------------------------------------
            if not pos_open:
                if hour < 13 or (hour == 13 and minute < 35) or hour > 19:
                    continue

            # --------------------------------------------------------------
            # C. Trade management logic when position **open**
            # --------------------------------------------------------------
            if pos_open:
                # Track dynamic stop / TP hits already handled; nothing else to do
                continue

            # --------------------------------------------------------------
            # D. Entry conditions (no open position)
            # --------------------------------------------------------------



            ema200_ema500_dist_atr = abs(ema_200 - ema_500) / atr
            ema200_dist_atr = abs(c - ema_200) / atr
            ema50_ema200_dist_atr = abs(ema_50 - ema_200) / atr

            if not (
                avg_vol_slope > 0.06 and
                0.96 > ema_500_slope > -0.17 and
                0.16 > macd_slope > -0.18 and
                abs(roc) > 0.002

            ):

                continue

            long_signal = (
                    macd_line < macd_sig and
                    prev_macd_line >= prev_macd_sig and
                    ema_50_slope < ema_200_slope and
                    ema_50 > ema_200 and
                    avg_vol_slope > vol_diff and
                    roc < 0.03

            )
            short_signal = (
                    macd_line > macd_sig and
                    prev_macd_line <= prev_macd_sig and
                    ema_50_slope > ema_200_slope and
                    ema_50 < ema_200 and
                    avg_vol_slope > vol_diff and
                    roc > -0.01

            )



            if not long_signal and not (short_signal and cfg.ALLOW_SHORT):
                continue

            # ----------------------------------------------------------
            # E. Open the position – snapshot indicator values immediately
            # ----------------------------------------------------------
            position_size = self._position_size(atr)
            if position_size < 1:
                continue

            pos_open = True
            side = "long" if long_signal else "short"
            entry_price = c
            entry_time = t

            entry_atr = atr
            fifteen_sec_snapshot = None
            fifteen_sec_recorded = False


            # Fill snapshot dict for later export
            entry_snapshot.update(
                macd_line=macd_line,
                macd_signal=macd_sig,
                macd_slope=macd_slope,
                roc=roc,
                ema_50=ema_50,
                ema_200=ema_200,
                ema_50_slope=ema_50_slope,
                ema_200_slope=ema_200_slope,
                ema_500_slope=ema_500_slope,
                atr=atr,
                atr_slope=atr_slope,
                checks=checks,
                avg_vol_slope=avg_vol_slope,
                ema_50_diff_slope=ema_50_diff_slope,
                ema_50_slopes_diff=ema_50_slopes_diff,
                ema200_dist_atr=ema200_dist_atr,
                ema200_ema500_dist_atr=ema200_ema500_dist_atr,
                ema50_ema200_dist_atr=ema50_ema200_dist_atr,
                avg_vol_15=arrays["avg_vol_15"][i],
                vol_diff_15=arrays["vol_diff_15"][i],
            )

            stop_price = (
                entry_price - cfg.ATR_SL_MULTIPLE * atr
                if long_signal
                else entry_price + cfg.ATR_SL_MULTIPLE * atr
            )
            big_stop_price = (
                entry_price - (MAX_LOSS * atr)
                if long_signal
                else entry_price + (MAX_LOSS * atr)
            )



            take_profit = (
                entry_price + cfg.ATR_TP_MULTIPLE * atr
                if long_signal
                else entry_price - cfg.ATR_TP_MULTIPLE * atr
            )

            if cfg.DEBUG:
                print(
                    f"{t} ENTRY {side.upper()} @ {entry_price:.2f} SIZE={position_size} "
                    f"SL={big_stop_price:.2f} TP={take_profit:.2f}"
                )

        # ------------------------------------------------------------------
        # 4.  Return structure (unchanged vs original)
        # ------------------------------------------------------------------
        return {"trades": trades, "pnl": profit, "candles": df.to_dict("records")}


if __name__ == "__main__":
    bt = Backtester()
    results = bt.run()
    df_trades = pd.DataFrame(results["trades"])
    print(f"Total PnL: {results['pnl']:.2f}")
    print(f"Trades: {len(df_trades)}")

    if not df_trades.empty:
        df_trades["win"] = df_trades["pnl"] > 0
        win_rate = df_trades["win"].mean()
        print(f"Win Rate: {win_rate:.2%}")
        fname = f"trades.csv"
        df_trades.to_csv(fname, index=False)
        print(f"Saved trades to {fname}")
