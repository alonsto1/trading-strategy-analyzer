"""
Backtest Visualizer — interactive toolkit for analysing trade output CSV.
- Equity curve, PnL-by-hour, PnL-by-indicator-ranges, heatmaps
- Summary stats panel, side filter, non‑blocking plots
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_equity(trades: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Return (timestamps, cumulative_pnl) from a sorted DataFrame."""
    t = trades["exit_time"] if "exit_time" in trades.columns else trades.index
    if isinstance(t.iloc[0], str):
        t = pd.to_datetime(t)
    ts = t.tolist()
    eq = trades["pnl"].cumsum().tolist()
    return ts, eq


def _summary_stats(trades: pd.DataFrame) -> dict[str, str]:
    """Compute a small summary dict from trade data."""
    if trades.empty:
        return {"total_pnl": "—", "trades": "0", "win_rate": "—",
                "avg_pnl": "—", "max_dd": "—", "sharpe": "—"}

    total = trades["pnl"].sum()
    n = len(trades)
    wr = (trades["pnl"] > 0).mean()
    avg = trades["pnl"].mean()

    cum = trades["pnl"].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()

    sharpe = (trades["pnl"].mean() / trades["pnl"].std()) * np.sqrt(252) \
        if trades["pnl"].std() > 1e-9 else 0.0

    return {
        "total_pnl": f"${total:+,.2f}",
        "trades": str(n),
        "win_rate": f"{wr:.1%}",
        "avg_pnl": f"${avg:+,.2f}",
        "max_dd": f"${max_dd:+,.2f}",
        "sharpe": f"{sharpe:.2f}",
    }


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class BacktestVisualizer:
    """Interactive visualiser for backtest trade CSV files."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("Backtest Visualizer")

        self.df: Optional[pd.DataFrame] = None

        # ── Left control panel ──────────────────────────────────────────
        controls = ttk.Frame(master, padding=10)
        controls.grid(row=0, column=0, sticky="nw")

        # 1. Load file
        ttk.Button(controls, text="Load CSV", command=self.load_file) \
            .grid(row=0, column=0, pady=5, sticky="ew")
        ttk.Separator(controls, orient="horizontal") \
            .grid(row=1, column=0, sticky="ew", pady=5)

        # 2. Summary stats label
        self.summary_var = tk.StringVar(value="No data loaded.")
        ttk.Label(controls, textvariable=self.summary_var,
                  font=("Consolas", 10), foreground="#333") \
            .grid(row=2, column=0, pady=5, sticky="w")
        ttk.Separator(controls, orient="horizontal") \
            .grid(row=3, column=0, sticky="ew", pady=5)

        # 3. Equity curve
        self.equity_btn = ttk.Button(controls, text="Equity Curve",
                                     command=self.plot_equity, state="disabled")
        self.equity_btn.grid(row=4, column=0, pady=5)

        # 4. PnL by hour
        self.plot_hour_btn = ttk.Button(controls, text="PnL by Hour",
                                        command=self.plot_pnl_by_hour,
                                        state="disabled")
        self.plot_hour_btn.grid(row=5, column=0, pady=5)

        # 5. PnL by indicator ranges
        ttk.Label(controls, text="Indicator:").grid(row=6, column=0, sticky="w")
        self.indicator_var = tk.StringVar()
        self.indicator_combo = ttk.Combobox(controls,
                                            textvariable=self.indicator_var,
                                            state="disabled")
        self.indicator_combo.grid(row=7, column=0, pady=5)

        ttk.Label(controls, text="Bins:").grid(row=8, column=0, sticky="w")
        self.bins_var = tk.IntVar(value=5)
        self.bins_entry = ttk.Entry(controls, textvariable=self.bins_var,
                                    width=5, state="disabled")
        self.bins_entry.grid(row=9, column=0, pady=5)

        self.plot_ranges_btn = ttk.Button(controls,
                                          text="PnL by Indicator Ranges",
                                          command=self.plot_pnl_ranges,
                                          state="disabled")
        self.plot_ranges_btn.grid(row=10, column=0, pady=5)

        # 6. Heatmap
        ttk.Separator(controls, orient="horizontal") \
            .grid(row=11, column=0, sticky="ew", pady=5)
        ttk.Label(controls, text="X Indicator:").grid(row=12, column=0, sticky="w")
        self.ind1_var = tk.StringVar()
        self.ind1_combo = ttk.Combobox(controls, textvariable=self.ind1_var,
                                       state="disabled")
        self.ind1_combo.grid(row=13, column=0, pady=5)

        ttk.Label(controls, text="Y Indicator:").grid(row=14, column=0, sticky="w")
        self.ind2_var = tk.StringVar()
        self.ind2_combo = ttk.Combobox(controls, textvariable=self.ind2_var,
                                       state="disabled")
        self.ind2_combo.grid(row=15, column=0, pady=5)

        ttk.Label(controls, text="Bins (X x Y):").grid(row=16, column=0, sticky="w")
        bin_frame = ttk.Frame(controls)
        bin_frame.grid(row=17, column=0, pady=5, sticky="w")
        self.bins1_var = tk.IntVar(value=5)
        self.bins2_var = tk.IntVar(value=5)
        self.bins1_entry = ttk.Entry(bin_frame, textvariable=self.bins1_var,
                                     width=5, state="disabled")
        self.bins1_entry.grid(row=0, column=0)
        ttk.Label(bin_frame, text="x").grid(row=0, column=1)
        self.bins2_entry = ttk.Entry(bin_frame, textvariable=self.bins2_var,
                                     width=5, state="disabled")
        self.bins2_entry.grid(row=0, column=2)

        ttk.Label(controls, text="Heatmap Metric:").grid(row=18, column=0, sticky="w")
        self.metric_var = tk.StringVar(value="net_pnl")
        self.metric_combo = ttk.Combobox(
            controls, textvariable=self.metric_var,
            values=["net_pnl", "win_rate"], state="disabled")
        self.metric_combo.grid(row=19, column=0, pady=5)

        # 7. Side filter (shared)
        ttk.Label(controls, text="Side filter:").grid(row=20, column=0, sticky="w")
        self.side_var = tk.StringVar(value="both")
        self.side_combo = ttk.Combobox(
            controls, textvariable=self.side_var,
            values=["both", "long", "short"], state="disabled")
        self.side_combo.grid(row=21, column=0, pady=5)

        self.plot_heatmap_btn = ttk.Button(controls, text="Plot Heatmap",
                                           command=self.plot_heatmap,
                                           state="disabled")
        self.plot_heatmap_btn.grid(row=22, column=0, pady=5)

        # Toggle-on-load widgets
        self._on_load = [
            self.indicator_combo, self.bins_entry,
            self.ind1_combo, self.ind2_combo,
            self.bins1_entry, self.bins2_entry,
            self.metric_combo, self.side_combo,
            self.equity_btn,
            self.plot_hour_btn, self.plot_ranges_btn, self.plot_heatmap_btn,
        ]

        # ── Auto-load ───────────────────────────────────────────────────
        default_path = Path("trades.csv")
        if default_path.exists():
            self.load_file(default_path)
        else:
            messagebox.showinfo(
                "Info",
                "trades.csv not found. Run backtester.py first, "
                "then launch visualizer.py, or use 'Load CSV'."
            )

    # ── helpers ──────────────────────────────────────────────────────────

    def _enable_widgets(self) -> None:
        for w in self._on_load:
            w.configure(state="normal")

    def _filter_side(self, df: pd.DataFrame) -> pd.DataFrame:
        side = self.side_var.get().lower()
        if side != "both" and "side" in df.columns:
            df = df[df["side"].str.lower() == side]
        return df

    # ── Load ─────────────────────────────────────────────────────────────

    def load_file(self, path: Optional[Union[str, Path]] = None) -> None:
        if path is None:
            path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv")]
            )
            if not path:
                return
        try:
            self.df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load {path}:\n{exc}")
            return

        numeric_cols = self.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        for combo in (self.indicator_combo, self.ind1_combo, self.ind2_combo):
            combo.configure(values=numeric_cols)

        self._enable_widgets()
        self._update_summary()

    def _update_summary(self) -> None:
        if self.df is None or self.df.empty:
            self.summary_var.set("No data.")
            return
        df = self._filter_side(self.df.dropna(subset=["pnl"]))
        stats = _summary_stats(df)
        self.summary_var.set(
            f"PnL: {stats['total_pnl']:>10}  |  Trades: {stats['trades']:>4}\n"
            f"Win%: {stats['win_rate']:>10}  |  Avg: {stats['avg_pnl']:>8}\n"
            f"MaxDD:{stats['max_dd']:>11}  |  Sharpe: {stats['sharpe']:>5}"
        )

    # ── Equity curve ─────────────────────────────────────────────────────

    def plot_equity(self) -> None:
        if self.df is None:
            return
        df = self.df.dropna(subset=["pnl"]).copy()
        df = self._filter_side(df)
        if df.empty:
            messagebox.showwarning("Empty", "No trades after side filter.")
            return

        sort_col = "exit_time" if "exit_time" in df.columns else None
        if sort_col:
            df = df.sort_values(sort_col)

        ts, eq = _compute_equity(df)
        stats = _summary_stats(df)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(range(len(eq)), eq, alpha=0.3)
        ax.plot(range(len(eq)), eq, linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL ($)")

        info = (
            f"Total: {stats['total_pnl']}   "
            f"Win: {stats['win_rate']}   "
            f"Trades: {stats['trades']}   "
            f"MaxDD: {stats['max_dd']}"
        )
        ax.set_title(
            f"Equity Curve \u2014 {info}".replace("$", "\\$"),
            fontsize=10,
        )
        fig.tight_layout()
        plt.show(block=False)

    # ── PnL by hour ──────────────────────────────────────────────────────

    def plot_pnl_by_hour(self) -> None:
        if self.df is None:
            return
        if "entry_time" not in self.df.columns:
            messagebox.showerror("Missing column", "'entry_time' not found.")
            return

        df = self.df.dropna(subset=["entry_time", "pnl"]).copy()
        df = self._filter_side(df)
        if df.empty:
            return

        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
        df = df.dropna(subset=["entry_time"])
        df["hour"] = df["entry_time"].dt.hour

        agg = df.groupby("hour").agg(
            net_pnl=("pnl", "sum"), trades=("pnl", "count")
        ).reset_index()

        fig, ax = plt.subplots()
        bars = ax.bar(agg["hour"], agg["net_pnl"], width=0.8)
        ax.set_xlabel("Entry Hour (UTC)")
        ax.set_ylabel("Net PnL ($)")

        for rect, count in zip(bars, agg["trades"]):
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h, str(count),
                    ha="center", va="bottom" if h >= 0 else "top")

        fig.tight_layout()
        plt.show(block=False)

    # ── PnL by indicator ranges ──────────────────────────────────────────

    def plot_pnl_ranges(self) -> None:
        if self.df is None:
            return
        ind = self.indicator_var.get()
        if not ind:
            messagebox.showwarning("Pick indicator", "Choose an indicator first.")
            return
        bins = self.bins_var.get()

        df = self.df.dropna(subset=[ind, "pnl"]).copy()
        df = self._filter_side(df)
        if df.empty:
            return

        df["bin"] = pd.cut(df[ind], bins)
        agg = df.groupby("bin", observed=False).agg(
            net_pnl=("pnl", "sum"), trades=("pnl", "count")
        ).reset_index()

        fig, ax = plt.subplots()
        bars = ax.bar(range(len(agg)), agg["net_pnl"], width=0.6)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels([str(b) for b in agg["bin"]], rotation=90)
        ax.set_xlabel(ind)
        ax.set_ylabel("Net PnL ($)")

        for rect, count in zip(bars, agg["trades"]):
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h, str(count),
                    ha="center", va="bottom" if h >= 0 else "top")

        fig.tight_layout()
        plt.show(block=False)

    # ── Heatmap ──────────────────────────────────────────────────────────

    def plot_heatmap(self) -> None:
        if self.df is None:
            return

        x, y = self.ind1_var.get(), self.ind2_var.get()
        if not x or not y:
            messagebox.showwarning("Pick indicators", "Choose X and Y.")
            return

        bins_x, bins_y = self.bins1_var.get(), self.bins2_var.get()
        metric = self.metric_var.get()

        required = [x, y, "pnl"]
        if metric == "win_rate":
            if "win" not in self.df.columns:
                messagebox.showerror(
                    "Missing column",
                    "'win' column required for win_rate.",
                )
                return
            required.append("win")

        df = self.df.dropna(subset=required).copy()
        df = self._filter_side(df)
        if df.empty:
            return

        df["bin_x"] = pd.cut(df[x], bins_x)
        df["bin_y"] = pd.cut(df[y], bins_y)

        if metric == "net_pnl":
            pivot_val = df.pivot_table(
                index="bin_x", columns="bin_y",
                values="pnl", aggfunc="sum",
            )
        else:
            df["_win_flag"] = df["win"].astype(int)
            pivot_val = df.pivot_table(
                index="bin_x", columns="bin_y",
                values="_win_flag", aggfunc="mean",
            )

        pivot_cnt = df.pivot_table(
            index="bin_x", columns="bin_y",
            values="pnl", aggfunc="count",
        )

        mask = pivot_cnt.isna() | (pivot_cnt == 0)

        fig, ax = plt.subplots()
        cax = ax.imshow(
            pivot_val.where(~mask).to_numpy(),
            aspect="auto",
            cmap="RdYlGn",
        )
        fig.colorbar(cax, label=metric.replace("_", " ").title())

        ax.set_xticks(range(len(pivot_val.columns)))
        ax.set_xticklabels([str(c) for c in pivot_val.columns], rotation=90)
        ax.set_yticks(range(len(pivot_val.index)))
        ax.set_yticklabels([str(r) for r in pivot_val.index])
        ax.set_xlabel(y)
        ax.set_ylabel(x)
        ax.set_title(f"Heatmap \u2014 {metric} (bin count overlaid)")

        for i in range(pivot_val.shape[0]):
            for j in range(pivot_val.shape[1]):
                cnt = pivot_cnt.iat[i, j]
                if pd.isna(cnt) or cnt == 0:
                    continue
                ax.text(
                    j, i, str(int(cnt)),
                    ha="center", va="center",
                    fontsize=8, color="white",
                )

        fig.tight_layout()
        plt.show(block=False)

    def run(self) -> None:
        self.master.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = BacktestVisualizer(root)
    app.run()
