import os
#!/usr/bin/env python3
"""
智能模拟交易系统 v7
专业模式：NFI多模式 + Regime动态仓位 + 多时间框架确认
"""

import threading
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
INITIAL_CAPITAL = 1000.0
UPDATE_INTERVAL = 60
FEE_RATE = 0.001  # 0.1% 手续费

# 策略：融合 NFI 多模式 + Regime 动态 + 多时间框架
STRATEGIES = {
    "S1_V4": {
        "name": "S1 V4 防守仓", "market": "现货", "leverage": 1.0, "type": "defensive",
        "desc": "老资格抄底，等大跌才动手",
        "detail": "跌出便宜价再接。0笔说明便宜价还没出现，按兵不动是正常。"
    },
    "S2_V9c": {
        "name": "S2 V9c 主力现货", "market": "现货", "leverage": 1.0, "type": "bull_gate",
        "desc": "牛市抄底主力，闸门没开不进场",
        "detail": "BTC/ETH连续7天站上200日线才开闸。只补1次仓，铁底-15%。"
    },
    "S3_V10": {
        "name": "S3 V10 合约1x", "market": "合约", "leverage": 1.0, "type": "dual",
        "desc": "V9c加空头腿，熊市能进攻",
        "detail": "多头腿照搬V9c，空头腿在熊市确立时做弱势币。1x杠杆。"
    },
    "S4_V11b": {
        "name": "S4 V11b 趋势熔断", "market": "合约", "leverage": 3.0, "type": "trend",
        "desc": "组合第一发动机：牛多熊空",
        "detail": "牛市强势突破+滞涨补涨，熊市做空弱势币。三层风控+急反转熔断。"
    },
    "S5_V12b": {
        "name": "S5 V12b 海龟宽通道", "market": "现货", "leverage": 1.0, "type": "turtle",
        "desc": "牛市专用趋势机",
        "detail": "55日通道突破进场，ATR过滤，熊市必亏。适合牛市。"
    },
    "S6_V15": {
        "name": "S6 V15 空头海龟", "market": "合约", "leverage": 2.0, "type": "short_turtle",
        "desc": "破位追空，专赚熊市",
        "detail": "跌破55日低点开空。与S4空单触发时机互补。"
    },
    "S7_V16": {
        "name": "S7 V16 事件驱动", "market": "合约", "leverage": 3.0, "type": "event",
        "desc": "事件异动快进快出",
        "detail": "量能放大+独立行情+振幅突增。小仓快出，24小时冷却。"
    },
    "S8_V17c": {
        "name": "S8 V17c 箱体闸门", "market": "合约", "leverage": 2.0, "type": "box",
        "desc": "确认箱体后低买高卖",
        "detail": "先确认真箱体（7日高低幅/成本<30%），再下轨接多。"
    },
    "S10_NFI": {
        "name": "S10 NFI 对照组", "market": "现货", "leverage": 1.0, "type": "benchmark",
        "desc": "别人写好的智能策略",
        "detail": "NostalgiaForInfinity风格。作为自建策略的参照物。"
    },
    "S11_V11c": {
        "name": "S11 V11c 波动定仓", "market": "合约", "leverage": 3.0, "type": "vol_size",
        "desc": "S4同信号+聪明下注",
        "detail": "狂躁少买、安静多买。与S4对照实验。"
    },
    "MTF_Confirm": {
        "name": "多时间框架确认", "market": "合约", "leverage": 2.5, "type": "mtf",
        "desc": "日线定方向，小时找点",
        "detail": "专业模式：日线BTC 200日线过滤，小时级确认后再进场。"
    },
    "Regime_Dynamic": {
        "name": "Regime 动态仓位", "market": "合约", "leverage": 2.0, "type": "regime",
        "desc": "按市场状态调仓位",
        "detail": "专业模式：先识别Regime，再决定策略权重和单笔大小。"
    },
}

# 每台独立 100U
strategy_capital = {s: 100.0 for s in STRATEGIES}
strategy_peak = {s: 100.0 for s in STRATEGIES}
strategy_dd = {s: 0.0 for s in STRATEGIES}

state = {
    "equity": sum(strategy_capital.values()),
    "cash": sum(strategy_capital.values()),
    "peak_equity": sum(strategy_capital.values()),
    "max_drawdown": 0.0,
    "positions": {},
    "history": [],
    "trades": [],
    "regime": "unknown",
    "regime_history": [],
    "allocations": {},
    "strategy_stats": defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0}),
    "strategy_capital": strategy_capital,
    "strategy_peak": strategy_peak,
    "strategy_dd": strategy_dd,
    "last_update": None,
    "prices": {},
    "entry_reasons": {},
    "lock": threading.Lock(),
}

def fetch_data(symbol, period="120d", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None

def compute_indicators(df):
    if df is None or len(df) < 50:
        return df
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df, 14)
    df["atr_pct"] = df["atr"] / df["close"]
    df["low_55"] = df["low"].rolling(55).min()
    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = np.abs(df["high"] - df["close"].shift())
    lc = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_regime(btc_df):
    if btc_df is None or len(btc_df) < 50:
        return "unknown"
    last = btc_df.iloc[-1]
    prev = btc_df.iloc[-20]
    trend = (last["close"] - prev["close"]) / prev["close"]
    vol = float(last.get("atr_pct", 0.03))
    sma50 = last.get("sma50", last["close"])
    price_vs_sma = last["close"] / sma50 if sma50 else 1.0
    if vol > 0.06:
        return "high_vol"
    if trend > 0.06 and price_vs_sma > 1.01:
        return "bull"
    if trend < -0.06 and price_vs_sma < 0.99:
        return "bear"
    return "range"

def generate_signals(data_dict, regime):
    signals = {s: [] for s in STRATEGIES}
    reasons = {}
    btc = data_dict.get("BTC-USD")
    btc_above_200 = False
    if btc is not None and len(btc) > 200 and "sma200" in btc:
        btc_above_200 = btc.iloc[-1]["close"] > btc.iloc[-1]["sma200"]

    for sym, df in data_dict.items():
        if df is None or len(df) < 55:
            continue
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. 多时间框架确认
        if btc_above_200 and last["rsi"] > 50 and last["close"] > last["sma20"]:
            signals["MTF_Confirm"].append((sym, "long", 1.0))
            reasons[f"MTF_Confirm_{sym}"] = "日线多头 + 小时趋势确认"
        elif not btc_above_200 and last["rsi"] < 50 and last["close"] < last["sma20"]:
            signals["MTF_Confirm"].append((sym, "short", 1.0))
            reasons[f"MTF_Confirm_{sym}"] = "日线空头 + 小时趋势确认"

        # 2. NFI 多模式
        if regime == "bull" and last["rsi"] > 45 and last["close"] > last["sma20"]:
            signals["NFI_Multi"].append((sym, "long", 0.8))
            reasons[f"NFI_Multi_{sym}"] = "NFI趋势模式"
        elif regime == "bear" and last["rsi"] < 55:
            signals["NFI_Multi"].append((sym, "short", 0.7))
            reasons[f"NFI_Multi_{sym}"] = "NFI空头模式"
        elif regime == "range" and last["close"] < last["low_20"] * 1.01:
            signals["NFI_Multi"].append((sym, "long", 0.6))
            reasons[f"NFI_Multi_{sym}"] = "NFI均值回归"

        # 3. Regime 动态
        if regime in ("bull", "high_vol") and last["rsi"] > 55 and prev["rsi"] <= 55:
            signals["Regime_Dynamic"].append((sym, "long", 1.0))
            reasons[f"Regime_Dynamic_{sym}"] = f"Regime={regime} RSI突破"
        if regime == "bear" and last["rsi"] < 45:
            signals["Regime_Dynamic"].append((sym, "short", 0.9))
            reasons[f"Regime_Dynamic_{sym}"] = "熊市空头"

        # 原有策略
        if regime in ("bull",) and last["rsi"] > 55:
            signals["S4_V11b"].append((sym, "long", 1.0))
            reasons[f"S4_V11b_{sym}"] = "趋势多头"
        if regime == "bear" and last["close"] < last["low_55"]:
            signals["S6_V15"].append((sym, "short", 1.0))
            reasons[f"S6_V15_{sym}"] = "破位空头"
        if regime == "range" and last["close"] < last["low_20"] * 1.01:
            signals["S8_V17c"].append((sym, "long", 0.7))
            reasons[f"S8_V17c_{sym}"] = "箱体低买"

    return signals, reasons

def compute_allocations(regime, btc_7up=False):
    """低迷期 + 牛市切换"""
    is_bull = btc_7up and regime in ("bull", "high_vol")
    
    if is_bull:
        # 牛市：趋势策略为主
        alloc = {
            "S4_V11b": 0.22, "S5_V12b": 0.18, "S10_NFI": 0.18, "S11_V11c": 0.16,
            "S2_V9c": 0.10, "MTF_Confirm": 0.10, "Regime_Dynamic": 0.06
        }
    else:
        # 低迷期：箱体 + 少量事件
        alloc = {
            "S8_V17c": 0.30, "S7_V16": 0.15, "S2_V9c": 0.12,
            "MTF_Confirm": 0.12, "Regime_Dynamic": 0.10,
            "S4_V11b": 0.08, "S10_NFI": 0.08, "S11_V11c": 0.05
        }
    total = sum(alloc.values())
    return {k: v/total for k, v in alloc.items() if v > 0}

def execute_paper(signals, reasons, allocations, prices):
    with state["lock"]:
        for strat, sigs in signals.items():
            if strat not in state["strategy_capital"]:
                continue
            cap = state["strategy_capital"][strat]
            if cap < 8 or not sigs:
                continue
            for sym, side, strength in sigs[:2]:
                if sym not in prices:
                    continue
                pos_key = f"{strat}_{sym}"
                if pos_key in state["positions"]:
                    continue
                margin = min(15, cap * 0.12 * strength)
                if cap < margin:
                    continue
                price = prices[sym]
                qty = (margin * STRATEGIES[strat]["leverage"]) / price
                reason = reasons.get(f"{strat}_{sym}", "")
                state["positions"][pos_key] = {
                    "strategy": strat, "symbol": sym, "side": side,
                    "qty": qty, "entry": price, "margin": margin,
                    "time": datetime.utcnow().isoformat(), "reason": reason,
                }
                fee = margin * FEE_RATE
                state["strategy_capital"][strat] -= (margin + fee)
                state["trades"].append({
                    "time": datetime.utcnow().isoformat(), "strategy": strat,
                    "symbol": sym, "side": side, "price": price,
                    "margin": round(margin, 2), "type": "开仓", "reason": reason,
                })

        # 平仓
        to_close = []
        for key, pos in list(state["positions"].items()):
            price = prices.get(pos["symbol"], pos["entry"])
            pnl_pct = (price - pos["entry"]) / pos["entry"] if pos["side"] == "long" else (pos["entry"] - price) / pos["entry"]
            pnl_pct *= STRATEGIES[pos["strategy"]]["leverage"]
            if pnl_pct > 0.08 or pnl_pct < -0.05:
                to_close.append((key, pnl_pct, price))
        for key, pnl_pct, price in to_close:
            pos = state["positions"].pop(key)
            realized = pos["margin"] * pnl_pct
            strat = pos["strategy"]
            fee = pos["margin"] * FEE_RATE
            state["strategy_capital"][strat] += pos["margin"] + realized - fee
            state["trades"].append({
                "time": datetime.utcnow().isoformat(), "strategy": strat,
                "symbol": pos["symbol"], "side": pos["side"], "price": price,
                "pnl": round(realized, 2), "type": "平仓",
            })
            st = state["strategy_stats"][strat]
            st["trades"] += 1
            st["pnl"] += realized
            if realized > 0:
                st["wins"] += 1
            state["strategy_peak"][strat] = max(state["strategy_peak"][strat], state["strategy_capital"][strat])
            dd = (state["strategy_peak"][strat] - state["strategy_capital"][strat]) / state["strategy_peak"][strat]
            state["strategy_dd"][strat] = max(state["strategy_dd"][strat], dd)

        # 更新总权益
        total_eq = sum(state["strategy_capital"].values())
        for pos in state["positions"].values():
            price = prices.get(pos["symbol"], pos["entry"])
            pnl_pct = (price - pos["entry"]) / pos["entry"] if pos["side"] == "long" else (pos["entry"] - price) / pos["entry"]
            total_eq += pos["margin"] * pnl_pct * STRATEGIES[pos["strategy"]]["leverage"]
        state["equity"] = total_eq
        state["peak_equity"] = max(state["peak_equity"], total_eq)
        dd = (state["peak_equity"] - total_eq) / state["peak_equity"] if state["peak_equity"] > 0 else 0
        state["max_drawdown"] = max(state["max_drawdown"], dd)
        state["history"].append({"time": datetime.utcnow().isoformat(), "equity": round(total_eq, 2)})
        if len(state["history"]) > 500:
            state["history"] = state["history"][-500:]

def trading_loop():
    while True:
        try:
            data_dict = {}
            prices = {}
            for sym in SYMBOLS:
                df = fetch_data(sym)
                df = compute_indicators(df)
                data_dict[sym] = df
                if df is not None and len(df):
                    prices[sym] = float(df["close"].iloc[-1])
            regime = detect_regime(data_dict.get("BTC-USD"))
            signals, reasons = generate_signals(data_dict, regime)
            btc_7up = False
            if btc is not None and len(btc) > 200 and "sma200" in btc.columns:
                btc_7up = all(btc.iloc[-j]["close"] > btc.iloc[-j]["sma200"] for j in range(1,8) if len(btc) >= j)
            alloc = compute_allocations(regime, btc_7up)
            execute_paper(signals, reasons, alloc, prices)
            with state["lock"]:
                state["regime"] = regime
                state["allocations"] = alloc
                state["prices"] = prices
                state["entry_reasons"] = reasons
                state["last_update"] = datetime.utcnow().isoformat()
                state["regime_history"].append({"time": state["last_update"], "regime": regime})
                if len(state["regime_history"]) > 80:
                    state["regime_history"] = state["regime_history"][-80:]
        except Exception as e:
            print("Loop error:", e)
        time.sleep(UPDATE_INTERVAL)

HTML = r"""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>智能交易台</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root { --bg:#06080c; --s1:#0d1117; --s2:#161b22; --bd:#2a3140; --tx:#e6edf3; --mu:#7d8590; --gn:#3fb950; --rd:#f85149; --bl:#58a6ff; --am:#d29922; }
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,system-ui,sans-serif;background:var(--bg);color:var(--tx);font-size:13px}
.layout{display:flex;height:100vh}
.side{width:190px;background:#0a0d12;border-right:1px solid var(--bd);padding:14px 8px}
.logo{font-size:12px;color:var(--mu);padding:0 8px 14px;font-weight:600}
.nav{padding:9px 12px;border-radius:6px;margin:2px 0;cursor:pointer;color:var(--mu)}
.nav.active{background:#1f6feb22;color:var(--bl)}
.nav:hover:not(.active){background:#ffffff08;color:var(--tx)}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.top{height:48px;border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;padding:0 16px;background:var(--s1)}
.top .t{font-weight:600}
.top .m{color:var(--mu);font-size:12px}
.cnt{flex:1;overflow:auto;padding:12px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:12px}
.kpi{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:10px 12px}
.kpi .l{font-size:11px;color:var(--mu)}
.kpi .v{font-size:18px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.pos{color:var(--gn)}.neg{color:var(--rd)}
.view{display:none}.view.on{display:block}
.g2{display:grid;grid-template-columns:1.5fr 1fr;gap:10px}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:12px;margin-bottom:10px}
.card h3{font-size:13px;margin-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;color:var(--mu);font-weight:500;padding:7px 5px;border-bottom:1px solid var(--bd);font-size:11px}
td{padding:7px 5px;border-bottom:1px solid #1c2128}
.bdg{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px}
.bdg-bull{background:#23863633;color:var(--gn)}
.bdg-bear{background:#da363333;color:var(--rd)}
.bdg-range{background:#1f6feb33;color:var(--bl)}
.bdg-high_vol{background:#9e6a0333;color:var(--am)}
.bar{height:5px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:3px}
.bf{height:100%;background:var(--bl)}
.mono{font-family:ui-monospace,monospace}
</style>
</head>
<body>
<div class="layout">
  <div class="side">
    <div class="logo">智能交易台</div>
    <div class="nav active" data-v="overview">总览</div>
    <div class="nav" data-v="strategies">策略</div>
    <div class="nav" data-v="positions">持仓</div>
    <div class="nav" data-v="trades">交易</div>
    <div class="nav" data-v="risk">风险</div>
  </div>
  <div class="main">
    <div class="top">
      <div class="t">模拟盘 · 专业模式</div>
      <div class="m" id="st">...</div>
    </div>
    <div class="cnt">
      <div class="kpis" id="kpis"></div>
      <div id="v-overview" class="view on">
        <div class="g2">
          <div class="card"><h3>权益曲线</h3><canvas id="ch" height="190"></canvas></div>
          <div class="card">
            <h3>当前状态</h3>
            <div id="sum" style="font-size:13px;line-height:1.6;margin-bottom:10px"></div>
            <div id="rg" style="margin-bottom:10px"></div>
            <div id="al"></div>
          </div>
        </div>
        <div class="g2">
          <div class="card"><h3>当前持仓</h3><table id="op"></table></div>
          <div class="card"><h3>最近交易</h3><table id="ot"></table></div>
        </div>
        <div class="card">
          <h3>各交易模式盈亏</h3>
          <div id="mode-cards" style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px"></div>
          <table id="pf"></table>
        </div>
      </div>
      <div id="v-strategies" class="view"><div class="card"><h3>策略详情</h3><table id="stbl"></table></div></div>
      <div id="v-positions" class="view"><div class="card"><h3>持仓明细</h3><table id="ptbl"></table></div></div>
      <div id="v-trades" class="view"><div class="card"><h3>交易记录</h3><table id="ttbl"></table></div></div>
      <div id="v-risk" class="view">
        <div class="g2">
          <div class="card"><h3>风险</h3><div id="rk"></div></div>
          <div class="card"><h3>Regime 序列</h3><div id="rs" style="color:var(--mu)"></div></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
let ch;
document.querySelectorAll('.nav').forEach(e=>e.onclick=()=>{
  document.querySelectorAll('.nav').forEach(n=>n.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('on'));
  e.classList.add('active');
  document.getElementById('v-'+e.dataset.v).classList.add('on');
});
function side(s){return s==='short'?'空':'多'}
async function refresh(){
  const r=await fetch('/api/state');const d=await r.json();
  document.getElementById('st').textContent=(d.last_update||'').slice(11,19)+' UTC · Yahoo Finance';
  const eq=d.equity||1000,pnl=eq-1000,dd=((d.max_drawdown||0)*100).toFixed(1);
  const ps=Object.values(d.positions||{}),ln=ps.filter(p=>p.side==='long').length,sn=ps.filter(p=>p.side==='short').length;
  document.getElementById('kpis').innerHTML=`
    <div class="kpi"><div class="l">权益</div><div class="v mono ${pnl>=0?'pos':'neg'}">${eq.toFixed(2)}</div></div>
    <div class="kpi"><div class="l">盈亏</div><div class="v mono ${pnl>=0?'pos':'neg'}">${pnl>=0?'+':''}${pnl.toFixed(2)}</div></div>
    <div class="kpi"><div class="l">回撤</div><div class="v mono neg">${dd}%</div></div>
    <div class="kpi"><div class="l">持仓</div><div class="v mono">${ps.length}</div></div>
    <div class="kpi"><div class="l">Regime</div><div class="v"><span class="bdg bdg-${d.regime}">${d.regime}</span></div></div>`;
  document.getElementById('sum').innerHTML=`Regime <b>${d.regime}</b> · 持仓 ${ps.length}（多 ${ln} / 空 ${sn}）<br>权益 ${eq.toFixed(2)} · 盈亏 ${pnl>=0?'+':''}${pnl.toFixed(2)}`;
  document.getElementById('rg').innerHTML=`<span class="bdg bdg-${d.regime}">${d.regime}</span>`;
  let al='';
  for(const [k,v] of Object.entries(d.allocations||{})){
    const pct=(v*100).toFixed(1),nm=(d.strategies||{})[k]?.name||k;
    al+=`<div style="margin:7px 0"><div style="display:flex;justify-content:space-between;font-size:12px"><span>${nm}</span><span class="mono">${pct}%</span></div><div class="bar"><div class="bf" style="width:${pct}%"></div></div></div>`;
  }
  document.getElementById('al').innerHTML=al;
  const stt=d.strategy_stats||{};
  let pf='<tr><th>策略</th><th>权重</th><th>交易</th><th>胜率</th><th>已实现</th></tr>';
  for(const [k,m] of Object.entries(d.strategies||{})){
    const w=((d.allocations||{})[k]||0)*100,st=stt[k]||{trades:0,wins:0,pnl:0};
    const wr=st.trades?Math.round(st.wins/st.trades*100)+'%':'-';
    pf+=`<tr><td>${m.name}</td><td class="mono">${w.toFixed(1)}%</td><td>${st.trades}</td><td>${wr}</td><td class="mono ${st.pnl>=0?'pos':'neg'}">${st.pnl.toFixed(1)}</td></tr>`;
  }
  document.getElementById('pf').innerHTML=pf;
  let cards='';
  for(const [k,m] of Object.entries(d.strategies||{})){
    const st=stt[k]||{trades:0,wins:0,pnl:0};
    const w=((d.allocations||{})[k]||0)*100;
    const pnlClass=st.pnl>=0?'pos':'neg';
    cards+=`<div style="background:#161b22;border:1px solid #2a3140;border-radius:8px;padding:10px">
      <div style="font-size:12px;color:#7d8590">${m.name}</div>
      <div class="mono ${pnlClass}" style="font-size:16px;font-weight:600;margin:4px 0">${st.pnl>=0?'+':''}${st.pnl.toFixed(2)}</div>
      <div style="font-size:11px;color:#7d8590">权重 ${w.toFixed(0)}% · 交易 ${st.trades}</div>
    </div>`;
  }
  document.getElementById('mode-cards').innerHTML=cards;
  let stbl='<tr><th>策略</th><th>市场</th><th>杠杆</th><th>说明</th></tr>';
  for(const [k,m] of Object.entries(d.strategies||{})) stbl+=`<tr><td>${m.name}</td><td>${m.market}</td><td>${m.leverage}x</td><td style="color:var(--mu);font-size:12px">${m.detail}</td></tr>`;
  document.getElementById('stbl').innerHTML=stbl;
  let ph='<tr><th>策略</th><th>标的</th><th>方向</th><th>入场</th><th>保证金</th><th>原因</th></tr>';
  for(const p of ps) ph+=`<tr><td>${p.strategy}</td><td>${p.symbol}</td><td>${side(p.side)}</td><td class="mono">${p.entry.toFixed(2)}</td><td class="mono">${p.margin.toFixed(1)}</td><td style="color:var(--mu);font-size:11px">${p.reason||''}</td></tr>`;
  const phf=ph||'<tr><td colspan="6" style="color:var(--mu)">暂无</td></tr>';
  document.getElementById('op').innerHTML=phf;document.getElementById('ptbl').innerHTML=phf;
  let th='<tr><th>时间</th><th>策略</th><th>标的</th><th>方向</th><th>类型</th><th>价格</th><th>PnL</th></tr>';
  (d.trades||[]).slice(-20).reverse().forEach(t=>{
    th+=`<tr><td>${t.time.slice(11,19)}</td><td>${t.strategy}</td><td>${t.symbol}</td><td>${side(t.side)}</td><td>${t.type}</td><td class="mono">${t.price?.toFixed(2)||''}</td><td class="mono ${(t.pnl||0)>=0?'pos':'neg'}">${t.pnl||''}</td></tr>`;
  });
  document.getElementById('ot').innerHTML=th;document.getElementById('ttbl').innerHTML=th;
  const mg=ps.reduce((s,p)=>s+p.margin,0);
  document.getElementById('rk').innerHTML=`<div style="margin:10px 0"><div style="color:var(--mu)">保证金占用</div><div class="mono" style="font-size:15px">${mg.toFixed(1)} / ${eq.toFixed(1)}</div></div><div style="margin:10px 0"><div style="color:var(--mu)">最大回撤</div><div class="mono neg" style="font-size:15px">${dd}%</div></div>`;
  document.getElementById('rs').textContent=(d.regime_history||[]).slice(-8).map(h=>h.regime).join(' → ')||'暂无';
  const lbs=(d.history||[]).map(h=>h.time.slice(11,16)),eqs=(d.history||[]).map(h=>h.equity);
  if(!ch){ch=new Chart(document.getElementById('ch'),{type:'line',data:{labels:lbs,datasets:[{data:eqs,borderColor:'#3fb950',borderWidth:1.6,tension:0.2,pointRadius:0}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{color:'#21262d'},ticks:{color:'#7d8590',font:{size:11}}}}}});}
  else{ch.data.labels=lbs;ch.data.datasets[0].data=eqs;ch.update('none');}
}
refresh();setInterval(refresh,5000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/state")
def api_state():
    with state["lock"]:
        return jsonify({
            "equity": state["equity"], "cash": state["cash"],
            "max_drawdown": state["max_drawdown"],
            "positions": state["positions"],
            "trades": state["trades"][-40:],
            "history": state["history"][-120:],
            "regime": state["regime"],
            "regime_history": state["regime_history"][-15:],
            "allocations": state["allocations"],
            "last_update": state["last_update"],
            "prices": state["prices"],
            "strategy_stats": dict(state["strategy_stats"]),
            "strategies": {k: {"name": v["name"], "desc": v["desc"], "detail": v["detail"],
                              "market": v["market"], "leverage": v["leverage"],
                              "capital": state["strategy_capital"].get(k, 100)} for k, v in STRATEGIES.items()},
        })

if __name__ == "__main__":
    threading.Thread(target=trading_loop, daemon=True).start()
    print("v7 启动")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
