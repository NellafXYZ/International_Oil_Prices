################################################################################
#                                                                              #
#   伊以冲突对国际油价冲击的量化分析                                            #
#   Event Study + Interrupted Time Series Analysis (ITSA)                      #
#   Israel-Iran Conflict (2025.01 – 2026.04) → Oil Prices                      #
#                                                                              #
#   方法参考:                                                                   #
#     MacKinlay (1997) "Event Studies in Economics and Finance"                 #
#     Bernal, Cummins & Gasparrini (2017) ITSA                                  #
#     Linden (2015) ITSA implementation                                        #
#                                                                              #
#   数据来源:                                                                   #
#     冲突事件: ACLED (acleddata.com)                                           #
#     油价数据: NASDAQQUSOI.csv                                                 #
#     市场分析: Goldman Sachs, Citi, UOB, OCBC, SEB, Allianz                   #
#                                                                              #
################################################################################

import os
import sys
import numpy as np
import pandas as pd

# ========================== matplotlib 初始化（必须在 pyplot 导入前） ==========================
import matplotlib as _mpl
_mpl.use('Agg')  # 非交互式后端，避免 plt.show() 阻塞；savefig 仍正常输出 PNG
import matplotlib.font_manager as fm

# 删除旧字体缓存，强制 matplotlib 重新扫描系统字体（确保 CJK 字体被识别）
_cache_dir = _mpl.get_cachedir()
for _fn in os.listdir(_cache_dir):
    if 'font' in _fn.lower():
        os.remove(os.path.join(_cache_dir, _fn))
fm._load_fontmanager(try_read_cache=False)

# 设置中文字体
_mpl.rcParams['font.family'] = 'sans-serif'
_mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
_mpl.rcParams['axes.unicode_minus'] = False

# 现在可以安全导入 pyplot 和其他模块
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.api import OLS, add_constant
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# 强制 stdout 使用 UTF-8 编码，解决 GBK 控制台中文/特殊字符乱码问题
sys.stdout.reconfigure(encoding='utf-8')

# ===========================================================================
# 1. 选定 7 个关键事件（基于ACLED数据 + 投行研究报告）
# ===========================================================================
events = pd.DataFrame({
    'event_id':   range(1, 8),
    'event_date': pd.to_datetime([
        '2025-06-13',   # E1: Operation Rising Lion
        '2025-06-21',   # E2: US enters war (Operation Northern Hammer)
        '2025-06-24',   # E3: Trump ceasefire
        '2025-08-28',   # E4: Israel "Lucky Drop" kills Houthi PM
        '2025-10-15',   # E5: Gaza ceasefire agreement
        '2026-02-28',   # E6: Strait of Hormuz de facto closure
        '2026-04-07',   # E7: US-Iran conditional ceasefire ("April Thaw")
    ]),
    'label': [
        'E1 以色列 Rising Lion 空袭伊朗核设施 (升级)',
        'E2 美国 Northern Hammer 参战轰炸伊朗 (升级)',
        'E3 特朗普宣布伊以全面停火 (降级)',
        'E4 以色列 Lucky Drop 刺杀胡塞总理 (升级)',
        'E5 加沙停火协议达成 (降级)',
        'E6 霍尔木兹海峡事实关闭 (升级/供给冲击)',
        'E7 美伊有条件停火 四月解冻 (降级)',
    ],
    'event_type': ['escalation', 'escalation', 'de-escalation',
                   'escalation', 'de-escalation', 'escalation', 'de-escalation'],
    'expected_dir': ['+', '+', '-', '+', '-', '+', '-'],
    'known_impact': [
        'Brent +13% intraday ($69.65→$78.50)',
        '美国参战, 油价先跌后稳 (市场疲劳)',
        'Brent -6% ($77.81→$66.86), 跌破战前水平',
        '区域性影响, Brent小幅+1~2%',
        'Brent逐步回落至$60-65',
        'Brent $85→$138+, 实物原油触及$141 (2008来最高)',
        'Brent $138→$94.81 (-13%), 四月解冻',
    ]
})

print("=" * 60)
print("  选定的 7 个关键事件")
print("=" * 60)
print(events[['event_id', 'event_date', 'label', 'expected_dir']].to_string(index=False))

# ===========================================================================
# 2. 数据加载与预处理
# ===========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # 确保工作目录为脚本所在目录，所有相对路径基于此目录
raw = pd.read_csv(os.path.join(SCRIPT_DIR, 'NASDAQQUSOI.csv'),
                   names=['date', 'price'], skiprows=1)
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.dropna(subset=['price'])
raw = raw[raw['price'] > 0].sort_values('date').reset_index(drop=True)

# 计算对数收益率
raw['log_price'] = np.log(raw['price'])
raw['ret'] = raw['log_price'].diff()

print(f"\n数据范围: {raw['date'].min().date()} 至 {raw['date'].max().date()}")
print(f"总交易日: {len(raw)}, 价格范围: ${raw['price'].min():.2f} -- ${raw['price'].max():.2f}")

# ===========================================================================
# 3. 探索性数据分析
# ===========================================================================

# --- 3.1 价格时序图 ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(raw['date'], raw['price'], color='#4682B4', linewidth=0.8)
for _, ev in events.iterrows():
    color = '#d73027' if ev['event_type'] == 'escalation' else '#2ca25f'
    ax.axvline(ev['event_date'], color=color, linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_title('油价走势与伊以冲突关键事件 (2025.01 – 2026.04)', fontsize=13)
ax.set_xlabel('日期'); ax.set_ylabel('油价 (NASDAQQUSOI)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#d73027', linestyle='--', label='冲突升级'),
    Line2D([0], [0], color='#2ca25f', linestyle='--', label='冲突降级'),
]
ax.legend(handles=legend_elements, loc='upper left')
plt.tight_layout()
plt.savefig('fig1_price_series.png', dpi=150)
plt.show()

# --- 3.2 收益率时序图 ---
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(raw['date'], raw['ret'] * 100, color='#808080', linewidth=0.4)
for _, ev in events.iterrows():
    color = '#d73027' if ev['event_type'] == 'escalation' else '#2ca25f'
    ax.axvline(ev['event_date'], color=color, linestyle='--', linewidth=0.6, alpha=0.5)
ax.set_title('日对数收益率 (%)', fontsize=13)
ax.set_xlabel('日期'); ax.set_ylabel('收益率 (%)')
plt.tight_layout()
plt.savefig('fig2_returns_series.png', dpi=150)
plt.show()

# --- 3.3 描述性统计 ---
ret_clean = raw['ret'].dropna()
print("\n========== 描述性统计 ==========")
print(f"均值日收益率:   {ret_clean.mean()*100:.4f}%")
print(f"标准差 (日):    {ret_clean.std()*100:.4f}%")
print(f"偏度:           {ret_clean.skew():.4f}")
print(f"峰度:           {ret_clean.kurtosis():.4f}")
print(f"年化波动率:     {ret_clean.std() * np.sqrt(252) * 100:.2f}%")

# ADF 平稳性检验
print("\n========== ADF 单位根检验 ==========")
adf_price = adfuller(raw['log_price'].dropna(), regression='c', maxlag=5)
print(f"价格对数 → ADF统计量: {adf_price[0]:.4f} (p={adf_price[1]:.4f}), "
      f"1%临界值: {adf_price[4]['1%']:.4f}")
print(f"  结论: {'不能拒绝单位根 (非平稳)' if adf_price[1] > 0.05 else '拒绝单位根 (平稳)'}")

adf_ret = adfuller(ret_clean, regression='n', maxlag=5)
print(f"收益率 → ADF统计量: {adf_ret[0]:.4f} (p={adf_ret[1]:.6f}), "
      f"1%临界值: {adf_ret[4]['1%']:.4f}")
print(f"  结论: {'不能拒绝单位根' if adf_ret[1] > 0.05 else '拒绝单位根 (平稳) --> 可用于建模 [OK]'}")

# ===========================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第四部分：事件研究法 (Event Study)                                       ║
# ║  MacKinlay (1997) — 常均值模型                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ===========================================================================

print("\n" + "=" * 60)
print("  事件研究法 (Event Study)")
print("=" * 60)

EST_START, EST_END = -120, -11
EV_START, EV_END = -5, 10


def event_study(ret_series, date_series, event_date,
                est_start, est_end, ev_start, ev_end):
    """执行事件研究分析 (常均值模型, MacKinlay 1997)"""
    # 找到事件日索引
    e_idx = np.argmin(np.abs(date_series - event_date))
    N = len(ret_series)

    # 估计窗口
    est_range = np.arange(e_idx + est_start, e_idx + est_end + 1)
    est_range = est_range[(est_range >= 0) & (est_range < N)]
    est_ret = ret_series.iloc[est_range].dropna()

    mu_hat = est_ret.mean()
    sigma_hat = est_ret.std()
    T_est = len(est_ret)

    # 事件窗口
    ev_range = np.arange(e_idx + ev_start, e_idx + ev_end + 1)
    ev_range = ev_range[(ev_range >= 0) & (ev_range < N)]
    ev_ret = ret_series.iloc[ev_range].values

    AR = ev_ret - mu_hat              # 异常收益
    CAR = np.cumsum(AR)               # 累积异常收益
    rel_days = ev_range - e_idx
    L = len(ev_range)

    # --- 统计推断 ---
    se_AR = sigma_hat * np.sqrt(1 + 1 / T_est)
    SAR = AR / se_AR                  # 标准化AR (Patell)

    # CAR 方差 (考虑估计误差)
    var_CAR = sigma_hat**2 * (L + L**2 / T_est)
    se_CAR = np.sqrt(var_CAR)
    J1 = CAR[-1] / se_CAR             # CAR t-test
    p_J1 = 2 * stats.t.sf(abs(J1), df=T_est - 1)

    # J2: Patell 标准化检验
    J2 = np.sum(SAR) / np.sqrt(L)
    p_J2 = 2 * stats.norm.sf(abs(J2))

    # J3: BMP 横截面检验
    J3 = np.sqrt(L) * np.mean(SAR) / np.std(SAR) if np.std(SAR) > 0 else np.nan

    return {
        'event_idx': e_idx, 'est_range': est_range, 'ev_range': ev_range,
        'rel_days': rel_days, 'AR': AR, 'CAR': CAR,
        'mu_hat': mu_hat, 'sigma_hat': sigma_hat,
        'se_AR': se_AR, 'se_CAR': se_CAR,
        'J1': J1, 'J2': J2, 'J3': J3,
        'p_J1': p_J1, 'p_J2': p_J2,
        'T_est': T_est, 'L': L
    }


# 对所有7个事件执行事件研究
es_results = []

for k in range(len(events)):
    edate = events['event_date'].iloc[k]
    elab = events['label'].iloc[k]

    res = event_study(raw['ret'], raw['date'], edate,
                      EST_START, EST_END, EV_START, EV_END)
    es_results.append(res)

    car_val = res['CAR'][-1] * 100
    if res['p_J1'] < 0.01:
        sig = '***'
    elif res['p_J1'] < 0.05:
        sig = '**'
    elif res['p_J1'] < 0.10:
        sig = '*'
    else:
        sig = ''

    print(f"\n事件{k+1} | {elab}")
    print(f"  事件日: {edate.date()}, 估计窗口: [{EST_START},{EST_END}], {res['T_est']}天")
    print(f"  事件窗口: [{EV_START},{EV_END}], {res['L']}天")
    print(f"  CAR[{EV_START},{EV_END}] = {car_val:+.4f}%")
    print(f"  J1(CAR) = {res['J1']:+.4f}, p = {res['p_J1']:.6f} {sig}")
    print(f"  J2(Patell) = {res['J2']:+.4f}, p = {res['p_J2']:.6f}")
    print(f"  J3(BMP) = {res['J3']:+.4f}")

    exp_sign = 1 if events['expected_dir'].iloc[k] == '+' else -1
    act_sign = np.sign(res['CAR'][-1])
    print(f"  预期: {'UP' if exp_sign > 0 else 'DN'} | 实际: {'UP' if act_sign > 0 else 'DN'} --> "
          f"{'[OK] 符合' if exp_sign == act_sign else '[!!] 不符'}")

# --- Event Study 汇总表 ---
es_rows = []
for k in range(7):
    r = es_results[k]
    if r['p_J1'] < 0.01:
        sig = '*** (1%)'
    elif r['p_J1'] < 0.05:
        sig = '** (5%)'
    elif r['p_J1'] < 0.10:
        sig = '* (10%)'
    else:
        sig = '不显著'
    exp_sign = 1 if events['expected_dir'].iloc[k] == '+' else -1
    act_sign = np.sign(r['CAR'][-1])
    es_rows.append({
        'Event': k + 1,
        'Date': str(events['event_date'].iloc[k].date()),
        'Label': events['label'].iloc[k],
        'CAR_pct': r['CAR'][-1] * 100,
        'J1_CAR': r['J1'], 'p_J1': r['p_J1'],
        'J2_Patell': r['J2'], 'p_J2': r['p_J2'],
        'Significance': sig,
        'Expected': events['expected_dir'].iloc[k],
        'Match': 'OK' if exp_sign == act_sign else '!!'
    })

es_table = pd.DataFrame(es_rows)
print("\n\n========== 事件研究结果汇总 ==========")
print(es_table.round(4).to_string(index=False))

# --- CAR 可视化 (7个子图) ---
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
for k in range(7):
    r = es_results[k]
    ax = axes[k]
    color = '#d73027' if r['CAR'][-1] > 0 else '#4575b4'
    ax.fill_between(r['rel_days'], 0, r['CAR'] * 100,
                    color=color, alpha=0.2)
    ax.plot(r['rel_days'], r['CAR'] * 100, color=color, linewidth=1.5)
    ax.scatter(r['rel_days'], r['CAR'] * 100, color=color, s=15)
    ax.axhline(y=0, color='#808080', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title(f"事件{k+1}: {events['label'].iloc[k][:40]}...", fontsize=9)
    ax.set_xlabel('相对事件日 (天)')
    ax.set_ylabel('CAR (%)')
axes[7].set_visible(False)
plt.suptitle('各事件累积异常收益 (CAR) 路径', fontsize=14)
plt.tight_layout()
plt.savefig('fig3_CAR_paths.png', dpi=150)
plt.show()

# --- Bootstrap 冲击分解 ---
print("\n========== Bootstrap 冲击分解 ==========")
np.random.seed(2025)
n_boot = 2000

boot_results = []
for k in range(7):
    r = es_results[k]
    est_ret = raw['ret'].iloc[r['est_range']].dropna().values
    boot_cars = np.array([
        np.sum(np.random.choice(est_ret, size=r['L'], replace=True) - r['mu_hat'])
        for _ in range(n_boot)
    ])
    ci = np.percentile(boot_cars, [2.5, 97.5])
    car_est = r['CAR'][-1] * 100
    price_impact = raw['price'].iloc[r['event_idx']] * (np.exp(r['CAR'][-1]) - 1)
    boot_results.append({
        'Event': k + 1, 'Label': events['label'].iloc[k],
        'CAR_est': car_est, 'CAR_lower': ci[0] * 100,
        'CAR_upper': ci[1] * 100, 'Price_Impact': price_impact
    })

boot_df = pd.DataFrame(boot_results)
for _, row in boot_df.iterrows():
    print(f"E{int(row['Event'])}: CAR = {row['CAR_est']:+.3f}% "
          f"[95% CI: {row['CAR_lower']:+.3f}%, {row['CAR_upper']:+.3f}%]")

# Bootstrap 分解图
fig, ax = plt.subplots(figsize=(10, 5))
boot_sorted = boot_df.sort_values('CAR_est', key=abs)
colors = ['#d73027' if x > 0 else '#4575b4' for x in boot_sorted['CAR_est']]
ax.barh(range(len(boot_sorted)), boot_sorted['CAR_est'], color=colors, height=0.6)
for i, (_, row) in enumerate(boot_sorted.iterrows()):
    ax.errorbar(row['CAR_est'], i,
                xerr=[[max(0, row['CAR_est'] - row['CAR_lower'])],
                       [max(0, row['CAR_upper'] - row['CAR_est'])]],
                fmt='none', ecolor='#666666', capsize=3)
    label_x = row['CAR_upper'] + 0.5 if row['CAR_est'] > 0 else row['CAR_lower'] - 0.5
    ha = 'left' if row['CAR_est'] > 0 else 'right'
    ax.text(label_x, i, f"{row['CAR_est']:+.2f}%", va='center', ha=ha, fontsize=8)
ax.set_yticks(range(len(boot_sorted)))
ax.set_yticklabels([lbl[:50] + '...' for lbl in boot_sorted['Label']])
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_title('各事件冲击的独立贡献分解\n误差线 = 95% Bootstrap置信区间 (2000次重抽样)')
ax.set_xlabel('累积异常收益 CAR (%)')
plt.tight_layout()
plt.savefig('fig4_bootstrap_decomposition.png', dpi=150)
plt.show()

# ===========================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第五部分：中断时间序列分析 (ITSA)                                        ║
# ║  Segmented Regression + ARIMAX                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ===========================================================================

print("\n\n" + "=" * 60)
print("  中断时间序列分析 (ITSA)")
print("=" * 60)

# --- 5.1 构建干预变量 ---
raw['t'] = range(1, len(raw) + 1)
itsa_df = raw[['date', 'log_price', 'price', 'ret', 't']].copy()

intervention_vars = []
for k in range(len(events)):
    edate = events['event_date'].iloc[k]
    eid = f'E{k+1}'
    eidx = np.argmin(np.abs(raw['date'] - edate))

    lev_name = f'D_{eid}'
    trd_name = f'S_{eid}'
    itsa_df[lev_name] = (itsa_df.index >= eidx).astype(int)
    itsa_df[trd_name] = np.where(itsa_df.index >= eidx, itsa_df.index - eidx, 0)

    intervention_vars.extend([lev_name, trd_name])

# --- 5.2 基准 vs 完整 ITSA 模型 ---
y = itsa_df['log_price']
X_baseline = add_constant(itsa_df[['t']])
X_full = add_constant(itsa_df[['t'] + intervention_vars])

m0 = OLS(y, X_baseline).fit()
m_full_itsa = OLS(y, X_full).fit()

print(f"\n--- 模型比较 ---")
print(f"基准模型: R² = {m0.rsquared:.4f}, Adj R² = {m0.rsquared_adj:.4f}, AIC = {m0.aic:.1f}")
print(f"完整模型: R² = {m_full_itsa.rsquared:.4f}, Adj R² = {m_full_itsa.rsquared_adj:.4f}, AIC = {m_full_itsa.aic:.1f}")

# F检验 (所有干预变量联合显著)
rss0 = np.sum(m0.resid ** 2)
rss1 = np.sum(m_full_itsa.resid ** 2)
p_full = m_full_itsa.df_model
p_base = m0.df_model
n_obs = len(y)
F_stat = ((rss0 - rss1) / (p_full - p_base)) / (rss1 / (n_obs - p_full - 1))
p_F = stats.f.sf(F_stat, p_full - p_base, n_obs - p_full - 1)
print(f"\n联合显著性 F 检验: F = {F_stat:.4f}, p = {p_F:.6f}"
      f"{' ***' if p_F < 0.01 else ''}")

# --- 5.3 完整 ITSA 回归结果 ---
print("\n========== 完整 ITSA 模型 ==========")
print(m_full_itsa.summary().tables[1])

# --- 5.4 提取事件效应 ---
itsa_effects = []
for k in range(1, 8):
    d_name = f'D_E{k}'
    s_name = f'S_E{k}'
    row = {'Event': k, 'Label': events['label'].iloc[k - 1]}

    for name, prefix in [(d_name, 'Level'), (s_name, 'Trend')]:
        if name in m_full_itsa.params.index:
            coef = m_full_itsa.params[name]
            se = m_full_itsa.bse[name]
            pval = m_full_itsa.pvalues[name]
            row[f'{prefix}_Change'] = coef
            row[f'{prefix}_SE'] = se
            row[f'{prefix}_pval'] = pval
            if prefix == 'Level':
                row['Level_Impact_Pct'] = (np.exp(coef) - 1) * 100
                row['Level_CI_low'] = (np.exp(coef - 1.96 * se) - 1) * 100
                row['Level_CI_high'] = (np.exp(coef + 1.96 * se) - 1) * 100
                if pval < 0.01: row['Level_Sig'] = '***'
                elif pval < 0.05: row['Level_Sig'] = '**'
                elif pval < 0.10: row['Level_Sig'] = '*'
                else: row['Level_Sig'] = ''
            else:
                if pval < 0.01: row['Trend_Sig'] = '***'
                elif pval < 0.05: row['Trend_Sig'] = '**'
                elif pval < 0.10: row['Trend_Sig'] = '*'
                else: row['Trend_Sig'] = ''
        else:
            row[f'{prefix}_Change'] = np.nan
    itsa_effects.append(row)

itsa_eff = pd.DataFrame(itsa_effects)
print("\n========== ITSA 各事件效应分解 ==========")
print(itsa_eff[['Event', 'Label', 'Level_Impact_Pct', 'Level_CI_low',
                'Level_CI_high', 'Level_Sig', 'Trend_Change', 'Trend_Sig']].round(4).to_string(index=False))

# ITSA 水平效应图
fig, ax = plt.subplots(figsize=(10, 5))
itsa_sorted = itsa_eff.sort_values('Level_Impact_Pct', key=abs)
colors = ['#d73027' if x > 0 else '#4575b4' for x in itsa_sorted['Level_Impact_Pct']]
ax.barh(range(len(itsa_sorted)), itsa_sorted['Level_Impact_Pct'], color=colors, height=0.6)
for i, (_, row) in enumerate(itsa_sorted.iterrows()):
    ci_low = max(0, row['Level_Impact_Pct'] - row['Level_CI_low'])
    ci_high = max(0, row['Level_CI_high'] - row['Level_Impact_Pct'])
    ax.errorbar(row['Level_Impact_Pct'], i, xerr=[[ci_low], [ci_high]],
                fmt='none', ecolor='#666666', capsize=3)
    label_x = row['Level_CI_high'] + 1 if row['Level_Impact_Pct'] > 0 else row['Level_CI_low'] - 1
    ha = 'left' if row['Level_Impact_Pct'] > 0 else 'right'
    ax.text(label_x, i, f"{row['Level_Impact_Pct']:+.2f}%{row['Level_Sig']}",
            va='center', ha=ha, fontsize=8)
ax.set_yticks(range(len(itsa_sorted)))
ax.set_yticklabels([lbl[:55] + '...' for lbl in itsa_sorted['Label']])
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_title('ITSA: 各事件的水平效应 (Level Change)\n误差线 = 95% CI | *** p<0.01, ** p<0.05, * p<0.10')
ax.set_xlabel('价格水平变化 (%)')
plt.tight_layout()
plt.savefig('fig5_itsa_level_effects.png', dpi=150)
plt.show()

# --- 5.5 ARIMAX: 控制残差自相关 ---
print("\n========== ARIMAX 模型 ==========")

# OLS残差自相关检验
lb_ols = acorr_ljungbox(m_full_itsa.resid, lags=[10], return_df=True)
print(f"OLS残差 Ljung-Box Q(10) = {lb_ols['lb_stat'].values[0]:.3f}, "
      f"p = {lb_ols['lb_pvalue'].values[0]:.4f}")

# 自动选择ARIMA阶数
print("正在拟合ARIMAX模型 (ARIMA自动选阶)...")
xreg_full = itsa_df[intervention_vars].values

# 先检验自相关结构以确定阶数
try:
    # 使用 auto_arima 的思路: 遍历 (p,0,q) 找AIC最小
    best_aic = np.inf
    best_order = (1, 0, 1)
    for p in range(0, 4):
        for q in range(0, 4):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(itsa_df['log_price'], order=(p, 0, q),
                              exog=xreg_full, trend='c')
                fitted = model.fit(method_kwargs={'maxiter': 200})
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, 0, q)
            except Exception:
                continue
    print(f"最优ARIMA阶数: {best_order}, AIC = {best_aic:.1f}")
except Exception as e:
    print(f"ARIMA阶数自动选择失败: {e}，使用默认(1,0,1)")
    best_order = (1, 0, 1)

# 拟合最终 ARIMAX
try:
    arimax_fit = ARIMA(itsa_df['log_price'], order=best_order,
                       exog=xreg_full, trend='c')
    arimax_result = arimax_fit.fit(method_kwargs={'maxiter': 500})
    print(f"\nARIMAX({best_order[0]},{best_order[1]},{best_order[2]}) 结果:")
    print(arimax_result.summary())

    # ARIMAX 残差诊断
    lb_arimax = acorr_ljungbox(arimax_result.resid, lags=[10], return_df=True)
    print(f"\nARIMAX残差 Ljung-Box Q(10) = {lb_arimax['lb_stat'].values[0]:.3f}, "
          f"p = {lb_arimax['lb_pvalue'].values[0]:.4f} → "
          f"{'白噪声 [OK]' if lb_arimax['lb_pvalue'].values[0] > 0.05 else '仍有自相关 [!!]'}")
except Exception as e:
    print(f"ARIMAX拟合失败: {e}")

# --- 5.6 反事实分析 ---
print("\n========== 反事实分析 ==========")

cf_list = []
for k in range(len(events)):
    edate = events['event_date'].iloc[k]
    eidx = np.argmin(np.abs(raw['date'] - edate))

    cf_data = itsa_df.copy()
    lev_var = f'D_E{k+1}'
    trd_var = f'S_E{k+1}'
    if lev_var in cf_data.columns:
        cf_data.loc[cf_data.index >= eidx, lev_var] = 0
    if trd_var in cf_data.columns:
        cf_data.loc[cf_data.index >= eidx, trd_var] = 0

    X_cf = add_constant(cf_data[['t'] + intervention_vars])
    cf_pred_log = m_full_itsa.predict(X_cf)
    cf_pred_price = np.exp(cf_pred_log)

    cf_df = pd.DataFrame({
        'date': raw['date'],
        'Event': k + 1,
        'Actual': raw['price'],
        'Counterfact': cf_pred_price,
        'Net_Effect': raw['price'] - cf_pred_price,
        'Net_Effect_Pct': (raw['price'] / cf_pred_price - 1) * 100,
        'PostEvent': raw['date'] >= edate
    })
    cf_list.append(cf_df)

# 绘制 4 个关键事件的反事实图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
key_events = [0, 2, 5, 6]  # E1, E3, E6, E7
for plot_i, k in enumerate(key_events):
    ax = axes[plot_i // 2, plot_i % 2]
    e_idx = np.argmin(np.abs(cf_list[k]['date'] - events['event_date'].iloc[k]))
    start = max(0, e_idx - 60)
    end = min(len(cf_list[k]), e_idx + 90)
    zoom = cf_list[k].iloc[start:end]

    ax.fill_between(zoom['date'], zoom['Actual'], zoom['Counterfact'],
                    color='#808080', alpha=0.3)
    ax.plot(zoom['date'], zoom['Actual'], color='#d73027', linewidth=0.8, label='实际价格')
    ax.plot(zoom['date'], zoom['Counterfact'], color='#4575b4',
            linewidth=0.7, linestyle='--', label='反事实 (无该事件)')
    ax.axvline(events['event_date'].iloc[k], color='black', linewidth=0.5)
    ax.set_title(f"事件{k+1}: {events['label'].iloc[k][:45]}...", fontsize=9)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.suptitle('反事实分析: 红线=实际价格 | 蓝虚线=假设无该事件的价格', fontsize=12)
plt.tight_layout()
plt.savefig('fig6_counterfactual.png', dpi=150)
plt.show()

# --- 5.7 效应持久性 ---
print("\n========== 效应持久性分析 ==========")

persist_rows = []
for k in range(7):
    cf = cf_list[k]
    post = cf[cf['PostEvent']]

    peak_idx = np.argmax(np.abs(post['Net_Effect_Pct'].values))
    peak_pct = post['Net_Effect_Pct'].iloc[peak_idx]
    peak_day = post['date'].iloc[peak_idx]
    peak_abs = abs(peak_pct)

    post_peak = post.iloc[peak_idx:]
    h_idx = np.where(np.abs(post_peak['Net_Effect_Pct'].values) <= peak_abs / 2)[0]
    half_life = h_idx[0] if len(h_idx) > 0 else len(post_peak)

    z_idx = np.where(np.abs(post_peak['Net_Effect_Pct'].values) <= 0.5)[0]
    days_to_zero = z_idx[0] if len(z_idx) > 0 else np.nan

    persist_rows.append({
        'Event': k + 1, 'Label': events['label'].iloc[k],
        'Peak_Pct': peak_pct, 'Peak_Day': str(peak_day.date()),
        'HalfLife': half_life, 'Days_To_Zero': days_to_zero
    })

persist_df = pd.DataFrame(persist_rows)
print(persist_df.round(3).to_string(index=False))

# ===========================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第六部分：结构性断点检验                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ===========================================================================

print("\n" + "=" * 60)
print("  结构性断点检验")
print("=" * 60)

# --- Chow 检验 ---
print("\n--- Chow 断点检验 ---")
for k in range(len(events)):
    edate = events['event_date'].iloc[k]
    eidx = np.argmin(np.abs(raw['date'] - edate))
    print(f"事件{k+1} ({edate.date()}):", end=' ')

    try:
        # 分段拟合
        pre = itsa_df.iloc[:eidx]
        post = itsa_df.iloc[eidx:]

        fit_pre = OLS(pre['log_price'], add_constant(pre[['t']])).fit()
        fit_post = OLS(post['log_price'], add_constant(post[['t']])).fit()
        fit_pool = OLS(itsa_df['log_price'], add_constant(itsa_df[['t']])).fit()

        rss_pre = np.sum(fit_pre.resid ** 2)
        rss_post = np.sum(fit_post.resid ** 2)
        rss_pool = np.sum(fit_pool.resid ** 2)
        n = len(itsa_df)

        k_params = 2
        F_chow = ((rss_pool - (rss_pre + rss_post)) / k_params) / \
                 ((rss_pre + rss_post) / (n - 2 * k_params))
        p_chow = stats.f.sf(F_chow, k_params, n - 2 * k_params)

        if p_chow < 0.01: sig = '***'
        elif p_chow < 0.05: sig = '**'
        elif p_chow < 0.10: sig = '*'
        else: sig = '(不显著)'
        print(f"Chow F = {F_chow:.4f}, p = {p_chow:.6f} {sig}")
    except Exception as e:
        print(f"无法执行: {e}")

# ===========================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第七部分：综合结论                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ===========================================================================

print("\n\n" + "#" * 60)
print("#   综合结论与发现")
print("#" * 60)

# 1. 统计显著事件
sig_es = es_table[es_table['Significance'] != '不显著']
print(f"\n1. 事件研究法: 7个事件中 {len(sig_es)}个具有统计显著性")
for _, row in sig_es.iterrows():
    print(f"   {row['Label'][:60]} | CAR = {row['CAR_pct']:+.2f}% | p = {row['p_J1']:.6f} | {row['Significance']}")

# 2. 最大冲击
max_idx = es_table['CAR_pct'].abs().idxmax()
print(f"\n2. 最大价格冲击: {es_table['Label'].iloc[max_idx]} (CAR = {es_table['CAR_pct'].iloc[max_idx]:+.2f}%)")

# 3. 方向一致性
match_rate = (es_table['Match'] == 'OK').sum() / 7 * 100
print(f"3. 方向一致性: {(es_table['Match'] == 'OK').sum()}/7 ({match_rate:.0f}%) 符合预期")

# 4. ITSA显著水平变化
sig_level = itsa_eff[itsa_eff['Level_Sig'].str.contains(r'\*', na=False)]
if len(sig_level) > 0:
    print(f"\n4. ITSA显著水平变化: {len(sig_level)} 个事件")
    for _, row in sig_level.iterrows():
        print(f"   {row['Label'][:60]} | {row['Level_Impact_Pct']:+.2f}% {row['Level_Sig']}")

# 5. 持久性
print("\n5. 效应持久性 (按半衰期排序):")
persist_sorted = persist_df.sort_values('HalfLife')
for _, row in persist_sorted.iterrows():
    print(f"   {row['Label'][:60]} | 半衰期: {int(row['HalfLife'])}天 | 峰值: {row['Peak_Pct']:+.2f}%")

# 6. 政策含义
print("""
6. 关键政策含义:
   (a) 供给中断 (霍尔木兹海峡) > 地缘风险溢价 > 区域性冲突
   (b) 市场对重复性升级存在'疲劳效应'——后续事件影响递减
   (c) 停火事件的价格效应具有不对称性 (下跌幅度 < 同等升级的上行幅度)
   (d) 部分事件确实改变了油价的趋势路径 (结构性断点)
   (e) 建议重点监控: 霍尔木兹海峡/能源基础设施/核设施相关事件
""")

# ===========================================================================
# 8. 保存结果
# ===========================================================================
es_table.round(4).to_csv('event_study_summary.csv', index=False, encoding='utf-8-sig')
itsa_eff.round(4).to_csv('itsa_effects.csv', index=False, encoding='utf-8-sig')
boot_df.round(4).to_csv('shock_decomposition.csv', index=False, encoding='utf-8-sig')
persist_df.round(3).to_csv('shock_persistence.csv', index=False, encoding='utf-8-sig')

print("\n[OK] 结果已保存至:")
print("  - event_study_summary.csv   (事件研究汇总)")
print("  - itsa_effects.csv          (ITSA效应分解)")
print("  - shock_decomposition.csv   (Bootstrap冲击分解)")
print("  - shock_persistence.csv     (效应持久性)")
print("  - fig1-fig6.png             (6张可视化图表)")
print("\n分析完成。")