import yfinance as yf
import pandas as pd
import numpy as np
import math

#-----------------------------------------------------Additional Regimes------------------------------------------------
# taking data frame with the data
xls = pd.ExcelFile("/XXXX/Options data for replicating dispersion.xlsx")
#***************************************************************************************************************+*****

#-----------------------------------------Parsing SPX data & Calculating Vols-------------------------------------------
#date_td = dt.date.today().strftime("%Y-%m-%d")
date_td = "2026-05-01"
date_td = (pd.to_datetime(date_td) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
spx = yf.download("^GSPC", start="2000-01-01", end=date_td, auto_adjust=False)

#log daily returns
spx["Daily_Returns"] = np.log(spx["Adj Close"] / spx["Adj Close"].shift(1))

#calculate EWMA
term1 = (np.log(spx["High"]/spx["Low"]))**2
term2 = (np.log(np.maximum(spx["Open"],spx["Close"])/np.minimum(spx["Open"],spx["Close"])))**2
term3 = (np.log(spx["Close"]/spx["Close"].shift(1)))**2
termsum = (0.8*(term1-term2)+0.2*term3)

#using 20 days of data, with a half-life of five days
spx["ewma_variance"] = termsum.ewm(halflife=5, adjust=False).mean()

#annualising ewma
adj_factor = math.sqrt(252)
spx["r_vol_est_annual"] = (np.sqrt(spx["ewma_variance"] )) * adj_factor * 100
spx.columns = spx.columns.get_level_values(0)
spx = spx.reset_index()
spx["Date"] = pd.to_datetime(spx["Date"], format = '%Y-%m-%d')
spx = spx.set_index("Date")
spx = spx.drop(columns=["Close", "High", "Low", "Open", "Volume", "ewma_variance"])
#------------------------------------------Taking IVs and option chaining-----------------------------------------------
spx_iv = pd.read_excel(xls, sheet_name=f"SPX HMM")
spx_iv = spx_iv.set_index('Date')
spx_iv = spx_iv.sort_index()
spx_iv = spx_iv.rename(columns={"SPX 3M 50D BVOL Index - Last Price (L1)": "3M50D",
                                "SPX 3M 10DP BVOL Index - Last Price (L1)": "3M10DP",
                                "SPX 3M 10DC BVOL Index - Last Price (L1)": "3M10DC"})
#------------------------------------------Matching rows and getting the dataframe--------------------------------------
hmm_df = pd.merge(spx_iv, spx, left_index=True, right_index=True, how='inner')
hmm_df = hmm_df.sort_values("Date")
spx_plot = hmm_df["Adj Close"].copy()
hmm_df = hmm_df.drop(columns=["Adj Close"])
hmm_df.dropna()
print(hmm_df.head())

#---------------------------------------------Calculating Variance Strikes--------------------------------------------------------
v = {}
for i in range(1,11):
    df_excel_sheet = pd.read_excel(xls, sheet_name=f"BBG Companies {i}")
    df_excel_sheet = df_excel_sheet.set_index('Date')
    df_excel_sheet = df_excel_sheet.sort_index()
    v[f"v{i}"] = df_excel_sheet

var_bbg = pd.concat(
    v.values(),
    axis=1,
    join='outer'
)
#do the calculation pairwise
processed_chunks = {}
cols = var_bbg.columns

#b = 90-100 skew in vega / ln(0.9)
#ATMF approx through the ATM 1 Month
#use long formula
for i in range(0, len(cols), 2):
    c1, c2 = (cols[i], cols[i+1])
    ticker = str(c1).split()[0]

    series_atm = var_bbg[c1] / 100
    series_90 = var_bbg[c2] / 100

    b = -(series_90 - series_atm) / (np.log(0.9))
    tm_t_mat = 21 / 252
    base_var = series_atm ** 2
    higher_order_skew = b * (series_atm ** 3) * tm_t_mat
    tail_correction = 0.25 * (b ** 2)
    inp = (12 * series_atm ** 2 * tm_t_mat + 5 * series_atm ** 4 * tm_t_mat ** 2)
    variance_strike = base_var + higher_order_skew + tail_correction * (inp)
    variance_strike = variance_strike * 10000
    processed_chunks[ticker] = variance_strike
var_final_df = pd.DataFrame(processed_chunks)
#-------------------Stocks Realised----------------------------------------
ticker_list = var_final_df.columns
ticker_list = ticker_list.str.replace('/', '-')
ticker_rvols = {}

for i in range (len(ticker_list)):
    ticker = ticker_list[i]
    data = yf.download(ticker, start="2009-01-01", end=date_td, auto_adjust=False, multi_level_index=False)
    data["Daily_Returns"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
    data["Daily_Returns"] = data["Daily_Returns"]**2
    data["Var"] = data["Daily_Returns"].rolling(window=21).sum()
    data["Var"] = data["Var"]*(252/21)
    data["Var"] = data["Var"].shift(-21)
    ticker_rvols[ticker] = data["Var"] * 10000

var_ticker_df = pd.DataFrame(ticker_rvols)
var_ticker_df = var_ticker_df.rename(columns={"BRK-B": "BRK/B"})
#---------------------------------------------------------------------------------

#---------------SPX--------------------------
spx_mkt = pd.read_excel(xls, sheet_name='SPX')
spx_mkt = spx_mkt.set_index('Date')
spx_mkt = spx_mkt.sort_index()
series_atm = spx_mkt.iloc[:,0]/100
series_90 = spx_mkt.iloc[:,1]/100
b = -(series_90-series_atm)/(np.log(0.9))
tm_t_mat = 21/252
base_var = series_atm**2
higher_order_skew = b * (series_atm**3) * tm_t_mat
tail_correction = 0.25 * (b**2)
inp = (12 * series_atm**2 * tm_t_mat + 5*series_atm**4*tm_t_mat**2)
variance_strike = base_var + higher_order_skew + tail_correction*(inp)
variance_strike = variance_strike * 10000
spx_mkt["IVar"] = variance_strike

#---------------------SPX Realised-----------------------------------------
spx["Daily_Returns2"] = spx["Daily_Returns"]**2
spx["Var"] = spx["Daily_Returns2"].rolling(window=21).sum()
spx["Var"] = spx["Var"]*(252/21)
spx["Var"] = spx["Var"].shift(-21) * 10000
spx_mkt = pd.merge(spx_mkt["IVar"], spx["Var"], how='left', left_index=True, right_index=True)
#---------------------------------------------------------------------------------



#-----------------------------------------------Creating Weight Masks------------------------------------------
#-----------MCAP Based--------------------
mcap_raw = pd.read_excel(
    "/Users/matviipolishchuk/Library/CloudStorage/OneDrive-WUWien/SPX_Top20.xlsx",
    sheet_name="Top20_Summary",
    skiprows=2
)
mcap_raw = mcap_raw.dropna(how='all', axis=1)
mcap_raw = mcap_raw.rename(columns={"Weight (%)": "Weight"})
mcap_raw["Weight"] = pd.to_numeric(
    mcap_raw["Weight"].astype(str).str.replace(r'[^\d.]', '', regex=True),
    errors='coerce'
)
mcap_raw = mcap_raw[['Quarter', 'Ticker', 'Weight']].copy()
mcap_raw['Quarter'] = mcap_raw['Quarter'].astype(str).str.replace(r'(Q\d) (\d+)', r'\2\1', regex=True)
mcap_raw['Date'] = pd.PeriodIndex(mcap_raw['Quarter'], freq='Q').to_timestamp()


def expand_group(group):
    start_date = group.index[0]
    end_date = start_date + pd.offsets.QuarterEnd(0)
    monthly_range = pd.date_range(start_date, end_date, freq='ME')
    full_range = group.index.union(monthly_range)
    expanded = group.reindex(full_range).ffill()
    return expanded.reindex(monthly_range).infer_objects(copy=False)

monthly_weight_df = (
    mcap_raw.set_index('Date')
    .groupby(['Quarter', 'Ticker'])
    .apply(expand_group, include_groups=False)
    .reset_index()
    .rename(columns={'level_2': 'Date'})
)

weight = monthly_weight_df[["Date", "Ticker", "Weight"]].copy()
weight["Ticker"] = weight["Ticker"].replace({'BRK.A': 'BRK/B'})
weight_wide = weight.pivot(index='Date', columns='Ticker', values='Weight')
weight_maskx = weight_wide.where(weight_wide.isna(), 1)

#---------------------------------------------------------------------------------


#split into 2 periods: testing and expanding window pnl

#-----------------------------------------------Applying Weight Masks------------------------------------------
#--------------Importing masks------------
xls_mask = pd.ExcelFile("/Users/matviipolishchuk/Downloads/spx_top100_factor_final.3.0 (1).xlsx")
vol260 = pd.read_excel(xls_mask, sheet_name="Vol_260d_Data")
vol120 = pd.read_excel(xls_mask, sheet_name="Vol_120d_Data")
eps = pd.read_excel(xls_mask, sheet_name="EPS_Growth_Data")
beta = pd.read_excel(xls_mask, sheet_name="Beta_Data")

weight_dfs = {'vol260': vol260, 'vol120': vol120, 'eps': eps, 'beta': beta}

def create_weight_mask(raw_df, df_name):
    df_copy = raw_df.copy()
    df_copy.columns = df_copy.iloc[2]
    df_copy = df_copy.iloc[3:]
    df_copy = df_copy.reset_index()
    df_copy = df_copy.drop(df_copy.columns[[0,2,3]], axis=1)
    df_copy = df_copy.replace('NM', np.nan)
    df_copy = df_copy.set_index('Ticker')
    df_copy = df_copy.T
    df_copy.index = pd.to_datetime(df_copy.index, errors='coerce')
    #producing ranking
    #if df_name == 'vol260'or df_name == 'vol120':
    ranks = df_copy.rank(axis=1, ascending=False, method='first')
    #producing masks
    mask_20 = (ranks <= 20)
    weight_mask_factor = mask_20.astype(float).replace(0, np.nan)
    return weight_mask_factor

weight_masks = {}
for name, data in weight_dfs.items():
    print(f"Processing mask for: {name}")
    weight_masks[name] = create_weight_mask(data, name)

weight_masks["mcap"] = weight_maskx

#precalculate the notional positions for stocks
# calculate the variance amount as of Vega Amount/(Strike*2), where Vega is 100000 for each stock
var_final_df_m = var_final_df.resample('BMS').first()
var_ticker_df_m = var_ticker_df.resample('BMS').first()
spx_mkt_m = spx_mkt.resample('BMS').first()
vega_amnt = 100000
variance_amnt = vega_amnt / (2 * np.sqrt(var_final_df_m))
spx_v_p = spx_mkt_m["Var"] - spx_mkt_m["IVar"]
stock_v_p = var_ticker_df_m - var_final_df_m


def apply_pnl_calculation(m_df, var_am, spx_var_payoff, stock_var_payoff):
    weight_mask = m_df.copy()
    weight_mask = weight_mask.dropna(how='all')
    weight_mask.index = weight_mask.index.to_period('M') + 1
    var_ym_index = var_am.index.to_period('M')
    common_ym = weight_mask.index.intersection(var_ym_index)
    weight_mask = weight_mask.loc[common_ym]
    common_dates = var_am.index[var_ym_index.isin(common_ym)]
    weight_mask.index = common_dates
    # ---------------------------------------------------------------------
    weight_mask = weight_mask.loc[common_dates]
    var_am_aligned = var_am.loc[common_dates]
    spx_payoff_aligned = spx_var_payoff.loc[common_dates]
    stock_payoff_aligned = stock_var_payoff.loc[common_dates]
    var_final_df_aligned = var_final_df_m.loc[common_dates]
    spx_mkt_aligned = spx_mkt_m.loc[common_dates]

    variance_amnt = var_am_aligned * weight_mask
    var_not_s = variance_amnt.sum(axis=1, min_count=1)

    active_stock_count = weight_mask.sum(axis=1)
    total_stock_vega = active_stock_count * vega_amnt

    index_vol_strike = np.sqrt(spx_mkt_aligned["IVar"])
    var_not_i = total_stock_vega / (2 * index_vol_strike)
    # ------------------------PnL in Variance Strike Neutral-------------------------
    long_leg_pnl = (stock_payoff_aligned * variance_amnt).sum(axis=1, min_count=1)
    total_short_leg_pnl = spx_payoff_aligned * var_not_i
    total_pnl_v = long_leg_pnl - total_short_leg_pnl

    # ------------------------ Transaction Costs (Basket Adjusted) -------------------------
    mid_vol_idx = np.sqrt(spx_mkt_aligned["IVar"] / 10000)
    bid_vol_idx = mid_vol_idx
    cost_idx_vp = spx_mkt_aligned["IVar"] - (10000 * bid_vol_idx ** 2)

    mid_vol_stock = np.sqrt(var_final_df_aligned / 10000)
    ask_vol_stock = mid_vol_stock
    cost_stock_vp = (10000 * ask_vol_stock ** 2) - var_final_df_aligned

    weighted_cost_stock_vp = cost_stock_vp * variance_amnt
    total_stock_cost_vp = weighted_cost_stock_vp.sum(axis=1, min_count=1)
    total_cost_vp = (cost_idx_vp * var_not_i) + total_stock_cost_vp

    # ------------------------ Final Net PnL Calculation -------------------------
    net_pnl_series = total_pnl_v - total_cost_vp
    final_pnl_v = net_pnl_series.to_frame(name='PnL')
    final_pnl_v.index = final_pnl_v.index.strftime('%Y-%m-%d')
    final_pnl_v.index.name = 'Date'
    final_pnl_v["PnL"] = final_pnl_v["PnL"] / vega_amnt
    final_pnl_v['Cumulative_PnL'] = final_pnl_v['PnL'].cumsum()

    # ---> INJECTED LINES: Expose the notionals to the output dataframe <---
    final_pnl_v['Notional_Stock_Var'] = var_not_s.values
    final_pnl_v['Notional_Index_Var'] = var_not_i.values

    return final_pnl_v

pnl_df = {}
for name, df in weight_masks.items():
    print(f"Processing mask for: {name}")
    pnl_df[name] = apply_pnl_calculation(df, variance_amnt,spx_v_p, stock_v_p)

import matplotlib.pyplot as plt
for name, df in pnl_df.items():
    print(f"Processing chart for: {name}")
    df_plot = pnl_df[name]
    df_plot["Cumulative_PnL"].plot(
        figsize=(10, 6),
        linewidth=2,
        color='royalblue'
    )
    plt.title(f'Strategy PnL {name}', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Profit', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

#---------------------------------------------Do the strategy implementation---------------------------------------
#-------------------------------------------------Creating and Learning a model-----------------------------------------
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import random

random.seed(42)
train_df = hmm_df[hmm_df.index < "2022-02-01"]
test_df = hmm_df[hmm_df.index >= "2022-02-01"]

# =====================================================================
# 1. IN-SAMPLE REGIMES (Backtested "as usual" on the static training set)
# =====================================================================
scaler_train = StandardScaler()
X_scaled_train = scaler_train.fit_transform(train_df)

model_train = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000, random_state=42, algorithm='viterbi')
model_train.fit(X_scaled_train)
real_means_train = scaler_train.inverse_transform(model_train.means_)
order_train = np.argsort(real_means_train[:, 4])
state_map_train = {original: standard for standard, original in enumerate(order_train)}
raw_states_train = model_train.predict(X_scaled_train)
aligned_train_regimes = [state_map_train[state] for state in raw_states_train]
train_regime_df = pd.DataFrame({'Regime': aligned_train_regimes}, index=train_df.index)
# =====================================================================
# 2. OUT-OF-SAMPLE REGIMES (Your Expanding Window Loop)
# =====================================================================
test_months = test_df.resample('ME').last().index
aligned_daily_regimes = []
for month_end in test_months:
    month_start = month_end.replace(day=1)
    history_df = hmm_df[hmm_df.index < month_start]
    current_month_df = hmm_df[(hmm_df.index >= month_start) & (hmm_df.index <= month_end)]
    if current_month_df.empty:
        continue
    scaler = StandardScaler()
    X_scaled_hist = scaler.fit_transform(history_df)
    model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000, random_state=42, algorithm='viterbi')
    model.fit(X_scaled_hist)

    real_means_pred = scaler.inverse_transform(model.means_)
    #sort for mean
    order = np.argsort(real_means_pred[:, 4])
    state_map = {original: standard for standard, original in enumerate(order)}

    #predict current month
    combined_df = pd.concat([history_df, current_month_df])
    X_scaled_current = scaler.transform(combined_df)
    raw_states = model.predict(X_scaled_current)
    current_month_raw_states = raw_states[-len(current_month_df):]


    for date, raw_state in zip(current_month_df.index, current_month_raw_states):
        aligned_daily_regimes.append({
            'Date': date,
            'Regime': state_map[raw_state]
        })

test_regime_df = pd.DataFrame(aligned_daily_regimes).set_index('Date')
# =====================================================================
# 3. COMBINE FOR FULL TIMELINE
# =====================================================================
full_regime_df = pd.concat([train_regime_df, test_regime_df])
full_regime_df.index.name = 'Date'

#-----------------------------------------------------Model fitted------------------------------------------------

#------------IC / RC Correlation--------
implied_corrs = pd.read_excel(xls, sheet_name='SPX IC RD')
implied_corrs["Date"] = pd.to_datetime(implied_corrs["Date"], format='%Y-%m-%d')
implied_corrs = implied_corrs.set_index("Date")
implied_corrs = implied_corrs.sort_index()
implied_corrs.dropna(inplace=True)
implied_corrs = implied_corrs.rename(columns={"SPX 1M RC BVOL Index - Last Price": "rcol", "SPX 1M IC BVOL Index - Last Price": "impcol"})
implied_corrs["rcol"] = implied_corrs["rcol"]*100
implied_corrs["impcol"] = implied_corrs["impcol"]*100
#apply the tercile
def get_tercile(window):
    if len(window) <= 3:
        return np.nan
    return pd.qcut(window, 3, labels=False).iloc[-1]

implied_corrs["Tercile"] = implied_corrs["impcol"].expanding().apply(get_tercile)

def split_med(window):
    if len(window) <= 3:
        return np.nan
    return pd.qcut(window, 3, labels=False).iloc[-1]

implied_corrs["med_split"] = implied_corrs["impcol"].expanding().apply(split_med)


df_states_backtest = pd.merge_asof(
    full_regime_df[["Regime"]],
    implied_corrs[["Tercile","med_split"]],
    left_index=True,
    right_index=True,
    direction='backward'
)
df_states_backtest.dropna(inplace=True)
df_states_backtest = df_states_backtest.rename(columns={"Tercile": "IC", "med_split": "RC"})

#resample into monthly
df_states_backtest_m = df_states_backtest.resample('ME').apply(lambda x: x.mode().iloc[0] if not x.empty else np.nan)
df_states_backtest_m = df_states_backtest_m.dropna().astype(int)

#redo into combined
df_states_backtest_m["Regime"] = df_states_backtest_m["Regime"].astype(int)
df_states_backtest_m["IC"] = df_states_backtest_m["IC"].astype(int)
df_states_backtest_m["RC"] = df_states_backtest_m["RC"].astype(int)
df_states_backtest_m['Combined_Regime'] = df_states_backtest_m['Regime'].astype(str) + df_states_backtest_m['IC'].astype(str) + df_states_backtest_m['RC'].astype(str)
df_states_backtest_m['Combined_Regime'] = df_states_backtest_m['Combined_Regime'].astype(str)
df_states_backtest_m.index = df_states_backtest_m.index.to_period('M') + 1

#merge pnl data and the regimes
#creating a benchmark
pnl_list = [df['PnL'].rename(factor_name) for factor_name, df in pnl_df.items()]
all_pnls_df = pd.concat(pnl_list, axis=1)
benchmark_pnl = all_pnls_df.mean(axis=1)
benchmark_df = pd.DataFrame({'PnL': benchmark_pnl})
benchmark_df = benchmark_df.dropna()

# ------------------------ Assembly ------------------------
master_df = all_pnls_df.copy()
master_df.index = pd.to_datetime(master_df.index).to_period('M')
benchmark_df.index = pd.to_datetime(benchmark_df.index).to_period('M')
master_df['Benchmark'] = benchmark_df['PnL']
master_df['Combined_Regime'] = df_states_backtest_m['Combined_Regime']
master_df = master_df.dropna(subset=['Combined_Regime'])

# ------------------------ Walk-Forward Backtest ------------------------
factor_cols = [col for col in master_df.columns if col not in ['Benchmark', 'Combined_Regime']]

tactical_pnl = []
allocation_history = []

# Define your Out-Of-Sample start date (adjust this to your specific test period)
oos_start_period = pd.Period("2022-01", freq='M')
oos_dates = master_df[master_df.index >= oos_start_period].index

rule_counter = 0
current_rules = {}
for current_date in oos_dates:

    history = master_df[master_df.index < current_date]

    # --- RECALCULATE RULES EVERY 1 MONTH ---
    if rule_counter % 1 == 0:
        if not history.empty:
            current_rules = {}

            # =================================================================
            # 1. EVALUATE FACTORS BY MEAN PNL FOR EVERY REGIME
            # =================================================================
            for r_key, r_history in history.groupby('Combined_Regime'):
                if r_history.empty:
                    continue

                # Rank strategies inside the regime by pure historical mean PnL
                factor_means = r_history[factor_cols].mean()

                # Identify the single absolute best performing strategy
                best_factor = factor_means.idxmax()

                current_rules[str(r_key)] = {
                    'best_factor': best_factor
                }

    # =================================================================
    # 2. APPLY RULES FOR THE CURRENT MONTH (EXECUTION)
    # =================================================================
    todays_regime = str(master_df.loc[current_date, 'Combined_Regime'])

    if todays_regime not in current_rules:
        # Failsafe for a completely unseen regime
        chosen_strategy = 'Wait (Unseen Regime)'
        monthly_pnl = 0.0
    else:
        # --- PURE PERFORMANCE ALLOCATION ---
        # Go long the historically best performing dispersion strategy for this regime
        best_strat = current_rules[todays_regime]['best_factor']
        chosen_strategy = best_strat
        monthly_pnl = master_df.loc[current_date, best_strat]

    # --- THE LIVE TRADING SAFEGUARD ---
    if pd.isna(monthly_pnl):
        monthly_pnl = 0.0

    tactical_pnl.append(monthly_pnl)
    allocation_history.append(chosen_strategy)

    rule_counter += 1
# --- ASSEMBLE FINAL RESULTS ---
tactical_df = pd.DataFrame({
    'Tactical_PnL': tactical_pnl,
    'Strategy_Played': allocation_history,
    'Benchmark': master_df.iloc[master_df.index >= "2022-01-01"]['Benchmark'],
    'Regime': master_df.iloc[master_df.index >= "2022-01-01"]['Combined_Regime'],
}, index=oos_dates)

tactical_df['Cumulative_Tactical'] = tactical_df['Tactical_PnL'].cumsum()

print("\nAllocation Breakdown:")
print(tactical_df['Strategy_Played'].value_counts())

#-----------------------------------------------------
#       Do the printing
#-----------------------------------------------------
df_plot_bch = benchmark_df.iloc[benchmark_df.index >= "2022-01-01"].copy()
df_plot_bch["Cumulative_PnL"] = df_plot_bch["PnL"].cumsum()
df_plot_bch["Cumulative_PnL"].plot(
    figsize=(10, 6),
    linewidth=2,
    color='royalblue'
)
plt.title(f'Strategy PnL Benchmark', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Profit', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

df_plot_tactics = tactical_df.copy()
df_plot_tactics["Cumulative_Tactical"].plot(
    figsize=(10, 6),
    linewidth=2,
    color='royalblue'
)
plt.title(f'Strategy PnL Tactical', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Profit', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==============================================================================
#                      LIVE PRODUCTION TRADING REPORT
# ==============================================================================

# 1. Identify the latest month in the dataset
latest_date = master_df.index[-1]
latest_regime = master_df.loc[latest_date, 'Combined_Regime']

print(f"\n{'=' * 55}")
print(f" TACTICAL ALLOCATION REPORT FOR: {latest_date}")
print(f"{'=' * 55}")
print(f"Current Macro/Vol Regime Detected : [{latest_regime}]")

# 2. Extract the Historical Performance Table for all Regimes
print("\n--- 1. Current Regime Hierarchy (Median Benchmark PnL) ---")
# We look at all data up to the latest date to see the rankings
regime_medians = master_df.groupby('Combined_Regime')['Benchmark'].median().sort_values(ascending=False)
regime_rankings_df = regime_medians.to_frame(name="Median_Benchmark_PnL")
print(regime_rankings_df)

# Determine where the current regime sits in the hierarchy
total_r = len(regime_rankings_df)
current_rank = regime_rankings_df.index.get_loc(latest_regime) + 1
print(f"\n=> Regime [{latest_regime}] is ranked #{current_rank} out of {total_r}.")

# 3. Extract Factor Performance specific to today's Regime
print(f"\n--- 2. Factor Mean Performance in Regime [{latest_regime}] ---")
current_regime_data = master_df[master_df['Combined_Regime'] == latest_regime]

if not current_regime_data.empty:
    factor_means = current_regime_data[factor_cols].mean().sort_values(ascending=False)
    print(factor_means.to_frame(name="Mean_PnL"))
else:
    print("WARNING: This is a completely new regime. No historical data exists.")

# 4. Print the Final Prescribed Action
print("\n--- 3. PRESCRIBED TACTICAL ACTION ---")
# We just pull the final decision your loop already calculated for this month
latest_decision = tactical_df.loc[latest_date, 'Strategy_Played']

print(f"ACTION TO EXECUTE: [ {latest_decision.upper()} ]")

# Add a helpful context note if the 3-month streak rule got triggered
if latest_decision == 'Wait (Not 1 Consecutive)':
    print(
        "CONTEXT: Regime is classified as bottom-tier, but the 3-month consecutive streak is not yet met. Staying flat.")
elif latest_decision == 'short_dispersion':
    print("CONTEXT: 3-month bottom-tier streak confirmed. Shorting dispersion.")

print(f"{'=' * 55}\n")

tactical_df["Cumulative_Tactical"] = tactical_df["Cumulative_Tactical"] / 10000
tactical_df["Tactical_PnL"] = tactical_df["Tactical_PnL"] / 10000
tactical_df["Benchmark"] = tactical_df["Benchmark"].cumsum() / 10000
master_df["vol260"] = master_df["vol260"].cumsum() / 10000
master_df["vol120"] = master_df["vol120"].cumsum()  / 10000
master_df["beta"] = master_df["beta"].cumsum()  / 10000
master_df["eps"] = master_df["eps"].cumsum()  / 10000
master_df["mcap"] = master_df["mcap"].cumsum()  / 10000
master_df["Benchmark"] = master_df["Benchmark"].cumsum()  / 10000

tactical_df.to_csv("tactical_df.csv")
master_df.to_csv("strategiespnlregime_df.csv")

# ==============================================================================
#                      MAY 1st VOL 120D NOTIONALS REPORT
# ==============================================================================
target_date = "2026-05-01"

if 'eps' in pnl_df:
    try:
        may_stock_notional = pnl_df['eps'].loc[target_date, 'Notional_Stock_Var']
        may_index_notional = pnl_df['eps'].loc[target_date, 'Notional_Index_Var']

        print(f"\n{'=' * 55}")
        print(f" EPS FOR {target_date}")
        print(f"{'=' * 55}")
        print(f"Sum of Notional Stock Variances : {may_stock_notional:,.4f}")
        print(f"Notional Index Variance         : {may_index_notional:,.4f}")
        print(f"{'=' * 55}\n")
    except KeyError:
        print(f"\nWARNING: Data for {target_date} is not yet available in the index. Latest date is {pnl_df['vol120'].index[-1]}")