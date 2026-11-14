"""Kết quả sơ bộ cho 5 phương pháp thống kê, chạy trên bảng feature cấp đợt KM.

Chạy sau build_episode_features.py:  python src/prelim_analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

F = Path(__file__).resolve().parents[1] / "output" / "features"
ep = pd.read_csv(F / "episode_features.csv", low_memory=False)
pl = pd.read_csv(F / "placebo_episodes.csv", low_memory=False)
ep, pl = ep[ep.eligible == 1], pl[pl.eligible == 1]
both = pd.concat([ep, pl], ignore_index=True).dropna(subset=["log_post_ratio"])

print("== Mô tả")
print(both.groupby(["source", "group"]).agg(
    n=("log_post_ratio", "size"), items=("item_id", "nunique"), lift_med=("lift", "median"),
    post_ratio_med=("post_ratio", "median"), dip_rate=("dip", "mean"),
    mean_log_post=("log_post_ratio", "mean")).round(3).to_string())
print(ep.hol_status.value_counts().to_string())
print(ep.promo_type.value_counts().head(6).to_string())

print("\n== Two-way ANOVA: log_post_ratio ~ group * source (HC3)")
m = smf.ols("log_post_ratio ~ C(group) * C(source)", data=both).fit()
a = anova_lm(m, typ=2, robust="hc3")
a["eta2_partial"] = a.sum_sq / (a.sum_sq + a.loc["Residual", "sum_sq"])
print(a.round(4).to_string())

print("\n== T-test: log_post_ratio (KM thật vs đợt giả), theo nhóm")
for g in ["food", "household"]:
    x = both[(both.group == g) & (both.source == "promo")].log_post_ratio
    y = both[(both.group == g) & (both.source == "placebo")].log_post_ratio
    t1 = stats.ttest_1samp(x, 0)
    t2 = stats.ttest_ind(x, y, equal_var=False)
    d = (x.mean() - y.mean()) / np.sqrt((x.var() + y.var()) / 2)
    print(f"{g}: mean promo={x.mean():.3f} (1-sample p={t1.pvalue:.2g}); placebo={y.mean():.3f}; "
          f"Welch t={t2.statistic:.2f} p={t2.pvalue:.2g}; Cohen d={d:.3f}; "
          f"Mann-Whitney p={stats.mannwhitneyu(x, y).pvalue:.2g}")

print("\n== Chi-square: nhóm hàng x dip (đợt KM thật)")
ct = pd.crosstab(ep.group, ep.dip)
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(ct.to_string()); print(f"chi2={chi2:.2f} dof={dof} p={p:.2g} Cramer V={np.sqrt(chi2 / ct.values.sum()):.3f}")
ct2 = pd.crosstab(both.source, both.dip)
chi2, p, dof, _ = stats.chi2_contingency(ct2)
print(ct2.to_string()); print(f"KM vs giả: chi2={chi2:.2f} p={p:.2g} Cramer V={np.sqrt(chi2 / ct2.values.sum()):.3f}")

print("\n== Logistic: dip ~ is_food + đặc trưng đợt KM")
d = ep.dropna(subset=["depth", "log_price_before"]).copy()
top = d.promo_type.value_counts().index[:4]
d["promo_type"] = d.promo_type.where(d.promo_type.isin(top), "other")
d["log_duration"] = np.log(d.duration)
lr = smf.logit("dip ~ is_food + depth + log_duration + log_price_before + pre_days + C(promo_type)"
               " + C(hol_status) + C(store_id) + C(start_month)", data=d).fit(disp=0, cov_type="HC1")
ci = np.exp(lr.conf_int())
for v in ["is_food", "depth", "log_duration", "log_price_before", "pre_days"]:
    print(f"{v}: OR={np.exp(lr.params[v]):.3f} [{ci.loc[v, 0]:.3f}, {ci.loc[v, 1]:.3f}] p={lr.pvalues[v]:.2g}")
print(f"n={int(lr.nobs)} pseudo R2={lr.prsquared:.3f}")

print("\n== Log-log elasticity (cấp đợt): log_lift ~ log_price_ratio * group")
e = ep[(ep.depth > 0) & ep.log_price_ratio.notna() & ep.log_lift.notna()]
el = smf.ols("log_lift ~ log_price_ratio * C(group) + C(store_id) + C(start_month)", data=e).fit(cov_type="HC1")
b_food = el.params["log_price_ratio"]
b_hh = b_food + el.params["log_price_ratio:C(group)[T.household]"]
print(f"n={int(el.nobs)}  elasticity food={b_food:.3f} (p={el.pvalues['log_price_ratio']:.2g})  "
      f"household={b_hh:.3f}  diff p={el.pvalues['log_price_ratio:C(group)[T.household]']:.2g}")
print(e.groupby("group").arc_elasticity.median().round(3).to_string())
