import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests


def get_sig_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''
    

def test_decoding_above_chance(df, n_voxels_list=None, ROIs=["EVC", "A1"], dv="correct",
                                chance_level=0.5, ci=0.95):
    """
    One-sample t-tests against chance level for each ROI × ROI size.
    """
    summary = []

    for roi in ROIs:
        for n_voxels in n_voxels_list:
            dat = df[(df["ROI"] == roi) & (df["n_voxels_label"] == n_voxels)]
            n_subj = dat["subj"].nunique()
            subj_means = dat.groupby("subj")[dv].mean()
            acc_values = subj_means

            t_val, p_val = stats.ttest_1samp(acc_values, chance_level)
            mean_acc = acc_values.mean()
            se = stats.sem(acc_values)
            df_ = len(acc_values) - 1
            h = stats.t.ppf((1 + ci) / 2., df_) * se
            ci_lower, ci_upper = mean_acc - h, mean_acc + h

            summary.append({
                "ROI": roi,
                "ROI size": n_voxels,
                "n": n_subj,
                "Mean Accuracy": round(mean_acc, 3),
                "t": round(t_val, 2),
                "p": round(p_val, 3),
                "CI Lower": round(ci_lower, 3),
                "CI Upper": round(ci_upper, 3),
                "significance": get_sig_stars(p_val)
            })

    return pd.DataFrame(summary)


def decoding_attended_modalities_stats(df, n_voxels_list=None, ROIs=["EVC", "A1"], dv="correct",
    subject_col="subj", cond_col="attended modality", conditions=("visual", "auditory"), ci=0.95):
    """
    Paired t-tests comparing decoding between attended modalities for each ROI × ROI size.
    """
    if n_voxels_list is None:
        n_voxels_list = df["n_voxels_label"].unique()

    results = []

    for roi in ROIs:
        for n_vox in n_voxels_list:
            subset = df[(df["ROI"] == roi) & (df["n_voxels_label"] == n_vox)]

            pivoted = subset.pivot(index=subject_col, columns=cond_col, values=dv)
            pivoted = pivoted.dropna(subset=conditions)

            if pivoted.shape[0] < 2:
                continue

            diff = pivoted[conditions[0]] - pivoted[conditions[1]]
            t_val, p_val = stats.ttest_rel(pivoted[conditions[0]], pivoted[conditions[1]])

            mean_diff = diff.mean()
            se = stats.sem(diff)
            df_ = len(diff) - 1
            h = stats.t.ppf((1 + ci) / 2., df_) * se
            ci_lower, ci_upper = mean_diff - h, mean_diff + h
            d = mean_diff / diff.std(ddof=1)

            results.append({
                "ROI": roi,
                "ROI size": n_vox,
                "n": len(diff),
                "Mean Difference": round(mean_diff, 3),
                "t": round(t_val, 2),
                "p": round(p_val, 3),
                "d": round(d, 2),
                "CI Lower": round(ci_lower, 3),
                "CI Upper": round(ci_upper, 3),
                "significance": get_sig_stars(p_val)
            })
    return pd.DataFrame(results)



def decoding_crossmodal_stats(
    df,
    dv="correct",
    subject="subj",
    within_factors=("a_pred", "v_pred"),
    voxel_label_col="n_voxels_label",
    p_adjust_method="fdr_bh",
    alternative="two-sided",
):
    """
    rmANOVA per ROI size and FDR-corrected pairwise v_pred tests across a_pred levels.
    """
    factor1, factor2 = within_factors
    anova_results = []
    all_ttests = []

    for roi_label, df_roi in df.groupby(voxel_label_col):
        aov = pg.rm_anova(
            dv=dv,
            within=[factor1, factor2],
            subject=subject,
            data=df_roi,
            detailed=True
        )
        aov[voxel_label_col] = roi_label
        anova_results.append(aov)

        roi_tests = []
        for a_val in df_roi[factor1].unique():
            subset = df_roi[df_roi[factor1] == a_val]
            res = pg.pairwise_tests(
                dv=dv,
                within=factor2,
                subject=subject,
                data=subset,
                padjust=None,
                effsize="cohen",
                alternative=alternative
            )
            res[factor1] = a_val
            res[voxel_label_col] = roi_label
            roi_tests.append(res)

        roi_tests_df = pd.concat(roi_tests, ignore_index=True)
        reject, pvals_corr, _, _ = multipletests(
            roi_tests_df["p-unc"].values, alpha=0.05, method=p_adjust_method
        )
        roi_tests_df["p-corr"] = np.round(pvals_corr, 3)
        roi_tests_df["significance"] = [get_sig_stars(p) for p in pvals_corr]

        all_ttests.append(roi_tests_df)

    anova_df = pd.concat(anova_results, ignore_index=True)
    anova_df = anova_df.rename(columns={
        "Source": "Effect", "DF": "df", "SS": "SS", "MS": "MS",
        "F": "F", "p-GG-corr": "p", "np2": "η²ₚ", voxel_label_col: "n_voxels_label"
    })
    anova_df = anova_df.round({"SS": 3, "MS": 3, "F": 2, "p": 3, "η²ₚ": 3})
    anova_df["significance"] = [get_sig_stars(p) for p in anova_df["p"]]

    ttest_df = pd.concat(all_ttests, ignore_index=True)
    ttest_df = ttest_df.rename(columns={
        voxel_label_col: "n_voxels_label",
        factor1: "a_pred",
        "Contrast": "comparison",
        "T": "t",
        "dof": "df",
        "p-unc": "p_uncorrected",
        "p-corr": "p_corrected",
        "cohen-d": "d"
    })
    ttest_df = ttest_df.round({"t": 2, "df": 0, "p_uncorrected": 3, "p_corrected": 3, "d": 2})

    return anova_df, ttest_df


def decoding_crossmodal_stats_byaEXP(
    df,
    dv="correct",
    subject="subj",
    within_factors=("modality", "v_pred"),
    voxel_label_col="n_voxels_label",
    p_adjust_method="fdr_bh",
    alternative="two-sided",
):
    """
    rmANOVA per ROI size and FDR-corrected pairwise v_pred tests across a_pred levels.
    """
    factor1, factor2 = within_factors
    anova_results = []
    all_ttests = []

    for roi_label, df_roi in df.groupby(voxel_label_col):
        aov = pg.rm_anova(
            dv=dv,
            within=[factor1, factor2],
            subject=subject,
            data=df_roi,
            detailed=True
        )
        aov[voxel_label_col] = roi_label
        anova_results.append(aov)

        roi_tests = []
        for a_val in df_roi[factor1].unique():
            subset = df_roi[df_roi[factor1] == a_val]
            res = pg.pairwise_tests(
                dv=dv,
                within=factor2,
                subject=subject,
                data=subset,
                padjust=None,
                effsize="cohen",
                alternative=alternative
            )
            res[factor1] = a_val
            res[voxel_label_col] = roi_label
            roi_tests.append(res)

        roi_tests_df = pd.concat(roi_tests, ignore_index=True)
        reject, pvals_corr, _, _ = multipletests(
            roi_tests_df["p-unc"].values, alpha=0.05, method=p_adjust_method
        )
        roi_tests_df["p-corr"] = np.round(pvals_corr, 3)
        roi_tests_df["significance"] = [get_sig_stars(p) for p in pvals_corr]

        all_ttests.append(roi_tests_df)

    anova_df = pd.concat(anova_results, ignore_index=True)
    anova_df = anova_df.rename(columns={
        "Source": "Effect", "DF": "df", "SS": "SS", "MS": "MS",
        "F": "F", "p-GG-corr": "p", "np2": "η²ₚ", voxel_label_col: "n_voxels_label"
    })
    anova_df = anova_df.round({"SS": 3, "MS": 3, "F": 2, "p": 3, "η²ₚ": 3})
    anova_df["significance"] = [get_sig_stars(p) for p in anova_df["p"]]

    ttest_df = pd.concat(all_ttests, ignore_index=True)
    ttest_df = ttest_df.rename(columns={
        voxel_label_col: "n_voxels_label",
        factor1: "a_pred",
        "Contrast": "comparison",
        "T": "t",
        "dof": "df",
        "p-unc": "p_uncorrected",
        "p-corr": "p_corrected",
        "cohen-d": "d"
    })
    ttest_df = ttest_df.round({"t": 2, "df": 0, "p_uncorrected": 3, "p_corrected": 3, "d": 2})

    return anova_df, ttest_df