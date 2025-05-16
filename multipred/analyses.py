import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests


def test_decoding_above_chance(df, n_voxels_list, ROIs = ["EVC", "A1"], dv="correct",
                                chance_level=0.5, ci=0.95):
    """
    Perform one-sample t-tests against chance level for each ROI × n_voxels × attended modality group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing subject-level decoding accuracy and grouping variables.
    dv : str
        Dependent variable column name (default: "correct").
    group_cols : tuple
        Columns to group by for testing (default: ("ROI", "n_voxels_label")).
    chance_level : float
        Chance level for t-test (default: 0.5).
    ci : float
        Confidence level for intervals (default: 0.95).

    Returns
    -------
    summary_df : pd.DataFrame
        Table of t-tests with mean accuracy, t, p, CI, and significance.
    """
    summary = []

    for roi in ROIs:
        for n_voxels in n_voxels_list:
            dat = df[(df["ROI"] == roi) & (df["n_voxels_label"] == n_voxels)]
            n_subj = len(dat["subj"].unique())
            subj_means = dat.groupby("subj")[dv].mean()
            acc_values = subj_means
            acc = acc_values.mean()
            t_val, p_val = stats.ttest_1samp(acc_values, chance_level)
            mean_acc = acc_values.mean()
            se = stats.sem(acc_values)
            df_ = len(acc_values) - 1
            h = stats.t.ppf((1 + ci) / 2., df_) * se
            ci_lower, ci_upper = mean_acc - h, mean_acc + h

            summary.append({
                "n_voxels_label": n_voxels,
                "ROI": roi,
                "n_subjects": n_subj,
                "accuracy": acc,
                "t_value": t_val,
                "p_value": p_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            })

    summary_df = pd.DataFrame(summary)

    return summary_df


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
    Perform rmANOVA per ROI and FDR-corrected pairwise v_pred tests within each ROI across a_pred levels.

    Parameters
    ----------
    df : pd.DataFrame
        Data with decoding accuracy and predictors.
    dv : str
        Dependent variable column (e.g., "correct").
    subject : str
        Subject ID column.
    within_factors : tuple of str
        ('a_pred', 'v_pred') assumed.
    voxel_label_col : str
        ROI label column (e.g., "n_voxels_label").
    p_adjust_method : str
        Correction method for multiple comparisons within each ROI (default: 'fdr_bh').
    alternative : str
        Hypothesis type for pairwise tests ('less', 'two-sided', etc.).

    Returns
    -------
    anova_df : pd.DataFrame
        rmANOVA results per ROI (MultiIndex: ROI x Source).
    ttest_df : pd.DataFrame
        Pairwise v_pred tests (MultiIndex: ROI x a_pred x Contrast) with FDR per ROI.
    """

    factor1, factor2 = within_factors
    anova_results = []
    all_ttests = []

    for roi_label, df_roi in df.groupby(voxel_label_col):
        # 1. rmANOVA for the current ROI
        aov = pg.rm_anova(
            dv=dv,
            within=[factor1, factor2],
            subject=subject,
            data=df_roi,
            detailed=True
        )
        aov[voxel_label_col] = roi_label
        anova_results.append(aov)

        # 2. Run pairwise tests for each level of a_pred
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

        # Concatenate and apply FDR *within this ROI*
        roi_tests_df = pd.concat(roi_tests, ignore_index=True)
        reject, pvals_corr, _, _ = multipletests(
            roi_tests_df["p-unc"].values, alpha=0.05, method=p_adjust_method
        )
        roi_tests_df["p-corr"] = pvals_corr
        roi_tests_df["significant"] = reject
        all_ttests.append(roi_tests_df)

    # Final combined results
    anova_df = pd.concat(anova_results, ignore_index=True)
    ttest_df = pd.concat(all_ttests, ignore_index=True)

    # Set MultiIndex
    anova_df.set_index([voxel_label_col, "Source"], inplace=True)
    ttest_df.set_index([voxel_label_col, factor1, "Contrast"], inplace=True)

    return anova_df, ttest_df
