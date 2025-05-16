import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

from multipred.data import extract_slice, validate_mri_input

def barplot_morey_maineffect(dat, x, y, palette1, palette2, barplot_plot=True, swarmplot_plot=True, within_lines=True, ax=None, fontsize=12):
    # Step 1: Normalize within subjects by subtracting the subject-specific mean
    dat['subject_mean'] = dat.groupby('subj')[y].transform('mean')
    dat['cousineau_normalized'] = dat[y] - dat['subject_mean']

    # Step 2: Add the grand mean to retain interpretability
    grand_mean = dat[y].mean()
    dat['cousineau_normalized'] += grand_mean

    # Step 3: Compute group-level mean and Cousineau-Morey corrected SE
    n_conditions = dat[x].nunique()
    cousineau_summary = dat.groupby(x).agg(
        group_mean=('cousineau_normalized', 'mean'),
        corrected_se=('cousineau_normalized', lambda x: x.std() / np.sqrt(len(x)) * np.sqrt(n_conditions / (n_conditions - 1)))
    ).reset_index()

    # Step 4: Plot
    if ax is None:
        ax = plt.gca()  # Use the current axis if none is provided

    if barplot_plot:
        sns.barplot(
            data=cousineau_summary,
            x=x,
            y='group_mean',
            palette=palette1,
            errorbar=None,
            ax=ax,
            legend=False
        )

    if swarmplot_plot:
        sns.swarmplot(
            data=dat,
            x=x,
            y=y,
            palette=palette2,
            size=6,
            alpha=1,
            edgecolor="grey",
            linewidth=0.5,
            ax=ax,
            legend=False
        )

    if within_lines:
        # Add lines connecting within-subject data points
        for subject_id in dat['subj'].unique():
            subject_data = dat[dat['subj'] == subject_id]
            x_positions = [list(dat[x].unique()).index(level) for level in subject_data[x]]
            ax.plot(
                x_positions,
                subject_data[y].values,
                color='grey',
                linewidth=0.5,
                alpha=0.7
            )

    # Overlay means with Cousineau-Morey SE
    for _, row in cousineau_summary.iterrows():
        x_position = list(dat[x].unique()).index(row[x])
        ax.errorbar(
            x=x_position,
            y=row['group_mean'],
            yerr=row['corrected_se'],
            fmt='o',
            color='black',
            capsize=4,
            elinewidth=2,
            markersize=0,
            zorder=10
        )
            # Set axis labels and tick labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)


def barplot_morey(dat, x, y, hue, hue_label, palette1, palette2, barplot_plot=True, swarmplot_plot=True, within_lines=True, ax=None, fontsize=12, legend=True, errorbar_style={"capsize": 4, "elinewidth": 2, "markersize": 4}):
    # Step 1: Normalize within subjects by subtracting the subject-specific mean
    dat['subject_mean'] = dat.groupby('subj')[y].transform('mean')
    dat['cousineau_normalized'] = dat[y] - dat['subject_mean']

    # Step 2: Add the grand mean to retain interpretability
    grand_mean = dat[y].mean()
    dat['cousineau_normalized'] += grand_mean

    # Step 3: Compute group-level mean and Cousineau-Morey corrected SE
    n_conditions = dat[x].nunique() * dat[hue].nunique()
    cousineau_summary = dat.groupby([x, hue]).agg(
        group_mean=('cousineau_normalized', 'mean'),
        corrected_se=('cousineau_normalized', lambda x: x.std() / np.sqrt(len(x)) * np.sqrt(n_conditions / (n_conditions - 1)))
    ).reset_index()

    # Step 4: Plot
    if ax is None:
        ax = plt.gca()  # Default to the current axis if no axis is provided

    if barplot_plot:
        sns.barplot(
            data=dat,
            x=x,
            y=y,
            hue=hue,
            palette=palette1,
            errorbar=None,
            ax=ax,
            dodge=True,
            legend=False
        )

    if swarmplot_plot:
        sns.swarmplot(
            data=dat,
            x=x,
            y=y,
            hue=hue,
            dodge=True,
            palette=palette2,
            size=4,
            alpha=1,
            edgecolor="grey",
            linewidth=0.5,
            ax=ax,
            legend=False
        )

    # Retrieve unique x_levels and estimated dodge offsets for hue levels
    x_levels = dat[x].unique()
    hue_levels = dat[hue].unique()
    hue_offsets = [-0.2, 0.2]  # Offset to align lines to dodge positions of each hue level in swarmplot


    if within_lines:    
        for x_lvl in x_levels:
            x_lvl_data = dat[dat[x] == x_lvl]
            for subject_id in x_lvl_data['subj'].unique():
                subject_data = x_lvl_data[x_lvl_data['subj'] == subject_id]

                if len(subject_data) == len(hue_levels):  # Ensure all hue levels exist
                    x_position = list(x_levels).index(x_lvl)
                    x_positions = [x_position + hue_offsets[i] for i in range(len(hue_levels))]
                    ax.plot(x_positions, subject_data[y].values, color='grey', linewidth=0.5, alpha=0.7)

    
    # Overlay means with Cousineau-Morey SE
    for _, row in cousineau_summary.iterrows():
        x_position = list(dat[x].unique()).index(row[x])
        x_offset = -0.2 if row[hue] == hue_levels[0] else 0.2
        if barplot_plot:
            color_error = 'black'
        else:
            color_error = palette1[0] if row[hue] == hue_levels[0] else palette1[1]
        ax.errorbar(
            x=x_position + x_offset,
            y=row['group_mean'],
            yerr=row['corrected_se'],
            fmt='o',
            color= color_error,
            capsize=errorbar_style["capsize"],
            elinewidth=errorbar_style["elinewidth"],
            markersize=errorbar_style["markersize"],
            zorder=10
        )

    # Set axis labels and tick labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    # Create legend handles using the hue levels and the corresponding palette colors
    if legend:
        legend_handles = [Patch(facecolor=palette1[i], label=hue_level) for i, hue_level in enumerate(hue_levels)]

        # Add legend to the plot only if there are valid handles
        if legend_handles:
            ax.legend(handles=legend_handles, title=hue_label, bbox_to_anchor=(1, 1), title_fontsize=fontsize, fontsize=fontsize-2, loc='best', frameon=True)


def plot_crossmodal(df, y, x, hue, hue_label, id, ROI, n_voxels, save_fig, ylim, yticks, legend=False):

    if x == "a_pred":
        x_label = "auditory"
    else:
        x_label = "visual"

    unique_modalities = ["auditory", "visual"]
    n_cols = len(unique_modalities)

    # Create the figure and axes for the subplots
    fig, axes = plt.subplots(1, n_cols, figsize=(8, 4))


    for i, modality in enumerate(unique_modalities):
        dat = df[
            df["modality"] == modality
        ].copy()  

        # Plot the data
        ax = axes[i]  # Get the specific axis for the current subplot

        swarmplot_plot = False
        within_lines = False
        barplot_plot = True

        # Define the color palettes
        palette1 = ["firebrick", "turquoise"]
        palette2 = ["lightcoral", "powderblue"]

        # Custom plotting function call
        barplot_morey(
            dat,
            x,
            y,
            hue,
            hue_label,
            palette1,
            palette2,
            barplot_plot,
            swarmplot_plot,
            within_lines,
            ax=ax,
            legend=False
        )

        # Customize each subplot
        ax.set_ylabel(f"Classification accuracy in {ROI}")
        ax.set_xlabel(x_label)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["UEX", "EXP"])
        ax.set_title(f"{modality} attended")
        ax.set_ylim(ylim)
        ax.set_xlim(-0.5, 1.5) 
        ax.set_yticks(yticks)
        ax.axhline(0.5, color="grey", linestyle="--")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"figures/MVPA/{ROI}_{n_voxels}_acc_interaction.svg", dpi=300, bbox_inches="tight")

    plt.show()



# def plot_errobars(
#     df, x, y, hue, palette=None, x_offset=1, ax=None, error_type='sem'
# ):
#     """
#     Flexible function to plot classification accuracy (or any metric) with error bars.

#     Parameters:
#     - df: DataFrame with data.
#     - x: x-axis variable (e.g., "n_voxels").
#     - y: dependent variable (e.g., "correct").
#     - hue: grouping variable (e.g., "v_pred").
#     - palette: list or dict of colors for each hue level.
#     - x_offset: spacing offset to prevent marker overlap.
#     - ax: matplotlib axis. If None, a standalone plot is created.
#     - error_type: 'ci95' (default) or 'sem' for standard error bars.
#     """
#     import numpy as np
#     import pandas as pd
#     from scipy import stats
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     standalone = ax is None
#     if standalone:
#         fig, ax = plt.subplots(figsize=(6, 4))

#     # Step 1: Average within subj, x(n_voxels), hue
#     df_grouped = df.groupby(["subj", x, hue])[y].mean().reset_index()

#     if error_type == "ws":  # Within-subject SEM (Cousineau-Morey)
#         # Step 2a: Normalize within subject
#         subj_means = df_grouped.groupby("subj")[y].transform("mean")
#         grand_mean = df_grouped[y].mean()
#         df_grouped["y_norm"] = df_grouped[y] - subj_means + grand_mean

#         # Step 2b: Compute summary on normalized data
#         summary_stats = df_grouped.groupby([x, hue])["y_norm"].agg(["mean", "std", "count"]).reset_index()

#         # Step 2c: Morey correction
#         n_conditions = df_grouped[x].nunique()
#         correction = np.sqrt(n_conditions / (n_conditions - 1))
#         summary_stats["yerr"] = (summary_stats["std"] / np.sqrt(summary_stats["count"])) * correction

#         summary_stats["x_plot"] = summary_stats[x]
#     else:
#         # Standard SEM or CI95
#         summary_stats = df_grouped.groupby([x, hue])[y].agg(['mean', 'count', 'std']).reset_index()
#         if error_type == "sem":
#             summary_stats["yerr"] = summary_stats["std"] / np.sqrt(summary_stats["count"])
#         elif error_type == "ci95":
#             summary_stats["yerr"] = (summary_stats["std"] / np.sqrt(summary_stats["count"])) * \
#                 stats.t.ppf(0.975, df=summary_stats["count"] - 1)
#         else:
#             raise ValueError("error_type must be 'sem', 'ci95', or 'ws'")
#         summary_stats["x_plot"] = summary_stats[x]

#     # Offset for clarity
#     hue_vals = sorted(df[hue].unique())
#     offset_map = {h: (-1)**i * x_offset for i, h in enumerate(hue_vals)}
#     summary_stats["x_plot"] = summary_stats[x] + summary_stats[hue].map(offset_map)

#     # Step 4: set colors
#     if isinstance(palette, dict):
#         color_map = palette
#     elif isinstance(palette, list):
#         color_map = dict(zip(hue_vals, palette))
#     else:
#         color_map = dict(zip(hue_vals, sns.color_palette("deep", len(hue_vals))))

#     # Step 5: plot
#     for h in hue_vals:
#         data_h = summary_stats[summary_stats[hue] == h]
#         ax.plot(data_h["x_plot"], data_h["mean"], label=str(h), color=color_map[h])
#         ax.errorbar(
#             data_h["x_plot"], data_h["mean"], yerr=data_h["yerr"],
#             fmt="o", color=color_map[h], capsize=5
#         )

#     ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
#     ax.set_xlabel(x.replace("_", " ").title())
#     ax.set_ylabel(y.replace("_", " ").title())
#     ax.legend(title=hue)
#     sns.despine(ax=ax)

#     if standalone:
#         plt.tight_layout()
#         plt.show()

#     return ax

def plot_errobars(
    df, x, y, hue, palette=None, x_offset=1, ax=None, error_type='sem'
):
    """
    Flexible function to plot classification accuracy (or any metric) with error bars.

    Parameters:
    - df: DataFrame with data.
    - x: x-axis variable (e.g., "n_voxels").
    - y: dependent variable (e.g., "correct").
    - hue: grouping variable (e.g., "v_pred").
    - palette: list or dict of colors for each hue level.
    - x_offset: spacing offset to prevent marker overlap.
    - ax: matplotlib axis. If None, a standalone plot is created.
    - error_type: 'sem', 'ci95', 'wsSE', or 'wsCI'
      - 'wsSE': within-subject standard error (Cousineau-Morey)
      - 'wsCI': within-subject 95% confidence interval (Cousineau-Morey with t-correction)
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Step 1: Average within subj, x(n_voxels), hue
    df_grouped = df.groupby(["subj", x, hue])[y].mean().reset_index()

    if error_type in ["wsSE", "wsCI"]:
        # Step 2a: Normalize within subject
        subj_means = df_grouped.groupby("subj")[y].transform("mean")
        grand_mean = df_grouped[y].mean()
        df_grouped["y_norm"] = df_grouped[y] - subj_means + grand_mean

        # Step 2b: Compute summary on normalized data
        summary_stats = df_grouped.groupby([x, hue])["y_norm"].agg(["mean", "std", "count"]).reset_index()

        # Step 2c: Morey correction factor
        n_conditions = df_grouped[x].nunique()
        correction = np.sqrt(n_conditions / (n_conditions - 1))
        se_corrected = (summary_stats["std"] / np.sqrt(summary_stats["count"])) * correction

        if error_type == "wsSE":
            summary_stats["yerr"] = se_corrected
        elif error_type == "wsCI":
            # Apply t-distribution correction for CI
            t_critical = stats.t.ppf(0.975, df=summary_stats["count"] - 1)
            summary_stats["yerr"] = se_corrected * t_critical

        summary_stats["x_plot"] = summary_stats[x]

    else:
        # Standard SEM or CI95
        summary_stats = df_grouped.groupby([x, hue])[y].agg(['mean', 'count', 'std']).reset_index()
        if error_type == "sem":
            summary_stats["yerr"] = summary_stats["std"] / np.sqrt(summary_stats["count"])
        elif error_type == "ci95":
            summary_stats["yerr"] = (summary_stats["std"] / np.sqrt(summary_stats["count"])) * \
                stats.t.ppf(0.975, df=summary_stats["count"] - 1)
        else:
            raise ValueError("error_type must be 'sem', 'ci95', 'wsSE', or 'wsCI'")
        summary_stats["x_plot"] = summary_stats[x]

    # Offset for clarity
    hue_vals = sorted(df[hue].unique())
    offset_map = {h: (-1)**i * x_offset for i, h in enumerate(hue_vals)}
    summary_stats["x_plot"] = summary_stats[x] + summary_stats[hue].map(offset_map)

    # Step 4: set colors
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, list):
        color_map = dict(zip(hue_vals, palette))
    else:
        color_map = dict(zip(hue_vals, sns.color_palette("deep", len(hue_vals))))

    # Step 5: plot
    for h in hue_vals:
        data_h = summary_stats[summary_stats[hue] == h]
        ax.plot(data_h["x_plot"], data_h["mean"], label=str(h), color=color_map[h])
        ax.errorbar(
            data_h["x_plot"], data_h["mean"], yerr=data_h["yerr"],
            fmt="o", color=color_map[h], capsize=5
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.legend(title=hue)
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plot_decoding_modalities(data_path, n_voxels_list, visualROI="EVC", auditoryROI="A1", error_type="ci95", save_fig=False):
    """
    Plot decoding accuracy across ROI sizes.
    Parameters
    ----------
    data_path : str
        Path to the data directory.
    n_voxels_list : list
        List of voxel sizes to plot.
    visualROI : str
        Name of the visual ROI.
    auditoryROI : str
        Name of the auditory ROI.  
    - error_type: 'sem', 'ci95', 'wsSE', or 'wsCI' (default: '    - error_type: 'sem', 'ci95', 'wsSE', or 'wsCI' (default: 'CI', as we want to test decoding above chance level with this)
', as we want to test decoding above chance level with this)
      - 'wsSE': within-subject standard error (Cousineau-Morey)
      - 'wsCI': within-subject 95% confidence interval (Cousineau-Morey with t-correction)
    save_fig : bool
        Whether to save the figure.
    """

    # Convert to integers for plotting
    n_voxels_int = [int(n) for n in n_voxels_list[:-2]]
    n_voxels_int.extend([n_voxels_int[-1] + 50, n_voxels_int[-1] + 100])

    summary_rows = []  # Collect stats here

    for i, n_voxels in zip(n_voxels_int, n_voxels_list):
        # Load and label data
        visual_df = pd.read_csv(f"{data_path}{visualROI}_{n_voxels}/balanced_group_data.csv")
        visual_df["ROI"] = visualROI
        auditory_df = pd.read_csv(f"{data_path}{auditoryROI}_{n_voxels}/balanced_group_data.csv")
        auditory_df["ROI"] = auditoryROI

        ROI_df = pd.concat([visual_df, auditory_df], ignore_index=True)
        ROI_df["n_voxels"] = i
        ROI_df["n_voxels_label"] = n_voxels

        ROI_df = ROI_df.groupby(["subj", "ROI", "n_voxels"], as_index=False)["correct"].mean()
        n_subj = ROI_df["subj"].nunique()

        #for modality, roi in [("visual", visualROI), ("auditory", auditoryROI)]:
        for roi in [visualROI, auditoryROI]:
            data_mod = ROI_df[ROI_df["ROI"] == roi]["correct"]
            acc = data_mod.mean()
            t_val, p_val = stats.ttest_1samp(data_mod, 0.5)

            summary_rows.append({
                "n_voxels_label": i,
                "ROI": roi,
                "n_subjects": n_subj,
                "accuracy": acc,
                "t_value": t_val,
                "p_value": p_val
            })

        # Combine across voxel sizes
        if n_voxels == n_voxels_list[0]:
            df_allROIs = ROI_df
        else:
            df_allROIs = pd.concat([df_allROIs, ROI_df])

    # Convert summary to DataFrame
    df_summary = pd.DataFrame(summary_rows)

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 5))
    palette1 = ["slateblue", "goldenrod"]

    plot_errobars(
        df_allROIs,
        "n_voxels",
        "correct",
        "ROI",
        palette=palette1,
        x_offset=1,
        ax=ax,
        error_type=error_type
    )

    # Formatting
    ax.set_ylim(0.45, 0.7)
    ax.axhline(y=0.5, color="grey", linestyle="--")
    ax.set_xlabel("Number of Voxels", fontsize=16)
    ax.set_xticks(n_voxels_int)
    ax.set_xticklabels(n_voxels_list, rotation=45)
    ax.set_ylabel("Classification Accuracy", fontsize=16)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"figures/MVPA/decoding_modalities.svg", dpi=300, bbox_inches="tight")
    plt.show()

    return df_allROIs, df_summary

def plot_decoding_modalities_attention(data_path, n_voxels_list, ROIs = {"visual":"EVC", "auditory": "A1"}, error_type="sem", save_fig=False):
  

    # Convert to integers for plotting
    n_voxels_int = [int(n) for n in n_voxels_list[:-2]]
    n_voxels_int.extend([n_voxels_int[-1] + 50, n_voxels_int[-1] + 100])

    df_allROIs = pd.DataFrame()
    for ROI in ROIs.values():
        for i, n_voxels in zip(n_voxels_int, n_voxels_list):
            # Load and label data
            ROI_df = pd.read_csv(f"{data_path}{ROI}_{n_voxels}/balanced_group_data.csv")

            ROI_df = ROI_df.groupby(["subj", "modality", "n_voxels"], as_index=False)["correct"].mean()
            ROI_df["ROI"] = ROI
            ROI_df["n_voxels"] = i
            ROI_df["n_voxels_label"] = n_voxels

            df_allROIs = pd.concat([df_allROIs, ROI_df]) # Concatenate dataframes

    df_allROIs.rename(columns={"modality": "attended modality"}, inplace=True)


    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for i, ROI in enumerate(ROIs.values()):
        # Filter data for each ROI
        df_ROI = df_allROIs[df_allROIs["ROI"] == ROI]

        if i == 0: # Visual ROI
            palette = ["palegoldenrod", "goldenrod"]
        else:
            palette = ["slateblue", "lightsteelblue"]

        # Plotting
        plot_errobars(
            df_ROI,
            "n_voxels",
            "correct",
            "attended modality",
            palette=palette,
            x_offset=3,
            ax=ax[i],
            error_type=error_type
        )

        # Formatting
        ax[i].set_ylim(0.45, 0.7)
        ax[i].axhline(y=0.5, color="grey", linestyle="--")
        ax[i].set_xlabel("Number of Voxels", fontsize=16)
        ax[i].set_xticks(n_voxels_int)
        ax[i].set_xticklabels(n_voxels_list, rotation=45)
        ax[i].set_ylabel(f"Classification Accuracy in {ROI}", fontsize=16)
        ax[i].tick_params(labelleft=True)
        sns.despine(ax=ax[i])



    return df_allROIs

def plot_decoding_pred(data_path, ROI, n_voxels_list, hue, col, palette, attended_modality=None, error_type="ws", save_fig=False):
    """
    Plot decoding accuracy across ROI sizes.
    Parameters
    ----------
    df : DataFrame
        DataFrame containing decoding results.
    n_voxels_list : list
        List of voxel sizes to plot.
    hue : str
        Column name for hue in the plot. Prediction of the decoded modality.
    col : str
        Column name for columns in the plot. Prediction of the other modality
    error_type : str
        Type of error bar to use. Options are "ws", "sem" or "ci". For this analysis, "ws" is used, as we will test prediction effects within-subjects
    save_fig : bool
    """
    # Convert to integers for plotting
    n_voxels_int = [int(n) for n in n_voxels_list[:-2]]
    n_voxels_int.extend([n_voxels_int[-1] + 50, n_voxels_int[-1] + 100])

    ##summary_rows = []  # Collect stats here

    for i, n_voxels in zip(n_voxels_int, n_voxels_list):
        # Load and label data
        df = pd.read_csv(f"{data_path}{ROI}_{n_voxels}/balanced_group_data.csv")
        if attended_modality is not None:
            df = df[df["modality"] == attended_modality]
        dat = df.groupby(["subj", hue, col,], as_index=False)["correct"].mean()
        dat["n_voxels"] = i
        dat["n_voxels_label"] = n_voxels

        #for modality, roi in [("visual", visualROI), ("auditory", auditoryROI)]:
        # calculate stats

        # Combine dataframes across voxel sizes
        if n_voxels == n_voxels_list[0]:
            df_allROIs = dat
        else:
            df_allROIs = pd.concat([df_allROIs, dat])

    # Convert summary to DataFrame
    ##df_summary = pd.DataFrame(summary_rows)

    # Plotting
    fig, ax = plt.subplots(1,2, figsize=(10, 5), sharey=True)

    for i, col_val in enumerate(df_allROIs[col].unique()):
        df_sub = df_allROIs[df_allROIs[col] == col_val]
        plot_errobars(
            df_sub,
            "n_voxels",
            "correct",
            hue,
            palette=palette,
            x_offset=3,
            ax=ax[i],
            error_type=error_type
        )


    # Format both subplots
    for i, col_val in enumerate(df_allROIs[col].unique()):
        ax[i].set_ylim(0.45, 0.7)
        ax[i].axhline(y=0.5, color="grey", linestyle="--")
        ax[i].set_xlabel("Number of Voxels", fontsize=16)
        ax[i].set_xticks(n_voxels_int)
        ax[i].set_xticklabels(n_voxels_list, rotation=45)
        ax[i].set_title(f"{col} == {col_val}", fontsize=16)
        if i == 0:
            ax[i].set_ylabel(f"Classification Accuracy in {ROI}", fontsize=16)
        else:
            ax[i].set_ylabel(' ', fontsize=16)
            ax[i].tick_params(labelleft=True)
        
        sns.despine(ax=ax[i])


    plt.tight_layout()
    if save_fig:
        plt.savefig(f"figures/MVPA/decoding_modalities.svg", dpi=300, bbox_inches="tight")
    plt.show()

    return df_allROIs


def pointplot_morey(dat, x, y, hue, id_col="id", swarmplot_plot=True, palette=["mediumorchid", "forestgreen"], dodge_width=0.05, hline = 0.5, linestyle="-", linealpha = 1, ax=None):
    """
    Plots data with means and Cousineau-Morey corrected standard errors.

    Parameters:
    - dat: DataFrame containing the data
    - x: Column name for x-axis (e.g., 'stim')
    - y: Column name for y-axis (e.g., 'classified_stim')
    - hue: Column name for hue grouping (e.g., 'expected_stim')
    - id_col: Subject ID column for within-subject normalization (default: 'id')
    - swarmplot_plot: Boolean to include swarmplot overlay (default: True)
    - palette: List of colors for different hue levels (default: predefined colors)
    - dodge_width: Controls how much points are dodged horizontally (default: 0.05)
    - ax: Matplotlib Axes to plot on (if None, creates a new figure)
    """

    # Cousineau normalization
    dat['subject_mean'] = dat.groupby(id_col)[y].transform('mean')
    dat['cousineau_normalized'] = dat[y] - dat['subject_mean']
    grand_mean = dat[y].mean()
    dat['cousineau_normalized'] += grand_mean

    # Compute group-level mean and Cousineau-Morey corrected SE
    n_conditions = dat[x].nunique() * dat[hue].nunique()
    cousineau_summary = dat.groupby([x, hue]).agg(
        group_mean=('cousineau_normalized', 'mean'),
        corrected_se=('cousineau_normalized', lambda x: x.std() / np.sqrt(len(x)) * np.sqrt(n_conditions / (n_conditions - 1)))
    ).reset_index()

    # Plotting
    unique_expected = cousineau_summary[hue].unique()
    stim_categories = cousineau_summary[x].unique()
    stim_mapping = {stim: pos for pos, stim in enumerate(stim_categories)}

    # Use provided axis or create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Point plot
    sns.pointplot(
        data=cousineau_summary,
        x=x,
        y="group_mean",
        hue=hue,
        ax=ax,
        dodge=True,
        join=True,
        errorbar=None,
        palette=palette,
        linestyles=linestyle,
        alpha=linealpha
    )

    # Optional swarmplot
    if swarmplot_plot:
        sns.swarmplot(
            data=dat,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            dodge=True,
            palette=palette,
            alpha=0.5,
            legend=False
        )

    # Add custom error bars
    for j, expected in enumerate(unique_expected):
        color = palette[j]
        dat_sub = cousineau_summary[cousineau_summary[hue] == expected]

        for _, row in dat_sub.iterrows():
            # Adjust x-position due to dodge
            x_pos = stim_mapping[row[x]] - dodge_width/2 + j * dodge_width

            ax.errorbar(
                x=x_pos,
                y=row["group_mean"],
                yerr=row["corrected_se"],
                fmt='none',
                capsize=4,
                color=color,
                elinewidth=2
            )

    # Final plot adjustments
    ax.set_xlabel(x.capitalize())
    ax.set_ylabel(y.capitalize())
    ax.axhline(hline, ls="--", color="black")
    ax.set_xticks(range(len(stim_categories)))
    ax.set_xticklabels(stim_categories)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(title=hue)

    # Only call tight_layout if we created a new figure
    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_learning_interaction(data, x, y, hue, prob_stimulus, palette, hline, ax, num_ticks=5):
    """
    Plots a learning interaction with barplots. Looks good only if y is centered around 0.
    
    Parameters:
    - data: DataFrame containing the data
    - x: Column name for x-axis variable
    - y: Column name for y-axis variable (use centered probability)
    - prob_stimulus: Name of the positive probabilities stimulus
    - hue: Column name for hue (legend categories)
    - palette: List of colors for hue levels
    - hline: Horizontal reference line position (usually 0)
    - ax: Matplotlib axis object
    - num_ticks: Odd number of y-ticks (default is 5)
    """
    # Direct comparisons between expected stimuli (without taking into account the presented stimulus)
    sns.barplot(data=data, x=x, y=y, hue=hue, palette=palette, ci=95, ax=ax)

    # Set axis labels and title
    ax.set_xticks([0, 1])  # Set tick positions
    
    ax.set_ylabel(f'Probability of "{prob_stimulus}" classification')

    if x == "v_pred":
        ax.set_xlabel("Visual")
        ax.set_xticklabels(["UEX", "EXP"])  # Set tick labels
    elif x == "a_pred":
        ax.set_xlabel("Auditory")
        ax.set_xticklabels(["UEX", "EXP"])
    elif x == "v_learned":
        ax.set_xlabel("Visual prediction")
        ax.set_xticklabels(["not learned", "learned"])
    else:
        ax.set_xlabel("Auditory prediction")
        ax.set_xticklabels(["not learned", "learned"])

    # Set legend
    ax.legend(title="Expected Stimulus")

    # Add horizontal reference line
    ax.axhline(hline, color="black", linestyle="--")

    # Remove top/right spines
    sns.despine(ax=ax)

    # ** Generalizing Y-Axis Scaling **
    # Determine min/max centered probability values
    y_min, y_max = data[y].min(), data[y].max()

    # Ensure symmetry around zero
    y_abs_max = max(abs(y_min), abs(y_max))

    # Define tick step dynamically based on range
    tick_step = round((2 * y_abs_max) / (num_ticks - 1), 3)  # Ensure num_ticks is used

    # Define symmetric y-tick positions centered at 0
    y_ticks = np.linspace(-y_abs_max, y_abs_max, num_ticks)

    # Convert centered probability values back to original probability scale
    original_ticks = [tick + 0.5 for tick in y_ticks]

    # Set y-ticks and labels
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick*100:.0f}%" for tick in original_ticks])

    # Adjust y-limits dynamically with a small buffer
    ax.set_ylim(y_ticks[0] - tick_step * 0.5, y_ticks[-1] + tick_step * 0.5)


# WHOLE BRAIN PLOTTING

def plot_static_slices(
    cope,
    zstat,
    z_thresh,
    slice_positions,
    mni_template,
    axis="z",
    clim=(-30, 30),
    contour_level=2.3,
    z_lim=False,
    background="white",
    flip_axial=False,
    title="Static Slices",
):
    """
    Plot static slices with:
    - MNI brain template as the base layer.
    - Cope data color-coded with 'bwr'.
    - Alpha transparency based on zstat values.
    - Contours drawn around significant clusters in z_thresh.

    Parameters:
    - cope: 3D input (file path, Nifti1Image, or NumPy array) for cope values.
    - zstat: 3D input (file path, Nifti1Image, or NumPy array) for zstat values.
    - z_thresh: 3D input (file path, Nifti1Image, or NumPy array) for thresholded clusters.
    - mni_template: 3D input (file path, Nifti1Image, or NumPy array) for the MNI template.
    - slice_positions: List of slice indices along the specified axis.
    - axis: Axis to slice ('x', 'y', or 'z').
    - clim: Color limits for cope values.
    - contour_level: Level for contours based on z_thresh.
    - background: Background color ('white' or 'black').
    - flip_axial: Flip z-axis slices to have posterior bottom, anterior top.
    """
    # Validate and load inputs
    cope = validate_mri_input(cope)
    zstat = validate_mri_input(zstat)
    z_thresh = validate_mri_input(z_thresh)
    mni_template = validate_mri_input(mni_template)

    # Check for consistent dimensions
    if not (cope.shape == zstat.shape == z_thresh.shape == mni_template.shape):
        raise ValueError(
            "Input volumes (cope, zstat, z_thresh, mni_template) must have the same shape."
        )

    # Set figure background color
    if background == "black":
        facecolor = "black"
        textcolor = "white"
    elif background == "white":
        facecolor = "white"
        textcolor = "black"
    else:
        raise ValueError("Invalid background color. Choose 'white' or 'black'.")

    # Plotting
    n_slices = len(slice_positions)
    fig, axes = plt.subplots(
        1, n_slices + 1, figsize=(5 * n_slices, 5), facecolor=facecolor
    )  # +1 for colorbar

    # Ensure axes is always a list
    axes = np.ravel(axes).tolist()

    im = None  # Initialize im to ensure it is defined for colorbar
    for i, pos in enumerate(slice_positions):
        ax = axes[i]

        # Extract slices
        mni_slice = extract_slice(mni_template, axis, pos, flip_axial=flip_axial)
        cope_slice = extract_slice(cope, axis, pos, flip_axial=flip_axial)
        zstat_slice = extract_slice(zstat, axis, pos, flip_axial=flip_axial)
        z_thresh_slice = extract_slice(z_thresh, axis, pos, flip_axial=flip_axial)

        # Normalize zstat for alpha
        if z_lim:
            alpha = np.clip((zstat_slice - z_lim[0]) / (z_lim[1] - z_lim[0]), 0, 1)
        else:
            alpha = np.clip(
                (zstat_slice - np.min(zstat_slice))
                / (np.max(zstat_slice) - np.min(zstat_slice)),
                0,
                1,
            )

        # Plot MNI brain template as base
        ax.imshow(mni_slice, cmap="gray", origin="lower", interpolation="nearest")

        # Overlay cope values with alpha blending
        im = ax.imshow(cope_slice, cmap="bwr", alpha=alpha, origin="lower", clim=clim)

        # Draw contours around significant clusters in black
        ax.contour(z_thresh_slice, levels=[contour_level], colors="black", linewidths=2)

        ax.axis("off")
        ax.set_title(f"Slice {pos} ({axis})", color=textcolor)

    # # Add colorbar
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("bottom", size="20%", pad=0.5)  # Adjust size as needed
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("EXP - UEX", color=textcolor)

    if not z_lim:
        cbar.ax.axhline(
            y=contour_level / np.max(zstat_slice), color="black", linewidth=2
        )
    else:
        cbar.ax.axhline(y=contour_level / z_lim[1], color="black", linewidth=2)

    # Set colorbar text color
    cbar.ax.xaxis.set_tick_params(color=textcolor)
    cbar.ax.yaxis.set_tick_params(color=textcolor)
    plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color=textcolor)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=textcolor)
    plt.suptitle(title, color=textcolor)

    # Return the figure for saving
    return fig


def plot_3dims(
    cope, zstat, z_thresh, slice_positions, mni_template,
    clim=(-30, 30), contour_level=2.3, z_lim = False, flip_axial=False, title='Static Slices'
):
    """
    Plot static slices with:
    - MNI brain template as the base layer.
    - Cope data color-coded with 'bwr'.
    - Alpha transparency based on zstat values.
    - Contours drawn around significant clusters in z_thresh.

    Parameters:
    - cope: 3D input (file path, Nifti1Image, or NumPy array) for cope values.
    - zstat: 3D input (file path, Nifti1Image, or NumPy array) for zstat values.
    - z_thresh: 3D input (file path, Nifti1Image, or NumPy array) for thresholded clusters.
    - mni_template: 3D input (file path, Nifti1Image, or NumPy array) for the MNI template.
    - slice_positions: List of slice indices along the specified axis.
    - axis: Axis to slice ('x', 'y', or 'z').
    - clim: Color limits for cope values.
    - contour_level: Level for contours based on z_thresh.
    - background: Background color ('white' or 'black').
    - flip_axial: Flip z-axis slices to have posterior bottom, anterior top.
    """
    # Validate and load inputs
    cope = validate_mri_input(cope)
    zstat = validate_mri_input(zstat)
    z_thresh = validate_mri_input(z_thresh)
    mni_template = validate_mri_input(mni_template)

    # Check for consistent dimensions
    if not (cope.shape == zstat.shape == z_thresh.shape == mni_template.shape):
        raise ValueError("Input volumes (cope, zstat, z_thresh, mni_template) must have the same shape.")


    # Plotting
    n_slices = len(slice_positions)
    fig, axes = plt.subplots(1, 3, figsize=(5 * n_slices, 5)) # +1 for colorbar
    dims = ['x', 'y', 'z']

    for i, pos in enumerate(slice_positions):
        ax = axes[i]

        # Extract slices
        mni_slice = extract_slice(mni_template, dims[i], pos, flip_axial=flip_axial)
        cope_slice = extract_slice(cope, dims[i], pos, flip_axial=flip_axial)
        zstat_slice = extract_slice(zstat, dims[i], pos, flip_axial=flip_axial)
        z_thresh_slice = extract_slice(z_thresh, dims[i], pos, flip_axial=flip_axial)

        # Normalize zstat for alpha
        if z_lim:
            alpha = np.clip((zstat_slice - z_lim[0]) / (z_lim[1] - z_lim[0]), 0, 1)
        else:
            alpha = np.clip((zstat_slice - np.min(zstat_slice)) / (np.max(zstat_slice) - np.min(zstat_slice)), 0, 1)

        # Plot MNI brain template as base
        ax.imshow(mni_slice, cmap='gray', origin='lower', interpolation='nearest')

        # Overlay cope values with alpha blending
        im = ax.imshow(cope_slice, cmap='bwr', alpha=alpha, origin='lower', clim=clim)
        
        # Draw contours around significant clusters in black
        ax.contour(z_thresh_slice, levels=[contour_level], colors='black', linewidths=2)

        ax.axis('off')
        ax.set_title(f"Slice {pos} ({dims[i]})")

        plt.suptitle(title)


    # Return the figure for saving
    return fig