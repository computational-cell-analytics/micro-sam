import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

# Path to your CSV
CSV_PATH = "./preliminary_tryouts.csv"  # or ./preliminary_tryouts_fixed.csv if you used the fixed one

# Use only mSA for now
METRIC_COLS = ["mSA"]
# Examples:
# METRIC_COLS = ["F1"]
# METRIC_COLS = ["mSA", "F1"]   # stacked bars of two metrics


# -------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# Ensure metric columns exist and are numeric
for col in METRIC_COLS:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")
    df[col] = pd.to_numeric(df[col], errors="coerce")


# -------------------------------------------------------------------
# Plotting function
# -------------------------------------------------------------------

def plot_experiments_bar(df, metric_cols):
    """
    Create one big figure with a grid of subplots:
    - one subplot per experiment (DSB, U20S, ...)
    - x-axis: methods
    - y-axis: selected metric(s)
      * if len(metric_cols) == 1: simple bar chart
      * if len(metric_cols) > 1: stacked bars using those metrics
    """
    # Drop rows that have all selected metrics as NaN
    df_plot = df.dropna(subset=metric_cols, how="all")

    experiments = df_plot["Experiment"].unique()
    n_exp = len(experiments)

    # Layout of subplots (tweak as you like)
    ncols = 3
    nrows = int(np.ceil(n_exp / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharey=True
    )

    # Normalize axes to a flat array
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    # Loop with index so we can know which column we're in
    for idx, (ax, exp) in enumerate(zip(axes, experiments)):
        sub = df_plot[df_plot["Experiment"] == exp]

        methods = sub["Method"].values
        x = np.arange(len(methods))

        # For multiple metrics we stack them; for single metric it's just a normal bar
        bottom = np.zeros(len(methods))
        for i, metric in enumerate(metric_cols):
            values = sub[metric].values

            ax.bar(
                x,
                values,
                bottom=bottom,
                label=metric if exp == experiments[0] else None  # only label once
            )

            bottom += values

        ax.set_title(exp)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)

        # First column axes: idx % ncols == 0
        if idx % ncols == 0:
            if len(metric_cols) == 1:
                ax.set_ylabel(metric_cols[0])
            else:
                ax.set_ylabel("Sum of metrics")

    # Remove unused axes if grid has more slots than experiments
    for j in range(len(experiments), len(axes)):
        fig.delaxes(axes[j])

    # Shared legend (only if we actually have labeled bars)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(metric_cols))

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, axes


# -------------------------------------------------------------------
# Run plotting
# -------------------------------------------------------------------

if __name__ == "__main__":
    fig, axes = plot_experiments_bar(df, METRIC_COLS)
    plt.savefig("./test.png", bbox_inches="tight", dpi=300)
