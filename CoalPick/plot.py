"""
Plotting functionality of CoalPick.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_residuals(residuals, output_path=None):
    """
    Plot the pick time residuals (predicted - analyst) and statistics.
    """

    def _remove_outliers(residuals):
        """Remove outliers with outer fence method."""
        q_25 = np.quantile(residuals, 0.25)
        q_75 = np.quantile(residuals, 0.75)
        inter_quantile_range = q_75 - q_25
        outer_low = q_25 - 1.5 * inter_quantile_range
        outer_high = q_75 + 1.5 * inter_quantile_range
        too_high = residuals > outer_high
        too_low = residuals < outer_low
        return residuals[~(too_high | too_low)]

    def _subplot_hists(ax, clean, color="b", stats=None):
        ax.grid(True)

        # plotting hist
        n, bins, patches = ax.hist(clean, alpha=0.35, color=color,
                                   bins=np.arange(int(min(clean) - 1), int(max(clean) + 2)))
        ax.set_ylabel("Count")

        # # plotting line
        hists, _bins = np.histogram(clean, bins=bins)
        assert np.array_equal(bins, _bins)
        xs = np.repeat(bins, 2)[1:-1]
        ys = np.repeat(hists, 2)
        ax.plot(xs, ys, color=color)

        if stats is not None:
            # getting stats
            mean = stats["mean"]
            std = stats["std"]
            q75 = stats["q75"]
            q90 = stats["q90"]

            # plot text
            l_txt = r"$\mu=$" + f"{mean:0.3f}\n" + r"$\sigma=$" + f"{std:0.3f}"
            r_txt = r"$Q_{|75|}=$" + f"{q75:0.3f}\n" + r"$Q_{|90|}=$" + f"{q90:0.3f}"
            props = dict(boxstyle="round", alpha=0.75, facecolor="white")
            ax.text(
                0.03,
                0.93,
                l_txt,
                fontsize=10,
                bbox=props,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
            )
            ax.text(
                0.975,
                0.94,
                r_txt,
                fontsize=10,
                bbox=props,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
            )

    # Use outer fence to remove outliers and get abs of residuals
    clean = _remove_outliers(residuals)
    abs_residuals = abs(residuals)

    # get stats used in paper
    stats = dict(
        std=np.std(clean),
        mean=np.mean(clean),
        q75=np.quantile(abs_residuals, 0.75),
        q90=np.quantile(abs_residuals, 0.90),
    )

    fig, ax = plt.subplots(1, 1)
    _subplot_hists(ax, clean, stats=stats)
    ax.set_xlabel("Pick Residuals (samples)")
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)
    else:
        return fig


def plot_waveforms(waveform, picks, output_path=None, buffer=30):
    """Plot zoomed in waveform and the picks made by various models. """
    picks = pd.Series(picks)
    start_sample = int(max(picks.min() - buffer, 0))
    end_sample = int(min(picks.max() + buffer, len(waveform)))
    x = np.arange(start_sample, end_sample)
    y = waveform[start_sample:end_sample]
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    pick_colors = ["r", "g", "k", "c"]
    for (name, pick), color in zip(picks.items(), pick_colors):
        ax.axvline(pick, label=name, color=color)
    ax.legend()
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)
    else:
        return fig


def plot_training(history, output_path=None):
    """Plots the training history."""
    # Plot each metric provided by history.
    fig, ax = plt.subplots(1, 1)
    for metric_name, metric in history.items():
        epoch = range(1, len(metric) + 1)
        ax.plot(epoch, metric, label=metric_name)
    ax.set_xlabel('epoch')
    ax.legend()
    # Save output
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)
    else:
        return fig

