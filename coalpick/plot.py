"""
Plotting functionality of coalpick.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_residuals(
    predictions: dict, target: np.ndarray, sr, output_path=None, colors=None, rng=10
):
    """
    Plot the pick time residuals (predicted - analyst) and statistics for each set of predictions.

    Parameters
    ----------
    predictions
        Dictionary on {name: prediction_array, ...}
    target
        Targeted array
    sr
        Sampling rate of the given dataset
    output_path
        path to save the figure to
    colors
        list of colors to use for each plot
    rng
        +/-x axis range (in samples) (i.e. if rng=10, the x axis will be -10 -> 10)
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

    def _get_stats(predictions, target):
        stats = {}
        for pkr, pred in predictions.items():
            res = pred - target
            clean = _remove_outliers(res)
            abs_res = abs(res)
            temp_stats = dict(
                mean=clean.mean(),
                std=clean.std(),
                q75=np.quantile(abs_res, 0.75),
                q90=np.quantile(abs_res, 0.9),
            )
            stats[pkr] = temp_stats
        return stats

    def _subplot_hists(ax, res, color, bins, stats):
        ax.grid(True)

        # plotting hist
        n, bins, patches = ax.hist(res, alpha=0.35, color=color, bins=bins)

        # # plotting line
        hists, _bins = np.histogram(res, bins=bins)
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

    if colors is None:
        colors = ["#1f77b4", "#ff7f0e", "#8c564b"]
    stats = _get_stats(predictions, target)

    fig = plt.figure(figsize=(3.5, 1.75 * len(predictions)))
    bins = np.arange(-rng, rng + 1)

    for cnt, (pkr, pred) in enumerate(predictions.items()):
        res = pred - target

        color = colors[cnt]
        pkr_stats = stats[pkr]

        if cnt == 0:
            master_ax = fig.add_subplot(len(predictions), 1, 1)
            _subplot_hists(master_ax, res, color, bins, pkr_stats)
            plt.setp(master_ax.get_xticklabels(), visible=False)

            master_ax.set_ylabel(pkr)
        else:
            ax = fig.add_subplot(len(predictions), 1, cnt + 1, sharey=master_ax)
            _subplot_hists(ax, res, color, bins, pkr_stats)

            ax.set_ylabel(pkr)

            if cnt == len(predictions) - 1:
                # addings second axis
                ax.set_xlabel("samples")
                ax2 = ax.twiny()
                new_pos = np.arange(-rng, rng + 1, rng / 2)
                new_labels = [n / sr for n in new_pos]
                ax2.set_xticks(new_pos)
                ax2.set_xticklabels(new_labels)
                ax2.xaxis.set_ticks_position("bottom")
                ax2.xaxis.set_label_position("bottom")
                ax2.spines["bottom"].set_position(("outward", 36))
                ax2.set_xlabel("time (s)")
                ax2.set_xlim(ax.get_xlim())
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_path)
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
    ax.set_xlabel("epoch")
    ax.legend()
    # Save output
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)
    else:
        return fig
