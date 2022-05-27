import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_training(
        trainer,
        test_set_labels=None,
        window=1,
        log_scale=False,
        plot_maxs=True,
        plot_means=False,
        plot_solve_steps=False,
        y_lims=(None, None),
):
    steps = np.array([x for x, _ in trainer.log['scores']])
    scores = torch.tensor([x for _, x in trainer.log['scores']]).permute(1, 2, 0).numpy()
    scores = scores
    if test_set_labels is None:
        test_set_labels = [f"graph set {idx}" for idx in range(len(scores))]

    if log_scale:
        y_label = "log(1-approx. ratio)"
    else:
        y_label = "approx. ratio"
    sns_cols = sns.color_palette("colorblind", desat=1)

    def process(data):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    steps = process(steps)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    if plot_solve_steps:
        ax_inset = ax.inset_axes([0.65, 0.05, 0.3, 0.3])

    with sns.plotting_context("paper"):

        for idx, scores in enumerate(scores):
            best_scores, mean_scores, best_scores_step, mean_init_score = scores[0], scores[1], scores[2], scores[3]
            max_best_score, max_mean_score = max(best_scores), max(mean_scores)

            if log_scale:
                best_scores, max_best_score = 1 - best_scores, 1 - max_best_score
                mean_scores, max_mean_score = 1 - mean_scores, 1 - max_mean_score

            if plot_maxs:
                plt.plot(steps, process(best_scores),
                         label=test_set_labels[idx],
                         color=sns_cols[idx])

            if plot_means:
                ax.plot(steps, process(mean_scores),
                        label=f"{test_set_labels[idx]} (mean)",
                        color=sns_cols[idx],
                        linestyle="--")

            if plot_solve_steps:
                num_nodes = trainer.test_graph_configs[idx].graphs[0].number_of_nodes()
                ax_inset.plot(steps, process(best_scores_step) / num_nodes,
                              color=sns_cols[idx],
                              linestyle="-")
                ax_inset.set_xticks([])
                ax_inset.set_title("Best step / |V|")

            plt.hlines(max_best_score, min(steps), max(steps), linestyle="--", linewidth=0.5, alpha=0.75,
                       color=sns_cols[idx])
            if log_scale:
                ax.legend(loc=4, bbox_to_anchor=(1.45, 0.7))
                ax.text(max(steps), max_best_score, f"apx. rt. = {1 - max_best_score:.3f}")
            else:
                ax.legend(loc=4, bbox_to_anchor=(1.45, 0))
                ax.text(max(steps), max_best_score, f"apx. rt. = {max_best_score:.3f}")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Steps")
    ax.set_ylabel(y_label)

    sns.despine()
    plt.autoscale()

    ax.set_ylim(bottom=y_lims[0], top=y_lims[1])

    return fig