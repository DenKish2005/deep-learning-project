"""Generate Figures 02-06 from the six frozen core runs and their test results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from scripts.analyze_per_class import FAMILIES, LABELS, MODELS, ROOT, SEEDS, read_results, sha256
from src.metrics import aggregate_seeds


REPORT = ROOT / 'report/core_v1'
OUTPUT = ROOT / 'outputs/figures'
COLORS = dict(zip(MODELS, ('#3569a6', '#cf6d27')))
SEED_MARKERS = ('o', '^', 's')


def read_inputs():
    raw_path = REPORT / 'raw_results.csv'
    raw = read_results(raw_path)
    tables = aggregate_seeds(raw)
    inputs = [raw_path]
    summaries = {}
    for name, keys in [('condition_summary', ['model_name', 'corruption', 'severity']),
                       ('run_summary', ['model_name', 'seed']), ('model_summary', ['model_name'])]:
        path = REPORT / 'tables' / f'{name}.csv'
        actual = tables[name].set_index(keys).sort_index()
        saved = pd.read_csv(path).set_index(keys).sort_index()
        pd.testing.assert_frame_equal(actual, saved, atol=1e-14, rtol=0, check_dtype=False)
        summaries[name] = actual
        inputs.append(path)

    histories = {}
    for (model, seed), rows in raw.groupby(['model_name', 'seed']):
        run_id = rows.run_id.unique().item()
        folder = ROOT / 'outputs/checkpoints/core-v1' / run_id
        history_path, manifest_path = folder / 'history.csv', folder / 'run_manifest.json'
        history = pd.read_csv(history_path, float_precision='round_trip').set_index('epoch')
        manifest = json.loads(manifest_path.read_text())
        best = manifest['best_epoch']
        selected = history.sort_values(['val_accuracy', 'val_loss', 'epoch'],
                                       ascending=[False, True, True]).index[0]
        if not (history.index.tolist() == list(range(1, 41))
                and history.model_name.eq(model).all() and history.seed.eq(seed).all()
                and history.experiment_id.eq(run_id).all()
                and (manifest['model_name'], manifest['seed'], manifest['run_id']) == (model, seed, run_id)
                and rows.checkpoint_sha256.eq(manifest['artifacts']['best.pt']).all()
                and sha256(history_path) == manifest['artifacts']['history.csv']
                and np.isfinite(history[['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']]).all().all()
                and best == history.best_epoch.iloc[-1] == selected):
            raise ValueError(f'Inconsistent history or checkpoint selection in {folder}')
        histories[model, seed] = history, best
        inputs.extend([history_path, manifest_path])
    return histories, summaries, inputs


def panels(titles, xlabel, ylabel):
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.85), sharey=True, layout='constrained')
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel=xlabel)
    axes[0].set_ylabel(ylabel)
    return fig, axes


def history_plot(histories, metric):
    accuracy = metric == 'accuracy'
    fig, axes = panels([f'Seed {s}' for s in SEEDS], 'Epoch',
                       'Accuracy (%)' if accuracy else 'Cross-entropy loss')
    for ax, seed in zip(axes, SEEDS):
        for model in MODELS:
            frame, best = histories[model, seed]
            for split, style in [('train', '-'), ('val', '--')]:
                values = frame[f'{split}_{metric}'] * (100 if accuracy else 1)
                ax.plot(frame.index, values, style, color=COLORS[model],
                        label=f'{LABELS[model]} · {split}')
            if accuracy:
                ax.plot(best, frame.at[best, 'val_accuracy'] * 100, marker='D',
                        ms=6, color=COLORS[model], mec='white', mew=.6)
        ax.set_xticks([1, 10, 20, 30, 40])
    handles, labels = axes[0].get_legend_handles_labels()
    if accuracy:
        handles.append(Line2D([], [], marker='D', linestyle='none', color='#444444'))
        labels.append('Selected checkpoint')
    fig.legend(handles, labels, loc='outside lower center', ncols=len(labels))
    return fig


def severity_plot(conditions, models):
    fig, axes = panels([LABELS[f] for f in FAMILIES], 'Severity', 'Test accuracy (%)')
    for ax, family in zip(axes, FAMILIES):
        for model in models:
            keys = [(model, 'clean', 0)] + [(model, family, level) for level in range(1, 6)]
            points = conditions.loc[keys] * 100
            mean, sd = points['mean'].to_numpy(), points.sample_sd.to_numpy()
            x = range(6)
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[model], alpha=.20)
            ax.plot(x, mean, marker='o' if model == MODELS[0] else '^', ms=4,
                    color=COLORS[model], label=LABELS[model])
        ax.set(xticks=range(6), ylim=(0, 100))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles + [Patch(facecolor='#bbbbbb')], labels + ['±1 sample SD'],
               loc='outside lower center', ncols=len(labels) + 1)
    return fig


def scatter_plot(runs, means):
    fig, ax = plt.subplots(figsize=(7.6, 4.2), layout='constrained')
    for model in MODELS:
        for seed, marker in zip(SEEDS, SEED_MARKERS):
            row = runs.loc[model, seed]
            ax.scatter(row.clean_accuracy * 100, row.mca * 100, marker=marker,
                       color=COLORS[model], s=47)
        row = means.loc[model]
        ax.scatter(row.clean_accuracy_mean * 100, row.mca_mean * 100, marker='*',
                   color=COLORS[model], edgecolors='#252525', linewidth=.7, s=170)
    ax.set(xlabel='Clean-test accuracy (%)', ylabel='Mean corrupted accuracy (%)')
    ax.margins(x=.10, y=.12)
    handles = [Patch(color=COLORS[m], label=LABELS[m]) for m in MODELS]
    for marker, label in zip((*SEED_MARKERS, '*'), [*(f'Seed {s}' for s in SEEDS), 'Model mean']):
        handles.append(Line2D([], [], marker=marker, color='#444444', linestyle='none',
                              markersize=10 if marker == '*' else 6, label=label))
    fig.legend(handles=handles, loc='outside right center')
    return fig


def main():
    histories, summaries, inputs = read_inputs()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    outputs = []
    with plt.rc_context({'font.size': 11, 'legend.fontsize': 10, 'legend.frameon': False,
                         'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
                         'grid.alpha': .22, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42}):
        conditions = summaries['condition_summary']
        figures = {'loss_curves': history_plot(histories, 'loss'),
                   'accuracy_curves': history_plot(histories, 'accuracy'),
                   'baseline_severity': severity_plot(conditions, MODELS[:1]),
                   'clean_vs_robust': severity_plot(conditions, MODELS),
                   'clean_vs_mca': scatter_plot(summaries['run_summary'], summaries['model_summary'])}
        for name, fig in figures.items():
            for ext in ('png', 'pdf'):
                path = OUTPUT / f'{name}.{ext}'
                metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None
                fig.savefig(path, dpi=300, metadata=metadata)
                outputs.append(path)
            plt.close(fig)

    def hashes(paths):
        return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}

    manifest = {'figures': list(figures), 'inputs_sha256': hashes(inputs),
                'scripts_sha256': hashes([Path(__file__).resolve(), ROOT / 'src/metrics.py',
                                          ROOT / 'scripts/analyze_per_class.py']),
                'outputs_sha256': hashes(outputs),
                'selection': 'Manifest best epoch: highest clean validation accuracy, then lowest loss, then earliest epoch.',
                'aggregation': 'Per-seed histories; severity 0 = clean, condition mean +/- sample SD (ddof=1) over three seeds; per-run mCA = mean of 15 corrupted accuracies, then mean over seeds.'}
    (OUTPUT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
