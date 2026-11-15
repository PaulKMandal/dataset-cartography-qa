#!/usr/bin/env python3
"""Generate the paper's data-map, overlay, and grid figures.

This is the canonical figure generator for the EACL 2027 Findings paper
(Ambiguous Examples Help, But Full Data Wins). It supersedes earlier
Allen-style and region-box map styles.

Conventions (final, matching the published figures):
  * Colors encode correctness categories: green = 100%,
    amber >= 50% and < 100%, red < 50% (historical thresholds; the
    `region` column of cartography_scores.csv encodes the same rule).
  * No region annotations are drawn on the maps; qualitative
    easy-to-learn / ambiguous / hard-to-learn regions are described in
    captions only, attributed to Swayamdipta et al. (2020).
  * Overlay panels are titled Easy / Ambiguous / Hard after the globally
    ranked 33% training subsets (top confidence / top variability /
    bottom confidence over all 87,599 grouped examples); highlights are
    subset IDs intersected with one fixed 10K sample (seed 1009), so
    highlight counts vary by panel and colors mix within panels.
  * All scatter panels share the main map's box aspect (h/w = 0.881) and
    dot size proportional to panel width (diameter = 0.764% of panel
    width), with the map's alpha (0.62) for colored points.

Usage:
  python scripts/make_paper_figures.py \
      --scores electra_small=path/to/cartography_scores.csv \
               bert_base=... [...] \
      --out-dir figures_out
The first panel listed is treated as the headline model (individual map
named after it); all panels appear in the grid.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

SEED, N_SAMPLE, FRACTION = 1009, 10000, 0.333
REGION_ORDER = ['Easy-to-learn', 'Ambiguous', 'Hard-to-learn']
REGION_COLORS = {'Easy-to-learn': '#1B9E3C', 'Ambiguous': '#F2B01E', 'Hard-to-learn': '#E31A1C'}
CORRECTNESS_LABEL = {'Easy-to-learn': '= 100%', 'Ambiguous': u'≥ 50%, < 100%', 'Hard-to-learn': '< 50%'}
PANEL_HEADINGS = [('easy', 'Easy'), ('ambiguous', 'Ambiguous'), ('hard', 'Hard')]
BOX_ASPECT = 0.881          # main map's scatter-box height/width
PLOT_ORDER = ['Hard-to-learn', 'Ambiguous', 'Easy-to-learn']
# dot sizes derived from dot-diameter/panel-width = 0.00764 (see paper repo)
S_MAP, S_OVERLAY_HL, S_OVERLAY_BG, S_GRID = 7.5, 3.75, 2.81, 1.95


def correctness_handles():
    return [mpatches.Patch(color=REGION_COLORS[r], label=CORRECTNESS_LABEL[r]) for r in REGION_ORDER]


def ensure_region(df):
    if 'region' not in df.columns:
        c = df['correctness']
        df['region'] = np.where(c >= 1.0, 'Easy-to-learn',
                        np.where(c >= 0.5, 'Ambiguous', 'Hard-to-learn'))
    return df


def correctness_bin(x):
    for v, lab in ((0, '0'), (1/3, '1/3'), (2/3, '2/3'), (1, '1')):
        if np.isclose(x, v, atol=2e-3):
            return lab
    return 'other'


def draw_map(df, sample, model_name, out_png):
    xmax = max(0.50, float(df['variability'].quantile(.995)) * 1.04)
    fig = plt.figure(figsize=(9.6, 6.4))
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.88], wspace=0.34, hspace=0.55)
    ax = fig.add_subplot(gs[:, :3])
    for region in PLOT_ORDER:
        sub = sample[sample['region'] == region]
        if len(sub):
            ax.scatter(sub['variability'], sub['confidence'], s=S_MAP, alpha=0.62,
                       c=REGION_COLORS[region], linewidths=0, rasterized=True)
    ax.set_xlabel('variability')
    ax.set_ylabel('confidence')
    ax.set_title(f'{model_name} SQuAD Data Map')
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02)
    ax.grid(True, linewidth=0.45, alpha=0.35)
    ax.legend(handles=correctness_handles(), title='correctness', loc='upper right',
              frameon=True, fontsize=8.5, title_fontsize=9)
    ax1 = fig.add_subplot(gs[0, 3]); ax1.hist(sample['confidence'], bins=35, color='#6A51A3')
    ax1.set_title('confidence', fontsize=9); ax1.set_ylabel('count', fontsize=8); ax1.tick_params(labelsize=7)
    ax2 = fig.add_subplot(gs[1, 3]); ax2.hist(sample['variability'], bins=35, color='#008B8B')
    ax2.set_title('variability', fontsize=9); ax2.set_ylabel('count', fontsize=8); ax2.tick_params(labelsize=7)
    ax3 = fig.add_subplot(gs[2, 3])
    counts = sample['correctness'].apply(correctness_bin).value_counts().reindex(['0', '1/3', '2/3', '1', 'other']).fillna(0)
    ax3.bar(counts.index, counts.values, color='#86BC8F')
    ax3.set_title('correctness', fontsize=9); ax3.set_ylabel('count', fontsize=8); ax3.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches='tight')
    plt.close(fig)


def draw_overlay(df, sample, model_name, out_png):
    n = int(round(FRACTION * len(df)))
    subsets = {'easy': set(df.nlargest(n, 'confidence')['idx']),
               'ambiguous': set(df.nlargest(n, 'variability')['idx']),
               'hard': set(df.nsmallest(n, 'confidence')['idx'])}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), sharex=True, sharey=True)
    for ax, (key, heading) in zip(axes, PANEL_HEADINGS):
        ax.scatter(sample['variability'], sample['confidence'], c='#d9d9d9', s=S_OVERLAY_BG,
                   alpha=.5, linewidths=0, rasterized=True, zorder=1)
        sel = sample[sample['idx'].isin(subsets[key])]
        for region in PLOT_ORDER:
            sub = sel[sel['region'] == region]
            if len(sub):
                ax.scatter(sub['variability'], sub['confidence'], c=REGION_COLORS[region],
                           s=S_OVERLAY_HL, alpha=.62, linewidths=0, rasterized=True, zorder=2)
        ax.set_box_aspect(BOX_ASPECT)
        ax.set_title(heading, fontsize=15)
        ax.set_xlabel('variability', fontsize=13)
        ax.set_xlim(0, .5); ax.set_ylim(-.005, 1.02)
        ax.tick_params(labelsize=11)
        ax.grid(True, linewidth=0.45, alpha=0.35)
    axes[0].set_ylabel('confidence', fontsize=13)
    fig.suptitle(f'{model_name} training-subset overlays', fontsize=16, y=.99)
    fig.legend(handles=[mpatches.Patch(color='#d9d9d9', label='unselected')], loc='lower left',
               bbox_to_anchor=(.12, -.015), frameon=False, fontsize=12)
    fig.legend(handles=correctness_handles(), title='correctness of highlighted points',
               loc='lower center', ncol=3, frameon=False, fontsize=12, title_fontsize=12,
               bbox_to_anchor=(.55, -.035))
    fig.subplots_adjust(left=.06, right=.985, bottom=.21, top=.86, wspace=.08)
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def draw_grid(samples, out_png):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10.5), sharey=True)
    axes = axes.ravel()
    for i, (ax, (model_name, df, sample)) in enumerate(zip(axes, samples)):
        for region in PLOT_ORDER:
            sub = sample[sample['region'] == region]
            if len(sub):
                ax.scatter(sub['variability'], sub['confidence'], s=S_GRID, alpha=.62,
                           c=REGION_COLORS[region], linewidths=0, rasterized=True)
        ax.set_box_aspect(BOX_ASPECT)
        ax.set_title(model_name, fontsize=14)
        ax.set_xlabel('variability', fontsize=12)
        if i % 3 == 0:
            ax.set_ylabel('confidence', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_ylim(0, 1.02); ax.set_xlim(0, max(0.5, float(df['variability'].quantile(.995)) * 1.04))
        ax.grid(True, linewidth=0.3, alpha=0.25)
    for ax in axes[len(samples):]:
        ax.axis('off')
    axes[-1].legend(handles=correctness_handles(), title='correctness', loc='center',
                    frameon=False, fontsize=13, title_fontsize=14)
    fig.suptitle('Model-specific SQuAD data maps from raw training dynamics', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out_png, dpi=190, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scores', nargs='+', required=True, metavar='NAME=CSV',
                   help='model display name = path to cartography_scores.csv')
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for spec in a.scores:
        name, _, path = spec.partition('=')
        df = ensure_region(pd.read_csv(path))
        df['idx'] = df['idx'].astype(int)
        sample = df.sample(n=min(N_SAMPLE, len(df)), random_state=SEED).reset_index(drop=True)
        slug = name.replace('-', '_').replace(' ', '_').lower()
        draw_map(df, sample, name, a.out_dir / f'{slug}_map.png')
        draw_overlay(df, sample, name, a.out_dir / f'{slug}_overlay.png')
        samples.append((name, df, sample))
        print(f'{name}: map + overlay written')
    draw_grid(samples, a.out_dir / 'all_model_maps.png')
    print('grid written')


if __name__ == '__main__':
    main()
