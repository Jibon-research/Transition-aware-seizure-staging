"""
========================================================================================
COMBINED 1x2 RADAR COMPARISON: ALPHA BAND VS. BROADBAND (5-STAGE CLINICAL RADAR)
========================================================================================
Combines the individual Alpha Band and BroadBand per-class radar charts into a unified
1x2 side-by-side publication figure:
- Panel (a): Focused Alpha Band (8–13 Hz) - 6 Deep Architectures
- Panel (b): BroadBand EEG (0.5–40 Hz) - 6 Deep Architectures

Outputs:
1. Master Rendered 1x2 Figure (Elsevier typography, bold ST-ATMR-Net highlight, shared legend):
   - 300 DPI: Figure_Combined_Alpha_vs_BroadBand_Per_Class_Radar_Comparison.png
   - 600 DPI: Figure_Combined_Alpha_vs_BroadBand_Per_Class_Radar_Comparison_600DPI.png
2. Direct 1x2 Lossless Stitch of the two original PNG figures with subpanel labels:
   - 300 DPI: Figure_Combined_Alpha_vs_BroadBand_Radar_Direct_Stitch_1x2.png
========================================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------------------
# Directories & Styling Configurations
# --------------------------------------------------------------------------------------
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

ROOT_DIR = str(PROJECT_ROOT)
CODE_DIR = str(CODE_DIR)
ALPHA_JSON_PATH = os.path.join(ROOT_DIR, "05 CSV_and_TXT", "Model_Benchmarks_and_Ablations", "Alpha_Band", "Alpha_Band_Metrics.json")
BROAD_JSON_PATH = os.path.join(ROOT_DIR, "05 CSV_and_TXT", "Model_Benchmarks_and_Ablations", "BroadBand", "BroadBand_Metrics.json")

SUPP_FIG_DIR = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures", "Result_Oriented_Figures", "Supplementary_Result_Figures")
SIX_MODEL_DIR = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures", "Result_Oriented_Figures", "Six_Model_Comparative_Diagnostics")
PUB_ROOT_DIR = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures")

os.makedirs(SUPP_FIG_DIR, exist_ok=True)
os.makedirs(SIX_MODEL_DIR, exist_ok=True)
os.makedirs(PUB_ROOT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12.0
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 9.5

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']

MODELS_STYLE = [
    {
        'key': 'Standard_LSTM',
        'label': 'Standard LSTM',
        'color': '#7f7f7f',    # Slate Muted Gray
        'linestyle': '--',
        'linewidth': 1.6,
        'marker': 'o',
        'markersize': 5,
        'alpha_fill': 0.02
    },
    {
        'key': 'BiLSTM',
        'label': 'BiLSTM',
        'color': '#2ca02c',    # Clinical Green
        'linestyle': '-.',
        'linewidth': 1.7,
        'marker': 's',
        'markersize': 5,
        'alpha_fill': 0.02
    },
    {
        'key': 'TriLSTM',
        'label': 'TriLSTM',
        'color': '#d62728',    # Crimson
        'linestyle': ':',
        'linewidth': 1.8,
        'marker': '^',
        'markersize': 5,
        'alpha_fill': 0.03
    },
    {
        'key': 'TCN_1',
        'label': 'TCN-1',
        'color': '#ff7f0e',    # Warm Amber
        'linestyle': '-',
        'linewidth': 1.9,
        'marker': 'v',
        'markersize': 5,
        'alpha_fill': 0.03
    },
    {
        'key': 'TCN_2',
        'label': 'TCN-2',
        'color': '#9467bd',    # Muted Purple
        'linestyle': '-',
        'linewidth': 2.0,
        'marker': 'D',
        'markersize': 5,
        'alpha_fill': 0.04
    },
    {
        'key': 'STATMR_Net_Proposed',
        'label': 'ST-ATMR-Net (Proposed)',
        'color': '#0b5394',    # Deep Royal Navy (Elsevier Blue)
        'linestyle': '-',
        'linewidth': 3.2,
        'marker': '*',
        'markersize': 10,
        'alpha_fill': 0.18
    }
]

def load_metrics():
    with open(ALPHA_JSON_PATH, 'r') as f:
        alpha_data = json.load(f)
    with open(BROAD_JSON_PATH, 'r') as f:
        broad_data = json.load(f)
    return alpha_data, broad_data

def plot_radar_axis(ax, data_dict, metric_key, panel_title, subtitle=None):
    """Plots a single polar radar chart for 6 architectures."""
    num_vars = len(STAGE_NAMES)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close polygon

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid circles
    ticks = [20, 40, 60, 80, 100]
    ax.set_rticks(ticks)
    ax.set_yticklabels([f"{t}%" for t in ticks], fontsize=8.5, color='#555555')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", color='#bbbbbb', alpha=0.7)

    # Category spokes
    ax.set_thetagrids(np.degrees(angles[:-1]), STAGE_NAMES, fontsize=10.5, fontweight='bold', color='#111111')

    # Draw model polygons
    for m_info in MODELS_STYLE:
        m_key = m_info['key']
        if m_key not in data_dict:
            continue
        per_class = data_dict[m_key]['test_metrics']['per_class']
        vals = [per_class[s][metric_key] * 100 for s in STAGE_NAMES]
        vals += vals[:1]  # Close polygon

        ax.plot(
            angles, vals,
            label=m_info['label'],
            color=m_info['color'],
            linestyle=m_info['linestyle'],
            linewidth=m_info['linewidth'],
            marker=m_info['marker'],
            markersize=m_info['markersize'],
            zorder=5 if 'Proposed' in m_info['label'] else 3
        )
        if m_info['alpha_fill'] > 0:
            ax.fill(angles, vals, color=m_info['color'], alpha=m_info['alpha_fill'], zorder=4 if 'Proposed' in m_info['label'] else 2)

    full_title = f"{panel_title}\n{subtitle}" if subtitle else panel_title
    ax.set_title(full_title, fontsize=11.5, fontweight='bold', pad=22, color='#0b5394' if 'Alpha' in panel_title else '#111111')

def render_1x2_combined_radar(alpha_data, broad_data, metric_key='f1', metric_name='F1-Score'):
    """Renders 1x2 side-by-side radar comparison (Alpha vs. BroadBand)."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.5), subplot_kw=dict(polar=True), dpi=300)

    # (a) Alpha Band
    plot_radar_axis(
        axes[0], alpha_data, metric_key=metric_key,
        panel_title="(a) Focused Alpha Band (8–13 Hz)",
        subtitle=f"5-Stage Clinical {metric_name} Profile (N = 1,071 Sequences)"
    )

    # (b) BroadBand
    plot_radar_axis(
        axes[1], broad_data, metric_key=metric_key,
        panel_title="(b) BroadBand EEG (0.5–40 Hz)",
        subtitle=f"5-Stage Clinical {metric_name} Profile (N = 364 Sequences)"
    )

    # Global Title
    fig.suptitle(
        f"Comparative 5-Stage Clinical {metric_name} Radar Profiles: Focused Alpha Band vs. BroadBand EEG",
        fontsize=14.5, fontweight='bold', y=0.98, color='#111111'
    )

    # Shared Legend at bottom center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=6,
        bbox_to_anchor=(0.5, 0.04),
        frameon=True,
        facecolor='#ffffff',
        edgecolor='#bbbbbb',
        fontsize=9.5,
        title="Model Architectures (Comparative Benchmark)"
    )

    plt.tight_layout(rect=[0.02, 0.10, 0.98, 0.93])

    fname_300 = "Figure_Combined_Alpha_vs_BroadBand_Per_Class_Radar_Comparison.png"
    fname_600 = "Figure_Combined_Alpha_vs_BroadBand_Per_Class_Radar_Comparison_600DPI.png"

    dest_paths_300 = [
        os.path.join(SUPP_FIG_DIR, fname_300),
        os.path.join(SIX_MODEL_DIR, fname_300),
        os.path.join(PUB_ROOT_DIR, fname_300)
    ]
    for p in dest_paths_300:
        fig.savefig(p, dpi=300, bbox_inches='tight')

    dest_paths_600 = [
        os.path.join(SUPP_FIG_DIR, fname_600),
        os.path.join(SIX_MODEL_DIR, fname_600),
        os.path.join(PUB_ROOT_DIR, fname_600)
    ]
    for p in dest_paths_600:
        fig.savefig(p, dpi=600, bbox_inches='tight')

    plt.close(fig)
    print(f"  [OK] Saved 1x2 Rendered Radar (300 DPI): {dest_paths_300[0]}")
    print(f"  [OK] Saved 1x2 Rendered Radar (600 DPI): {dest_paths_600[0]}")
    return dest_paths_300[0], dest_paths_600[0]

def create_direct_1x2_stitch():
    """Directly stitches the original two PNG files side-by-side with clean subpanel labels."""
    p1 = os.path.join(SUPP_FIG_DIR, "Alpha_Band_Per_Class_Radar_Comparison.png")
    p2 = os.path.join(SUPP_FIG_DIR, "BroadBand_Per_Class_Radar_Comparison.png")

    if not os.path.exists(p1) or not os.path.exists(p2):
        print(f"  [WARN] Missing original radar PNGs for direct stitch.")
        return None

    im1 = Image.open(p1)
    im2 = Image.open(p2)

    max_h = max(im1.height, im2.height)
    w1 = int(im1.width * (max_h / im1.height))
    w2 = int(im2.width * (max_h / im2.height))

    im1_r = im1.resize((w1, max_h), Image.Resampling.LANCZOS)
    im2_r = im2.resize((w2, max_h), Image.Resampling.LANCZOS)

    padding = 40
    header_h = 110
    total_w = w1 + w2 + padding * 3
    total_h = max_h + header_h + padding * 2

    combined = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    combined.paste(im1_r, (padding, header_h + padding))
    combined.paste(im2_r, (padding * 2 + w1, header_h + padding))

    draw = ImageDraw.Draw(combined)
    try:
        font_title = ImageFont.truetype('arial.ttf', 44)
        font_sub = ImageFont.truetype('arialbd.ttf', 38)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    title_text = "Comparative 5-Stage Clinical Performance Radar: Alpha Band vs. BroadBand"
    draw.text((total_w // 2, 55), title_text, fill=(17, 17, 17), font=font_title, anchor='mm')

    draw.text((padding + 30, header_h + padding + 15), "(a) Focused Alpha Band (8–13 Hz)", fill=(11, 83, 148), font=font_sub)
    draw.text((padding * 2 + w1 + 30, header_h + padding + 15), "(b) BroadBand EEG (0.5–40 Hz)", fill=(11, 83, 148), font=font_sub)

    out_name = "Figure_Combined_Alpha_vs_BroadBand_Radar_Direct_Stitch_1x2.png"
    dest_paths = [
        os.path.join(SUPP_FIG_DIR, out_name),
        os.path.join(SIX_MODEL_DIR, out_name),
        os.path.join(PUB_ROOT_DIR, out_name)
    ]
    for p in dest_paths:
        combined.save(p, dpi=(300, 300))

    print(f"  [OK] Saved 1x2 Direct Stitched Radar (300 DPI): {dest_paths[0]}")
    return dest_paths[0]

def main():
    print("=" * 80)
    print("GENERATING COMBINED 1x2 RADAR FIGURE (ALPHA BAND VS. BROADBAND)")
    print("=" * 80)
    alpha_data, broad_data = load_metrics()

    print("\n[1/2] Generating Master Rendered 1x2 Radar (300 & 600 DPI)...")
    render_1x2_combined_radar(alpha_data, broad_data, metric_key='f1', metric_name='F1-Score')

    print("\n[2/2] Generating Direct Stitched 1x2 Radar from original PNGs (300 DPI)...")
    create_direct_1x2_stitch()

    print("\n" + "=" * 80)
    print("COMBINED 1x2 RADAR GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
