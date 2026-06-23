import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

datasets = ['Charades-STA', 'QVHighlights', 'TACoS', 'Average']
frag_rate   = [1.91, 3.79, 5.78, 3.83]
fixes_clip  = [92.67, 76.59, 96.81, 88.69]
wn_coverage = [93.53, 91.63, 92.86, 92.67]

x = np.arange(len(datasets))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
fig.subplots_adjust(wspace=0.38)

# ── Panel 1: CLIP Fragmentation Rate ──────────────────────────────────────────
colors_frag = ['#5B8DB8', '#5B8DB8', '#5B8DB8', '#A8BDD0']
bars1 = ax1.bar(x, frag_rate, width=0.5, color=colors_frag, edgecolor='white',
                linewidth=0.6, zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=9, rotation=12, ha='right')
ax1.set_ylabel('Fragmentation Rate (%)', fontsize=10)
ax1.set_title('CLIP BPE Fragmentation Rate', fontsize=11, fontweight='bold', pad=8)
ax1.set_ylim(0, 8)
ax1.yaxis.grid(True, linestyle='--', alpha=0.35, color='#bbbbbb', zorder=0)
ax1.set_axisbelow(True)
ax1.spines[['top', 'right']].set_visible(False)

for bar, val in zip(bars1, frag_rate):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# ── Panel 2: GloVe Fixes CLIP & WordNet Coverage ──────────────────────────────
c_glove = '#C4845A'
c_wn    = '#6AA88A'

bars_g = ax2.bar(x - width/2, fixes_clip,  width, label='GloVe Fixes CLIP',
                 color=c_glove, edgecolor='white', linewidth=0.8, zorder=3)
bars_w = ax2.bar(x + width/2, wn_coverage, width, label='WordNet Coverage',
                 color=c_wn,    edgecolor='white', linewidth=0.8, zorder=3)

ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=9, rotation=12, ha='right')
ax2.set_ylabel('Coverage (%)', fontsize=10)
ax2.set_title('GloVe & WordNet Complementary Coverage', fontsize=11, fontweight='bold', pad=8)
ax2.set_ylim(60, 102)
ax2.yaxis.grid(True, linestyle='--', alpha=0.35, color='#bbbbbb', zorder=0)
ax2.set_axisbelow(True)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(fontsize=8.5, framealpha=0.85, loc='lower right')

for bar, val in zip(bars_g, fixes_clip):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar, val in zip(bars_w, wn_coverage):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

fig.suptitle('Vocabulary and Tokenization Analysis across VMR Datasets',
             fontsize=12, fontweight='bold', y=1.01)

out = '/Users/jihuang/Downloads/EventSem/vocab_coverage_chart.pdf'
plt.savefig(out, bbox_inches='tight', dpi=300)
out_png = out.replace('.pdf', '.png')
plt.savefig(out_png, bbox_inches='tight', dpi=300)
print(f'Saved: {out}')
print(f'Saved: {out_png}')
