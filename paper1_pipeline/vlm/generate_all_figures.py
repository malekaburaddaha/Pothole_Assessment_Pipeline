"""
Generate all publication figures for the ablation study paper
Run this after completing the analysis to create PDF/PNG figures
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# Tables_Dir
Tables_path = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/HIL_Warning_Evaluation_Analysis")
# Load data
urgency = pd.read_csv(Tables_path/'table_urgency_distribution.csv')
irr = pd.read_csv(Tables_path/'table_inter_rater_reliability.csv')
human_eval = pd.read_csv(Tables_path/'table_human_evaluation.csv')
preferences = pd.read_csv(Tables_path/'table_preference_votes.csv')

print("Generating figures...")

# ============================================================================
# FIGURE 1: Urgency Distribution Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(urgency))
width = 0.35

bars1 = ax.bar(x - width/2, urgency['Image_Only_Count'], width, 
               label='Image-Only', color='#FF7F7F', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, urgency['Hybrid_Count'], width, 
               label='Multimodal Hybrid', color='#7FCCFF', edgecolor='black', linewidth=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Warning Urgency Level', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Scenes', fontsize=13, fontweight='bold')
ax.set_title('Warning Urgency Distribution Comparison\n(n=1,119 scenes)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(urgency['Urgency_Level'], fontweight='bold')
ax.legend(fontsize=11, frameon=True, shadow=True, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add statistical annotation
ax.text(0.02, 0.98, 'χ²=256.09, p<0.0001\nCramér\'s V=0.338 (medium effect)', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure1_urgency_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure1_urgency_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure1_urgency_distribution.pdf/png")

# ============================================================================
# FIGURE 2: Human Evaluation Ratings (Grouped Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 6))
metrics = ['Clarity', 'Safety']
x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, human_eval['Image_Only_Mean'], width,
               yerr=human_eval['Image_Only_Std'], 
               label='Image-Only', color='#FF7F7F', 
               edgecolor='black', linewidth=0.8, capsize=7, 
               error_kw={'linewidth': 2})

bars2 = ax.bar(x + width/2, human_eval['Hybrid_Mean'], width,
               yerr=human_eval['Hybrid_Std'], 
               label='Multimodal Hybrid', color='#7FCCFF', 
               edgecolor='black', linewidth=0.8, capsize=7,
               error_kw={'linewidth': 2})

# Add value labels
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + human_eval.iloc[i]['Image_Only_Std'] + 0.1,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + human_eval.iloc[i]['Hybrid_Std'] + 0.1,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Rating Dimension', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Rating (1-5 scale)', fontsize=13, fontweight='bold')
ax.set_title('Human Evaluation: Clarity & Safety Ratings\n(n=50 scenes, 3 annotators, 150 total ratings)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.legend(fontsize=11, frameon=True, shadow=True)
ax.set_ylim([0, 5.5])
ax.axhline(y=3, color='gray', linestyle='--', alpha=0.4, linewidth=1.5, label='Neutral (3.0)')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add significance stars
for i, (metric, p_val) in enumerate(zip(metrics, human_eval['P_Value'])):
    y_pos = max(human_eval.iloc[i]['Image_Only_Mean'] + human_eval.iloc[i]['Image_Only_Std'],
                human_eval.iloc[i]['Hybrid_Mean'] + human_eval.iloc[i]['Hybrid_Std']) + 0.4
    stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax.text(x[i], y_pos, stars, ha='center', va='bottom', fontsize=16, fontweight='bold')

# Add p-value legend
ax.text(0.98, 0.02, '* p<0.05  ** p<0.01  *** p<0.001', 
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figure2_human_evaluation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure2_human_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure2_human_evaluation.pdf/png")

# ============================================================================
# FIGURE 3: Preference Distribution (Pie Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#7FCCFF', '#FF7F7F', '#90EE90']
explode = (0.08, 0, 0)  # Emphasize the winner

wedges, texts, autotexts = ax.pie(
    preferences['Count'], 
    labels=preferences['Choice'],
    autopct=lambda pct: f'{pct:.1f}%\n(n={int(pct/100.*150)})',
    colors=colors,
    explode=explode,
    startangle=90,
    textprops={'fontsize': 13, 'weight': 'bold'},
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
)

# Enhance the winner slice
for i, autotext in enumerate(autotexts):
    if i == 0:  # Hybrid
        autotext.set_color('darkblue')
        autotext.set_fontsize(15)

ax.set_title('Overall Method Preference\n(n=150 votes from 3 independent annotators)', 
             fontsize=16, fontweight='bold', pad=20)

# Add statistical note
fig.text(0.5, 0.05, 
         'Hybrid preferred 3.5× more than Image-Only (p<0.001, χ² test)',
         ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure3_preference_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure3_preference_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure3_preference_distribution.pdf/png")

# ============================================================================
# FIGURE 4: Inter-Rater Reliability (Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Simplify dimension names
dimensions = ['Method A\nClarity', 'Method A\nSafety', 
              'Method B\nClarity', 'Method B\nSafety']
kappas = irr['Fleiss_Kappa'].values
colors_irr = ['#FFB366' if k < 0.5 else '#66B366' if k < 0.6 else '#3366CC' for k in kappas]

bars = ax.barh(dimensions, kappas, color=colors_irr, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, kappa) in enumerate(zip(bars, kappas)):
    width = bar.get_width()
    interpretation = 'Moderate' if kappa < 0.6 else 'Substantial'
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
            f'κ={kappa:.3f}\n({interpretation})',
            ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Fleiss\' Kappa (κ)', fontsize=13, fontweight='bold')
ax.set_title('Inter-Rater Reliability Analysis\n(3 annotators, 50 scenes)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim([0, 1.0])
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add interpretation zones
ax.axvspan(0, 0.2, alpha=0.1, color='red', label='Slight (κ<0.2)')
ax.axvspan(0.2, 0.4, alpha=0.1, color='orange', label='Fair (0.2-0.4)')
ax.axvspan(0.4, 0.6, alpha=0.1, color='yellow', label='Moderate (0.4-0.6)')
ax.axvspan(0.6, 0.8, alpha=0.1, color='lightgreen', label='Substantial (0.6-0.8)')
ax.axvspan(0.8, 1.0, alpha=0.1, color='green', label='Almost Perfect (>0.8)')

# Add legend for interpretation zones
ax.legend(loc='lower right', fontsize=9, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('figure4_inter_rater_reliability.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure4_inter_rater_reliability.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure4_inter_rater_reliability.pdf/png")

print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*80)
print("\nFiles created:")
print("  - figure1_urgency_distribution.pdf/png")
print("  - figure2_human_evaluation.pdf/png")
print("  - figure3_preference_distribution.pdf/png")
print("  - figure4_inter_rater_reliability.pdf/png")
print("\nNext steps:")
print("  1. Review figures for quality")
print("  2. Fill LaTeX tables with values")
print("  3. Write Results section")
print("="*80)
