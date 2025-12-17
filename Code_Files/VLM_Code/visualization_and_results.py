"""
POTHOLE WARNING GENERATION - VISUALIZATION & RESULTS PROCESSING
Created: December 17, 2025

This script creates publication-ready visualizations and processes
human evaluation results.

REQUIREMENTS:
pip install pandas numpy matplotlib seaborn openpyxl

USAGE:
python visualization_and_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/evaluation_study"
COMPARISON_CSV = f"{OUTPUT_DIR}/detailed_comparison.csv"
EVALUATION_EXCEL = f"{OUTPUT_DIR}/human_evaluation_study.xlsx"
FIGURE_DIR = f"{OUTPUT_DIR}/figures"

# Create figure directory
Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# FIGURE 1: URGENCY DISTRIBUTION COMPARISON
# ============================================================================

def create_urgency_distribution_chart(csv_path):
    """Create side-by-side urgency distribution comparison"""
    print("\n" + "=" * 80)
    print("CREATING FIGURE 1: Urgency Distribution Comparison")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    # Count urgencies
    urgency_levels = ['low', 'medium', 'high', 'critical']

    image_counts = [
        sum(df['image_only_urgency'] == level) for level in urgency_levels
    ]
    hybrid_counts = [
        sum(df['hybrid_urgency'] == level) for level in urgency_levels
    ]

    # Calculate percentages
    total = len(df)
    image_pcts = [100 * c / total for c in image_counts]
    hybrid_pcts = [100 * c / total for c in hybrid_counts]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(urgency_levels))
    width = 0.35

    # Subplot 1: Counts
    bars1 = ax1.bar(x - width/2, image_counts, width, label='Image-Only', 
                    color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, hybrid_counts, width, label='Multimodal Hybrid',
                    color='#4ECDC4', alpha=0.8)

    ax1.set_xlabel('Urgency Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Scenes', fontsize=12, fontweight='bold')
    ax1.set_title('Urgency Classification Distribution (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.capitalize() for l in urgency_levels])
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

    # Subplot 2: Percentages
    bars3 = ax2.bar(x - width/2, image_pcts, width, label='Image-Only',
                    color='#FF6B6B', alpha=0.8)
    bars4 = ax2.bar(x + width/2, hybrid_pcts, width, label='Multimodal Hybrid',
                    color='#4ECDC4', alpha=0.8)

    ax2.set_xlabel('Urgency Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Urgency Classification Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.capitalize() for l in urgency_levels])
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    fig_path = Path(FIGURE_DIR) / "urgency_distribution_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {fig_path}")

    # Also create stacked version
    fig2, ax = plt.subplots(figsize=(10, 6))

    methods = ['Image-Only', 'Multimodal\nHybrid']
    data = np.array([image_pcts, hybrid_pcts])

    colors = ['#95E1D3', '#F8B739', '#F38181', '#AA4465']

    bottom = np.zeros(2)
    for i, level in enumerate(urgency_levels):
        ax.barh(methods, data[:, i], left=bottom, label=level.capitalize(),
               color=colors[i], alpha=0.85)

        # Add percentage labels
        for j in range(2):
            if data[j, i] > 3:  # Only show if > 3%
                ax.text(bottom[j] + data[j, i]/2, j, f'{data[j, i]:.1f}%',
                       ha='center', va='center', fontsize=11, fontweight='bold')

        bottom += data[:, i]

    ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Urgency Distribution: Image-Only vs Multimodal Hybrid', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    fig2_path = Path(FIGURE_DIR) / "urgency_distribution_stacked.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {fig2_path}")

    plt.close('all')

# ============================================================================
# FIGURE 2: CONFUSION MATRIX HEATMAP
# ============================================================================

def create_confusion_matrix_heatmap(csv_path):
    """Create confusion matrix visualization"""
    print("\n" + "=" * 80)
    print("CREATING FIGURE 2: Confusion Matrix Heatmap")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    urgency_levels = ['low', 'medium', 'high', 'critical']

    # Create confusion matrix
    conf_matrix = np.zeros((4, 4), dtype=int)

    for _, row in df.iterrows():
        img_urgency = row['image_only_urgency']
        hyb_urgency = row['hybrid_urgency']

        if img_urgency in urgency_levels and hyb_urgency in urgency_levels:
            i = urgency_levels.index(img_urgency)
            j = urgency_levels.index(hyb_urgency)
            conf_matrix[i, j] += 1

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=[l.capitalize() for l in urgency_levels],
                yticklabels=[l.capitalize() for l in urgency_levels],
                cbar_kws={'label': 'Number of Scenes'},
                linewidths=1, linecolor='gray', ax=ax)

    ax.set_xlabel('Multimodal Hybrid Classification', fontsize=13, fontweight='bold')
    ax.set_ylabel('Image-Only Classification', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix: Image-Only vs Multimodal Hybrid\n(How methods disagree on urgency classification)',
                fontsize=14, fontweight='bold')

    # Highlight diagonal (perfect agreement)
    for i in range(4):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    plt.tight_layout()

    fig_path = Path(FIGURE_DIR) / "confusion_matrix_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {fig_path}")

    plt.close('all')

# ============================================================================
# FIGURE 3: SPEED REDUCTION COMPARISON
# ============================================================================

def create_speed_reduction_chart(csv_path):
    """Create speed reduction recommendation comparison"""
    print("\n" + "=" * 80)
    print("CREATING FIGURE 3: Speed Reduction Comparison")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    # Extract numeric speeds
    def extract_speed(text):
        if pd.isna(text):
            return None
        import re
        match = re.search(r'(\d+)\s*mph', str(text).lower())
        if match:
            return int(match.group(1))
        if 'maintain' in str(text).lower():
            return 0
        return None

    df['image_speed_num'] = df['image_only_speed'].apply(extract_speed)
    df['hybrid_speed_num'] = df['hybrid_speed'].apply(extract_speed)

    # Remove None values
    df_valid = df.dropna(subset=['image_speed_num', 'hybrid_speed_num'])

    # Create scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Scatter plot
    ax1.scatter(df_valid['image_speed_num'], df_valid['hybrid_speed_num'], 
               alpha=0.5, s=30, color='#4ECDC4')

    # Add diagonal line (perfect agreement)
    max_speed = max(df_valid['image_speed_num'].max(), df_valid['hybrid_speed_num'].max())
    ax1.plot([0, max_speed], [0, max_speed], 'r--', label='Perfect Agreement', linewidth=2)

    ax1.set_xlabel('Image-Only Speed Reduction (mph)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Multimodal Hybrid Speed Reduction (mph)', fontsize=12, fontweight='bold')
    ax1.set_title('Speed Reduction Recommendations Correlation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Calculate correlation
    correlation = df_valid[['image_speed_num', 'hybrid_speed_num']].corr().iloc[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax1.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')

    # Subplot 2: Distribution comparison
    speed_bins = [0, 5, 10, 15, 20, 100]
    speed_labels = ['0-5', '6-10', '11-15', '16-20', '20+']

    image_dist = []
    hybrid_dist = []

    for i in range(len(speed_bins)-1):
        img_count = sum((df_valid['image_speed_num'] >= speed_bins[i]) & 
                       (df_valid['image_speed_num'] < speed_bins[i+1]))
        hyb_count = sum((df_valid['hybrid_speed_num'] >= speed_bins[i]) & 
                       (df_valid['hybrid_speed_num'] < speed_bins[i+1]))
        image_dist.append(100 * img_count / len(df_valid))
        hybrid_dist.append(100 * hyb_count / len(df_valid))

    x = np.arange(len(speed_labels))
    width = 0.35

    bars1 = ax2.bar(x - width/2, image_dist, width, label='Image-Only',
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, hybrid_dist, width, label='Multimodal Hybrid',
                   color='#4ECDC4', alpha=0.8)

    ax2.set_xlabel('Speed Reduction Range (mph)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage of Scenes (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Speed Reduction Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(speed_labels)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only show if > 1%
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    fig_path = Path(FIGURE_DIR) / "speed_reduction_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {fig_path}")

    plt.close('all')

# ============================================================================
# PROCESS HUMAN EVALUATION RESULTS
# ============================================================================

def process_human_evaluation_results(excel_path):
    """Process completed human evaluation study"""
    print("\n" + "=" * 80)
    print("PROCESSING HUMAN EVALUATION RESULTS")
    print("=" * 80)

    if not Path(excel_path).exists():
        print(f"⚠️  File not found: {excel_path}")
        print("   Run this after annotators complete the evaluation")
        return

    # Load evaluation data
    df_eval = pd.read_excel(excel_path, sheet_name='Evaluation')
    df_answer_key = pd.read_excel(excel_path, sheet_name='Answer_Key_Researcher_Only')

    # Merge with answer key
    df_results = df_eval.merge(df_answer_key, on='Scene_ID')

    # Calculate metrics for each method
    image_only_clarity = []
    image_only_safety = []
    hybrid_clarity = []
    hybrid_safety = []

    image_only_preferences = 0
    hybrid_preferences = 0
    ties = 0

    for _, row in df_results.iterrows():
        # Determine which ratings correspond to which method
        if row['Method_A_True_Method'] == 'image_only':
            if pd.notna(row['Method_A_Clarity_Rating']):
                image_only_clarity.append(row['Method_A_Clarity_Rating'])
            if pd.notna(row['Method_A_Safety_Rating']):
                image_only_safety.append(row['Method_A_Safety_Rating'])
            if pd.notna(row['Method_B_Clarity_Rating']):
                hybrid_clarity.append(row['Method_B_Clarity_Rating'])
            if pd.notna(row['Method_B_Safety_Rating']):
                hybrid_safety.append(row['Method_B_Safety_Rating'])

            # Preference
            pref = str(row['Overall_Preference_A_B_TIE']).upper()
            if pref == 'A':
                image_only_preferences += 1
            elif pref == 'B':
                hybrid_preferences += 1
            elif pref == 'TIE':
                ties += 1
        else:
            if pd.notna(row['Method_B_Clarity_Rating']):
                image_only_clarity.append(row['Method_B_Clarity_Rating'])
            if pd.notna(row['Method_B_Safety_Rating']):
                image_only_safety.append(row['Method_B_Safety_Rating'])
            if pd.notna(row['Method_A_Clarity_Rating']):
                hybrid_clarity.append(row['Method_A_Clarity_Rating'])
            if pd.notna(row['Method_A_Safety_Rating']):
                hybrid_safety.append(row['Method_A_Safety_Rating'])

            pref = str(row['Overall_Preference_A_B_TIE']).upper()
            if pref == 'B':
                image_only_preferences += 1
            elif pref == 'A':
                hybrid_preferences += 1
            elif pref == 'TIE':
                ties += 1

    # Calculate averages
    print(f"\n📊 HUMAN EVALUATION RESULTS:")
    print(f"\n   Total scenes evaluated: {len(df_results)}")
    print(f"\n   Average Ratings (1-5 scale):")
    print(f"   {'Metric':<25} {'Image-Only':<15} {'Hybrid':<15} {'Difference':<15}")
    print("   " + "-" * 70)

    if image_only_clarity:
        avg_img_clarity = np.mean(image_only_clarity)
        avg_hyb_clarity = np.mean(hybrid_clarity)
        print(f"   {'Clarity':<25} {avg_img_clarity:<15.2f} {avg_hyb_clarity:<15.2f} {avg_hyb_clarity - avg_img_clarity:+.2f}")

    if image_only_safety:
        avg_img_safety = np.mean(image_only_safety)
        avg_hyb_safety = np.mean(hybrid_safety)
        print(f"   {'Safety':<25} {avg_img_safety:<15.2f} {avg_hyb_safety:<15.2f} {avg_hyb_safety - avg_img_safety:+.2f}")

    total_prefs = image_only_preferences + hybrid_preferences + ties
    if total_prefs > 0:
        print(f"\n   Overall Preferences:")
        print(f"   {'Preferred Image-Only:':<30} {image_only_preferences} ({100*image_only_preferences/total_prefs:.1f}%)")
        print(f"   {'Preferred Hybrid:':<30} {hybrid_preferences} ({100*hybrid_preferences/total_prefs:.1f}%)")
        print(f"   {'Tie:':<30} {ties} ({100*ties/total_prefs:.1f}%)")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Rating comparison
    metrics = ['Clarity', 'Safety']
    image_scores = [avg_img_clarity, avg_img_safety]
    hybrid_scores = [avg_hyb_clarity, avg_hyb_safety]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width/2, image_scores, width, label='Image-Only', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, hybrid_scores, width, label='Multimodal Hybrid', color='#4ECDC4', alpha=0.8)

    ax1.set_ylabel('Average Rating (1-5)', fontsize=12, fontweight='bold')
    ax1.set_title('Human Evaluation: Average Ratings', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 5)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i in range(len(metrics)):
        ax1.text(i - width/2, image_scores[i] + 0.1, f'{image_scores[i]:.2f}',
                ha='center', fontsize=10)
        ax1.text(i + width/2, hybrid_scores[i] + 0.1, f'{hybrid_scores[i]:.2f}',
                ha='center', fontsize=10)

    # Preference pie chart
    if total_prefs > 0:
        labels = ['Image-Only', 'Hybrid', 'Tie']
        sizes = [image_only_preferences, hybrid_preferences, ties]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        explode = (0.05, 0.05, 0.05)

        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Overall Preference Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()

    fig_path = Path(FIGURE_DIR) / "human_evaluation_results.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization: {fig_path}")

    plt.close('all')

    # Save summary to CSV
    summary_data = {
        'Metric': ['Avg_Clarity_Image_Only', 'Avg_Clarity_Hybrid', 
                  'Avg_Safety_Image_Only', 'Avg_Safety_Hybrid',
                  'Preference_Image_Only', 'Preference_Hybrid', 'Preference_Tie'],
        'Value': [avg_img_clarity, avg_hyb_clarity, avg_img_safety, avg_hyb_safety,
                 image_only_preferences, hybrid_preferences, ties]
    }

    df_summary = pd.DataFrame(summary_data)
    summary_path = Path(OUTPUT_DIR) / "human_evaluation_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS AND PROCESSING RESULTS")
    print("=" * 80)

    # Generate visualizations from comparison data
    if Path(COMPARISON_CSV).exists():
        create_urgency_distribution_chart(COMPARISON_CSV)
        create_confusion_matrix_heatmap(COMPARISON_CSV)
        create_speed_reduction_chart(COMPARISON_CSV)
    else:
        print(f"⚠️  Comparison CSV not found: {COMPARISON_CSV}")
        print("   Run ablation_analysis_complete.py first")

    # Process human evaluation results (if available)
    process_human_evaluation_results(EVALUATION_EXCEL)

    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 All figures saved to: {FIGURE_DIR}")
    print(f"\nGenerated figures:")
    print(f"   • urgency_distribution_comparison.png")
    print(f"   • urgency_distribution_stacked.png")
    print(f"   • confusion_matrix_heatmap.png")
    print(f"   • speed_reduction_comparison.png")
    print(f"   • human_evaluation_results.png (if evaluation complete)")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
