"""
POTHOLE WARNING GENERATION - COMPLETE ABLATION ANALYSIS WITH HUMAN EVALUATION
Created: December 30, 2025

This script performs:
1. Statistical comparison between Image-Only and Multimodal Hybrid methods
2. Analysis of completed human evaluation study (3 annotators, .ods files)
3. Inter-rater reliability (Fleiss' Kappa for 3 annotators)
4. Generates publication-ready tables and statistics

REQUIREMENTS:
pip install pandas numpy scikit-learn scipy openpyxl odfpy statsmodels

USAGE:
python ablation_analysis_with_human_eval.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import chi2_contingency, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
IMAGE_ONLY_PATH = "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/scene_warnings_image_only_ablation.json"
HYBRID_PATH = "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/scene_warnings_multimodal_hybrid.json"

# Human annotation files
ANNOTATOR_FILES = [
    "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/Warnings_evaluation_study/Annotators_Copies/human_evaluation_study_1.ods",
    "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/Warnings_evaluation_study/Annotators_Copies/human_evaluation_study_2.ods",
    "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/Warnings_evaluation_study/Annotators_Copies/human_evaluation_study_3.ods"
]

OUTPUT_DIR = "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/evaluation_study/results"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_urgency(scene_data):
    """Extract urgency level from scene data"""
    if 'warning' in scene_data and 'urgency_level' in scene_data['warning']:
        return scene_data['warning']['urgency_level']
    return None

def calculate_cramers_v(confusion_mat):
    """Calculate Cramér's V for effect size"""
    chi2 = chi2_contingency(confusion_mat)[0]
    n = confusion_mat.sum()
    min_dim = min(confusion_mat.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def fleiss_kappa(ratings_matrix):
    """
    Calculate Fleiss' Kappa for inter-rater reliability

    Parameters:
    -----------
    ratings_matrix : numpy array of shape (n_subjects, n_raters)
        Each row is a subject, each column is a rater
        Values are the category ratings (e.g., 1-5 for clarity)

    Returns:
    --------
    kappa : float
        Fleiss' Kappa value
    """
    n_subjects, n_raters = ratings_matrix.shape

    # Get unique categories
    categories = np.unique(ratings_matrix[~np.isnan(ratings_matrix)])
    n_categories = len(categories)

    # Create count matrix: n_subjects x n_categories
    counts = np.zeros((n_subjects, n_categories))
    for i, cat in enumerate(categories):
        counts[:, i] = np.sum(ratings_matrix == cat, axis=1)

    # Calculate P_i (proportion of all assignments in category i)
    p_i = np.sum(counts, axis=0) / (n_subjects * n_raters)

    # Calculate P_j (extent of agreement for subject j)
    P_j = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # Calculate P_bar (mean of P_j)
    P_bar = np.mean(P_j)

    # Calculate P_e_bar (expected agreement by chance)
    P_e_bar = np.sum(p_i ** 2)

    # Calculate Fleiss' Kappa
    if P_e_bar == 1:
        return 1.0
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa

# ============================================================================
# PART 1: AUTOMATED STATISTICAL ANALYSIS
# ============================================================================

def load_warning_data():
    """Load both warning generation JSON files"""
    print("="*80)
    print("LOADING WARNING GENERATION DATA")
    print("="*80)

    with open(IMAGE_ONLY_PATH) as f:
        image_only = json.load(f)

    with open(HYBRID_PATH) as f:
        hybrid = json.load(f)

    # Find common scenes
    common_scenes = list(set(image_only.keys()) & set(hybrid.keys()))

    print(f"Image-Only scenes: {len(image_only)}")
    print(f"Hybrid scenes: {len(hybrid)}")
    print(f"Common scenes: {len(common_scenes)}\n")

    return image_only, hybrid, common_scenes

def analyze_urgency_distribution(image_only, hybrid, common_scenes):
    """Analyze urgency level distributions"""
    print("="*80)
    print("URGENCY DISTRIBUTION ANALYSIS")
    print("="*80)

    urgency_io = [extract_urgency(image_only[s]) for s in common_scenes]
    urgency_hyb = [extract_urgency(hybrid[s]) for s in common_scenes]

    # Count distributions
    count_io = Counter(urgency_io)
    count_hyb = Counter(urgency_hyb)

    print("\nImage-Only Distribution:")
    for level in ['low', 'medium', 'high', 'critical']:
        count = count_io.get(level, 0)
        pct = 100 * count / len(common_scenes)
        print(f"  {level:10s}: {count:4d} ({pct:5.1f}%)")

    print("\nMultimodal Hybrid Distribution:")
    for level in ['low', 'medium', 'high', 'critical']:
        count = count_hyb.get(level, 0)
        pct = 100 * count / len(common_scenes)
        print(f"  {level:10s}: {count:4d} ({pct:5.1f}%)")

    # Chi-squared test
    urgency_order = ['low', 'medium', 'high', 'critical']
    io_counts = [count_io.get(level, 0) for level in urgency_order]
    hyb_counts = [count_hyb.get(level, 0) for level in urgency_order]

    contingency = np.array([io_counts, hyb_counts])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    cramers_v = calculate_cramers_v(contingency)

    print(f"\nChi-squared test: χ²={chi2:.2f}, p={p_value:.4f}")
    print(f"Cramér\'s V: {cramers_v:.3f}")
    print(f"Effect size: {'small' if cramers_v < 0.3 else 'medium' if cramers_v < 0.5 else 'large'}")

    return {
        'image_only': count_io,
        'hybrid': count_hyb,
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v
    }

def calculate_agreement_metrics(image_only, hybrid, common_scenes):
    """Calculate Cohen\'s Kappa for urgency agreement"""
    print("\n" + "="*80)
    print("AGREEMENT ANALYSIS")
    print("="*80)

    urgency_io = [extract_urgency(image_only[s]) for s in common_scenes]
    urgency_hyb = [extract_urgency(hybrid[s]) for s in common_scenes]

    # Cohen\'s Kappa
    kappa = cohen_kappa_score(urgency_io, urgency_hyb)

    # Exact agreement
    exact_match = sum(1 for i, h in zip(urgency_io, urgency_hyb) if i == h)
    exact_pct = 100 * exact_match / len(common_scenes)

    print(f"\nCohen\'s Kappa: {kappa:.3f}")
    print(f"Exact agreement: {exact_match}/{len(common_scenes)} ({exact_pct:.1f}%)")
    print(f"Interpretation: {'slight' if kappa < 0.2 else 'fair' if kappa < 0.4 else 'moderate' if kappa < 0.6 else 'substantial'}")

    # Confusion matrix
    urgency_order = ['low', 'medium', 'high', 'critical']
    cm = confusion_matrix(urgency_io, urgency_hyb, labels=urgency_order)

    print("\nConfusion Matrix (Image-Only rows × Hybrid columns):")
    print("         ", "  ".join(f"{u:8s}" for u in urgency_order))
    for i, label_row in enumerate(urgency_order):
        print(f"{label_row:8s}", "  ".join(f"{cm[i,j]:8d}" for j in range(len(urgency_order))))

    return {
        'kappa': kappa,
        'exact_agreement': exact_pct,
        'confusion_matrix': cm
    }

# ============================================================================
# PART 2: HUMAN EVALUATION ANALYSIS
# ============================================================================

def load_human_annotations():
    """Load and merge annotations from 3 annotators"""
    print("\n" + "="*80)
    print("LOADING HUMAN ANNOTATIONS")
    print("="*80)

    annotations = []

    for i, filepath in enumerate(ANNOTATOR_FILES, 1):
        try:
            # Read ODS file (LibreOffice format)
            df = pd.read_excel(filepath, engine='odf', sheet_name='Evaluation')
            df['Annotator_ID'] = i
            annotations.append(df)
            print(f"✓ Loaded Annotator {i}: {len(df)} scenes")
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return None

    if not annotations:
        print("ERROR: No annotation files loaded!")
        return None

    # Merge all annotations
    merged = pd.concat(annotations, ignore_index=True)

    print(f"\nTotal annotations: {len(merged)}")
    print(f"Unique scenes: {merged['Scene_ID'].nunique()}")
    print(f"Annotators: {merged['Annotator_ID'].nunique()}")

    return merged

def analyze_inter_rater_reliability(df):
    """Calculate Fleiss\' Kappa for inter-rater reliability"""
    print("\n" + "="*80)
    print("INTER-RATER RELIABILITY (FLEISS\' KAPPA)")
    print("="*80)

    results = {}

    # Get unique scenes
    scenes = df['Scene_ID'].unique()
    n_scenes = len(scenes)
    n_raters = df['Annotator_ID'].nunique()

    print(f"Scenes: {n_scenes}, Raters: {n_raters}\n")

    # For each rating dimension
    dimensions = {
        'Method_A_Clarity_Rating': 'Method A Clarity',
        'Method_A_Safety_Rating': 'Method A Safety',
        'Method_B_Clarity_Rating': 'Method B Clarity',
        'Method_B_Safety_Rating': 'Method B Safety'
    }

    for col, name in dimensions.items():
        # Create ratings matrix: rows=scenes, cols=annotators
        ratings_matrix = np.full((n_scenes, n_raters), np.nan)

        for scene_idx, scene_id in enumerate(scenes):
            scene_data = df[df['Scene_ID'] == scene_id]
            for _, row in scene_data.iterrows():
                annotator_idx = row['Annotator_ID'] - 1
                ratings_matrix[scene_idx, annotator_idx] = row[col]

        # Remove rows with missing data
        valid_rows = ~np.any(np.isnan(ratings_matrix), axis=1)
        ratings_clean = ratings_matrix[valid_rows]

        if len(ratings_clean) == 0:
            print(f"{name}: No complete ratings")
            continue

        # Calculate Fleiss\' Kappa
        kappa = fleiss_kappa(ratings_clean)

        # Calculate mean and std
        mean_rating = np.nanmean(ratings_matrix)
        std_rating = np.nanstd(ratings_matrix)

        print(f"{name}:")
        print(f"  Fleiss\' κ: {kappa:.3f}")
        print(f"  Mean: {mean_rating:.2f} ± {std_rating:.2f}")
        print(f"  Complete ratings: {len(ratings_clean)}/{n_scenes}")

        results[col] = {
            'kappa': kappa,
            'mean': mean_rating,
            'std': std_rating,
            'n_complete': len(ratings_clean)
        }

    return results

def analyze_preference_votes(df):
    """Analyze overall preference votes"""
    print("\n" + "="*80)
    print("OVERALL PREFERENCE ANALYSIS")
    print("="*80)

    # Count preferences
    prefs = df['Overall_Preference_A_B_TIE'].value_counts()

    total = len(df)

    print(f"\nTotal votes: {total}\n")

    for pref in ['A', 'B', 'TIE']:
        count = prefs.get(pref, 0)
        pct = 100 * count / total
        print(f"  Prefer {pref:3s}: {count:3d} ({pct:5.1f}%)")

    # Map to methods using ground truth
    # Need to decode which is which from the _TRUE_METHOD columns
    method_prefs = {'hybrid': 0, 'image_only': 0, 'tie': 0}

    for _, row in df.iterrows():
        pref = row['Overall_Preference_A_B_TIE']
        method_a = row.get('_TRUE_METHOD_A', '').lower()
        method_b = row.get('_TRUE_METHOD_B', '').lower()

        if pref == 'A':
            if 'hybrid' in method_a or 'multimodal' in method_a:
                method_prefs['hybrid'] += 1
            else:
                method_prefs['image_only'] += 1
        elif pref == 'B':
            if 'hybrid' in method_b or 'multimodal' in method_b:
                method_prefs['hybrid'] += 1
            else:
                method_prefs['image_only'] += 1
        elif pref == 'TIE':
            method_prefs['tie'] += 1

    print(f"\nBy Method (decoded):")
    for method, count in method_prefs.items():
        pct = 100 * count / total
        print(f"  {method:12s}: {count:3d} ({pct:5.1f}%)")

    return {
        'raw_votes': prefs.to_dict(),
        'method_votes': method_prefs
    }

def analyze_clarity_safety_ratings(df):
    """Compare clarity and safety ratings between methods"""
    print("\n" + "="*80)
    print("CLARITY & SAFETY RATINGS COMPARISON")
    print("="*80)

    # Map Method A/B to actual methods
    df_decoded = df.copy()

    # Create columns for actual method ratings
    df_decoded['Hybrid_Clarity'] = np.where(
        df_decoded['_TRUE_METHOD_A'].str.contains('hybrid', case=False, na=False),
        df_decoded['Method_A_Clarity_Rating'],
        df_decoded['Method_B_Clarity_Rating']
    )

    df_decoded['ImageOnly_Clarity'] = np.where(
        df_decoded['_TRUE_METHOD_A'].str.contains('image', case=False, na=False),
        df_decoded['Method_A_Clarity_Rating'],
        df_decoded['Method_B_Clarity_Rating']
    )

    df_decoded['Hybrid_Safety'] = np.where(
        df_decoded['_TRUE_METHOD_A'].str.contains('hybrid', case=False, na=False),
        df_decoded['Method_A_Safety_Rating'],
        df_decoded['Method_B_Safety_Rating']
    )

    df_decoded['ImageOnly_Safety'] = np.where(
        df_decoded['_TRUE_METHOD_A'].str.contains('image', case=False, na=False),
        df_decoded['Method_A_Safety_Rating'],
        df_decoded['Method_B_Safety_Rating']
    )

    # Calculate statistics
    results = {}

    for metric in ['Clarity', 'Safety']:
        hybrid_col = f'Hybrid_{metric}'
        io_col = f'ImageOnly_{metric}'

        hybrid_vals = df_decoded[hybrid_col].dropna()
        io_vals = df_decoded[io_col].dropna()

        hybrid_mean = hybrid_vals.mean()
        hybrid_std = hybrid_vals.std()
        io_mean = io_vals.mean()
        io_std = io_vals.std()

        # Paired t-test
        # Need to pair by scene
        paired_data = []
        for scene in df_decoded['Scene_ID'].unique():
            scene_data = df_decoded[df_decoded['Scene_ID'] == scene]
            if len(scene_data) > 0:
                h_vals = scene_data[hybrid_col].dropna()
                i_vals = scene_data[io_col].dropna()
                if len(h_vals) > 0 and len(i_vals) > 0:
                    paired_data.append((h_vals.mean(), i_vals.mean()))

        if len(paired_data) > 1:
            h_paired, i_paired = zip(*paired_data)
            t_stat, p_value = ttest_ind(h_paired, i_paired)
        else:
            t_stat, p_value = np.nan, np.nan

        print(f"\n{metric} Ratings:")
        print(f"  Hybrid:      {hybrid_mean:.2f} ± {hybrid_std:.2f} (n={len(hybrid_vals)})")
        print(f"  Image-Only:  {io_mean:.2f} ± {io_std:.2f} (n={len(io_vals)})")
        print(f"  Difference:  {hybrid_mean - io_mean:+.2f}")
        print(f"  t-test:      t={t_stat:.2f}, p={p_value:.4f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

        results[metric] = {
            'hybrid_mean': hybrid_mean,
            'hybrid_std': hybrid_std,
            'io_mean': io_mean,
            'io_std': io_std,
            'difference': hybrid_mean - io_mean,
            't_stat': t_stat,
            'p_value': p_value
        }

    return results

# ============================================================================
# PART 3: GENERATE PUBLICATION-READY OUTPUTS
# ============================================================================

def generate_summary_tables(stats):
    """Generate CSV files for LaTeX tables"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Table 1: Urgency Distribution
    urgency_dist = pd.DataFrame({
        'Urgency_Level': ['Low', 'Medium', 'High', 'Critical'],
        'Image_Only_Count': [
            stats['urgency']['image_only'].get('low', 0),
            stats['urgency']['image_only'].get('medium', 0),
            stats['urgency']['image_only'].get('high', 0),
            stats['urgency']['image_only'].get('critical', 0)
        ],
        'Hybrid_Count': [
            stats['urgency']['hybrid'].get('low', 0),
            stats['urgency']['hybrid'].get('medium', 0),
            stats['urgency']['hybrid'].get('high', 0),
            stats['urgency']['hybrid'].get('critical', 0)
        ]
    })

    total = urgency_dist['Image_Only_Count'].sum()
    urgency_dist['Image_Only_Pct'] = 100 * urgency_dist['Image_Only_Count'] / total
    urgency_dist['Hybrid_Pct'] = 100 * urgency_dist['Hybrid_Count'] / total

    urgency_dist.to_csv(output_path / 'table_urgency_distribution.csv', index=False)
    print(f"✓ Saved: table_urgency_distribution.csv")

    # Table 2: Human Evaluation Summary
    if 'human_ratings' in stats:
        human_eval = pd.DataFrame({
            'Metric': ['Clarity (1-5)', 'Safety (1-5)'],
            'Image_Only_Mean': [
                stats['human_ratings']['Clarity']['io_mean'],
                stats['human_ratings']['Safety']['io_mean']
            ],
            'Image_Only_Std': [
                stats['human_ratings']['Clarity']['io_std'],
                stats['human_ratings']['Safety']['io_std']
            ],
            'Hybrid_Mean': [
                stats['human_ratings']['Clarity']['hybrid_mean'],
                stats['human_ratings']['Safety']['hybrid_mean']
            ],
            'Hybrid_Std': [
                stats['human_ratings']['Clarity']['hybrid_std'],
                stats['human_ratings']['Safety']['hybrid_std']
            ],
            'P_Value': [
                stats['human_ratings']['Clarity']['p_value'],
                stats['human_ratings']['Safety']['p_value']
            ]
        })

        human_eval.to_csv(output_path / 'table_human_evaluation.csv', index=False)
        print(f"✓ Saved: table_human_evaluation.csv")

    # Table 3: Preference Votes
    if 'preferences' in stats:
        pref_votes = pd.DataFrame({
            'Choice': ['Prefer Hybrid', 'Prefer Image-Only', 'Tie'],
            'Count': [
                stats['preferences']['method_votes']['hybrid'],
                stats['preferences']['method_votes']['image_only'],
                stats['preferences']['method_votes']['tie']
            ]
        })

        total = pref_votes['Count'].sum()
        pref_votes['Percentage'] = 100 * pref_votes['Count'] / total

        pref_votes.to_csv(output_path / 'table_preference_votes.csv', index=False)
        print(f"✓ Saved: table_preference_votes.csv")

    # Table 4: Inter-Rater Reliability
    if 'irr' in stats:
        irr_data = []
        for metric, values in stats['irr'].items():
            irr_data.append({
                'Dimension': metric.replace('_', ' ').title(),
                'Fleiss_Kappa': values['kappa'],
                'Mean_Rating': values['mean'],
                'Std_Rating': values['std']
            })

        irr_df = pd.DataFrame(irr_data)
        irr_df.to_csv(output_path / 'table_inter_rater_reliability.csv', index=False)
        print(f"✓ Saved: table_inter_rater_reliability.csv")

    print(f"\nAll tables saved to: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete analysis pipeline"""
    print("\n" + "="*80)
    print("POTHOLE WARNING ABLATION STUDY - COMPLETE ANALYSIS")
    print("="*80)

    stats = {}

    # Part 1: Automated statistical analysis
    print("\n### PART 1: AUTOMATED STATISTICAL ANALYSIS ###\n")

    image_only, hybrid, common_scenes = load_warning_data()

    stats['urgency'] = analyze_urgency_distribution(image_only, hybrid, common_scenes)
    stats['agreement'] = calculate_agreement_metrics(image_only, hybrid, common_scenes)

    # Part 2: Human evaluation analysis
    print("\n\n### PART 2: HUMAN EVALUATION ANALYSIS ###\n")

    human_df = load_human_annotations()

    if human_df is not None:
        stats['irr'] = analyze_inter_rater_reliability(human_df)
        stats['preferences'] = analyze_preference_votes(human_df)
        stats['human_ratings'] = analyze_clarity_safety_ratings(human_df)
    else:
        print("\nWARNING: Skipping human evaluation analysis (files not found or invalid)")

    # Part 3: Generate outputs
    print("\n\n### PART 3: GENERATING OUTPUTS ###\n")

    generate_summary_tables(stats)

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print(f"1. Urgency Distribution: χ²={stats['urgency']['chi2']:.2f}, p={stats['urgency']['p_value']:.4f}")
    print(f"2. Method Agreement: κ={stats['agreement']['kappa']:.3f}")

    if 'human_ratings' in stats:
        clarity_diff = stats['human_ratings']['Clarity']['difference']
        safety_diff = stats['human_ratings']['Safety']['difference']
        print(f"3. Human Ratings:")
        print(f"   - Clarity: Hybrid {clarity_diff:+.2f} vs Image-Only")
        print(f"   - Safety:  Hybrid {safety_diff:+.2f} vs Image-Only")

    if 'preferences' in stats:
        hybrid_pct = 100 * stats['preferences']['method_votes']['hybrid'] / sum(stats['preferences']['method_votes'].values())
        print(f"4. Preference: {hybrid_pct:.1f}% prefer Hybrid")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
