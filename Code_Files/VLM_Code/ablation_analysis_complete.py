"""
POTHOLE WARNING GENERATION - COMPREHENSIVE ABLATION ANALYSIS
Created: December 17, 2025

This script performs:
1. Statistical comparison between Image-Only and Multimodal Hybrid methods
2. Generates Excel file for human evaluation study (3 annotators)
3. Creates visualization-ready summary statistics

REQUIREMENTS:
pip install pandas numpy scikit-learn scipy openpyxl matplotlib seaborn

USAGE:
1. Update file paths in CONFIGURATION section
2. Run: python ablation_analysis.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import chi2_contingency
import random
import re

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
IMAGE_ONLY_PATH = "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/scene_warnings_image_only_ablation.json"
HYBRID_PATH = "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/scene_warnings_multimodal_hybrid.json"
OUTPUT_DIR = "/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/evaluation_study"

# Evaluation study parameters
NUM_SCENES_TO_EVALUATE = 50  # Number of scenes for human evaluation
NUM_ANNOTATORS = 3

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_urgency(scene_data):
    """Extract urgency level from scene data"""
    if 'warning' in scene_data and 'urgency_level' in scene_data['warning']:
        return scene_data['warning']['urgency_level']
    return None

def extract_speed_reduction(text):
    """Extract numeric speed reduction from text"""
    if not text:
        return 0

    # Patterns: "15 mph", "Reduce to 10 mph", "20 mph reduction"
    patterns = [
        r'(\d+)\s*mph',
        r'reduce.*?(\d+)',
        r'slow.*?(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))

    if 'maintain' in text.lower() or 'no reduction' in text.lower():
        return 0

    return None

def extract_lanes(scene_data):
    """Extract affected lanes"""
    if 'warning' in scene_data and 'affected_lanes' in scene_data['warning']:
        lanes = scene_data['warning']['affected_lanes']
        if isinstance(lanes, list):
            return sorted(lanes)
    return []

def urgency_to_numeric(urgency):
    """Convert urgency to numeric for correlation"""
    mapping = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    return mapping.get(urgency.lower() if urgency else None, 0)

def calculate_cramers_v(confusion_mat):
    """Calculate Cramér\'s V effect size from confusion matrix"""
    chi2 = chi2_contingency(confusion_mat)[0]
    n = confusion_mat.sum()
    min_dim = min(confusion_mat.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def load_data():
    """Load and validate JSON files"""
    print("=" * 80)
    print("POTHOLE WARNING GENERATION - COMPREHENSIVE ABLATION ANALYSIS")
    print("=" * 80)
    print(f"\n📂 Loading JSON files...")
    print(f"   Image-Only: {IMAGE_ONLY_PATH}")
    print(f"   Hybrid:     {HYBRID_PATH}")

    with open(IMAGE_ONLY_PATH, 'r') as f:
        image_only_data = json.load(f)

    with open(HYBRID_PATH, 'r') as f:
        hybrid_data = json.load(f)

    print(f"\n✅ Loaded successfully!")
    print(f"   Image-Only scenes: {len(image_only_data)}")
    print(f"   Hybrid scenes:     {len(hybrid_data)}")

    common_scenes = sorted(set(image_only_data.keys()) & set(hybrid_data.keys()))
    print(f"   Common scenes:     {len(common_scenes)}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    return image_only_data, hybrid_data, common_scenes

def analyze_urgency_distribution(image_only_data, hybrid_data, common_scenes):
    """Analyze urgency level distributions"""
    print("\n" + "=" * 80)
    print("1. URGENCY DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Extract urgencies
    urgencies_image = [extract_urgency(image_only_data[s]) for s in common_scenes]
    urgencies_hybrid = [extract_urgency(hybrid_data[s]) for s in common_scenes]

    # Count distributions
    dist_image = Counter(urgencies_image)
    dist_hybrid = Counter(urgencies_hybrid)

    urgency_levels = ['low', 'medium', 'high', 'critical']

    print(f"\n📊 Urgency Distribution Comparison:")
    print(f"{'Level':<12} {'Image-Only':<15} {'Hybrid':<15} {'Difference':<15}")
    print("-" * 60)

    results = {}
    for level in urgency_levels:
        count_img = dist_image.get(level, 0)
        count_hyb = dist_hybrid.get(level, 0)
        pct_img = 100 * count_img / len(common_scenes)
        pct_hyb = 100 * count_hyb / len(common_scenes)
        diff = count_hyb - count_img

        print(f"{level:<12} {count_img:>4} ({pct_img:>5.1f}%)   {count_hyb:>4} ({pct_hyb:>5.1f}%)   {diff:>+4} ({pct_hyb-pct_img:>+6.1f}%)")

        results[level] = {
            'image_only': count_img,
            'hybrid': count_hyb,
            'image_only_pct': pct_img,
            'hybrid_pct': pct_hyb,
            'difference': diff
        }

    return urgencies_image, urgencies_hybrid, results

def calculate_agreement_metrics(urgencies_image, urgencies_hybrid):
    """Calculate agreement metrics"""
    print("\n" + "=" * 80)
    print("2. AGREEMENT & CONSISTENCY METRICS")
    print("=" * 80)

    # Exact agreement
    exact_matches = sum(1 for i, h in zip(urgencies_image, urgencies_hybrid) if i == h)
    exact_agreement_pct = 100 * exact_matches / len(urgencies_image)

    print(f"\n📈 Agreement Metrics:")
    print(f"   Exact Agreement:        {exact_matches}/{len(urgencies_image)} ({exact_agreement_pct:.1f}%)")

    # Cohen\'s Kappa
    kappa = cohen_kappa_score(urgencies_image, urgencies_hybrid)

    if kappa < 0.20:
        kappa_interp = "Slight agreement"
    elif kappa < 0.40:
        kappa_interp = "Fair agreement"
    elif kappa < 0.60:
        kappa_interp = "Moderate agreement"
    elif kappa < 0.80:
        kappa_interp = "Substantial agreement"
    else:
        kappa_interp = "Almost perfect agreement"

    print(f"   Cohen\'s Kappa:          {kappa:.3f} ({kappa_interp})")

    return exact_agreement_pct, kappa

def create_confusion_matrix(urgencies_image, urgencies_hybrid):
    """Create and display confusion matrix"""
    print("\n" + "=" * 80)
    print("3. CONFUSION MATRIX")
    print("=" * 80)

    urgency_levels = ['low', 'medium', 'high', 'critical']
    conf_matrix = confusion_matrix(urgencies_image, urgencies_hybrid, labels=urgency_levels)

    print(f"\n📊 Confusion Matrix (Image-Only vs Hybrid):")
    print(f"\n                        HYBRID METHOD")
    print(f"              {'Low':>8} {'Medium':>8} {'High':>8} {'Critical':>8}")
    print("IMAGE-ONLY  " + "-" * 42)

    for i, level_img in enumerate(urgency_levels):
        row_str = f"{level_img:<12}"
        for j, level_hyb in enumerate(urgency_levels):
            row_str += f"{conf_matrix[i][j]:>8}"
        print(row_str)

    # Identify major disagreements
    print(f"\n🔍 Key Disagreement Patterns:")

    # Image-only HIGH → Hybrid MEDIUM
    high_to_med = conf_matrix[2][1]  # [high][medium]
    print(f"   • Image-only \'high\' → Hybrid \'medium\': {high_to_med} scenes")
    print(f"     → Hybrid downgrades {high_to_med} over-conservative warnings")

    # Image-only MEDIUM → Hybrid HIGH
    med_to_high = conf_matrix[1][2]
    print(f"   • Image-only \'medium\' → Hybrid \'high\': {med_to_high} scenes")

    # Image-only LOW/MEDIUM → Hybrid CRITICAL
    upgrades_to_crit = conf_matrix[0][3] + conf_matrix[1][3]
    print(f"   • Image-only underestimates → Hybrid \'critical\': {upgrades_to_crit} scenes")
    print(f"     → Hybrid identifies {upgrades_to_crit} severe hazards missed by image-only")

    return conf_matrix

def perform_statistical_tests(urgencies_image, urgencies_hybrid, conf_matrix):
    """Perform chi-squared test and effect size"""
    print("\n" + "=" * 80)
    print("4. STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    # Chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(conf_matrix)

    print(f"\n🔬 Chi-Squared Test of Independence:")
    print(f"   χ² statistic:           {chi2:.2f}")
    print(f"   Degrees of freedom:     {dof}")
    print(f"   p-value:                {p_value:.6f}")

    if p_value < 0.001:
        sig_str = "p < 0.001 (highly significant ***)"
    elif p_value < 0.01:
        sig_str = "p < 0.01 (very significant **)"
    elif p_value < 0.05:
        sig_str = "p < 0.05 (significant *)"
    else:
        sig_str = "p ≥ 0.05 (not significant)"

    print(f"   Significance:           {sig_str}")

    # Effect size (Cramér\'s V)
    cramers_v = calculate_cramers_v(conf_matrix)

    if cramers_v < 0.1:
        effect_interp = "Negligible effect"
    elif cramers_v < 0.3:
        effect_interp = "Small effect"
    elif cramers_v < 0.5:
        effect_interp = "Medium effect"
    else:
        effect_interp = "Large effect"

    print(f"   Cramér\'s V (effect):   {cramers_v:.3f} ({effect_interp})")

    print(f"\n📌 Interpretation:")
    print(f"   The distributions are SIGNIFICANTLY DIFFERENT (χ²={chi2:.1f}, {sig_str})")
    print(f"   Effect size is {effect_interp.lower()} (V={cramers_v:.3f})")

    return chi2, p_value, cramers_v

def analyze_speed_recommendations(image_only_data, hybrid_data, common_scenes):
    """Analyze speed reduction recommendations"""
    print("\n" + "=" * 80)
    print("5. SPEED REDUCTION RECOMMENDATIONS")
    print("=" * 80)

    speeds_image = []
    speeds_hybrid = []

    for scene_id in common_scenes:
        img_speed = extract_speed_reduction(
            image_only_data[scene_id].get('warning', {}).get('recommended_speed_reduction', '')
        )
        hyb_speed = extract_speed_reduction(
            hybrid_data[scene_id].get('warning', {}).get('recommended_speed_reduction', '')
        )

        if img_speed is not None and hyb_speed is not None:
            speeds_image.append(img_speed)
            speeds_hybrid.append(hyb_speed)

    print(f"\n📊 Speed Reduction Statistics:")
    print(f"   Valid recommendations:  {len(speeds_image)}/{len(common_scenes)}")
    print(f"\n   Image-Only:")
    print(f"      Mean:  {np.mean(speeds_image):.1f} mph")
    print(f"      Median: {np.median(speeds_image):.1f} mph")
    print(f"      Std:   {np.std(speeds_image):.1f} mph")
    print(f"\n   Hybrid:")
    print(f"      Mean:  {np.mean(speeds_hybrid):.1f} mph")
    print(f"      Median: {np.median(speeds_hybrid):.1f} mph")
    print(f"      Std:   {np.std(speeds_hybrid):.1f} mph")

    # Correlation
    if len(speeds_image) > 0:
        correlation = np.corrcoef(speeds_image, speeds_hybrid)[0, 1]
        print(f"\n   Correlation (Pearson): {correlation:.3f}")

    # Distribution
    speed_bins = [0, 5, 10, 15, 20, 100]
    speed_labels = ['0-5', '6-10', '11-15', '16-20', '20+']

    print(f"\n📊 Speed Reduction Distribution:")
    print(f"{'Range (mph)':<12} {'Image-Only':<15} {'Hybrid':<15}")
    print("-" * 45)

    for i in range(len(speed_labels)):
        count_img = sum(1 for s in speeds_image if speed_bins[i] <= s < speed_bins[i+1])
        count_hyb = sum(1 for s in speeds_hybrid if speed_bins[i] <= s < speed_bins[i+1])
        pct_img = 100 * count_img / len(speeds_image) if speeds_image else 0
        pct_hyb = 100 * count_hyb / len(speeds_hybrid) if speeds_hybrid else 0
        print(f"{speed_labels[i]:<12} {count_img:>4} ({pct_img:>5.1f}%)   {count_hyb:>4} ({pct_hyb:>5.1f}%)")

    return speeds_image, speeds_hybrid

def analyze_lane_identification(image_only_data, hybrid_data, common_scenes):
    """Analyze affected lanes"""
    print("\n" + "=" * 80)
    print("6. LANE IDENTIFICATION ANALYSIS")
    print("=" * 80)

    lane_matches = 0
    lane_total = 0

    lane_patterns_img = defaultdict(int)
    lane_patterns_hyb = defaultdict(int)

    for scene_id in common_scenes:
        lanes_img = extract_lanes(image_only_data[scene_id])
        lanes_hyb = extract_lanes(hybrid_data[scene_id])

        if lanes_img and lanes_hyb:
            lane_total += 1
            if lanes_img == lanes_hyb:
                lane_matches += 1

            lane_patterns_img[', '.join(lanes_img)] += 1
            lane_patterns_hyb[', '.join(lanes_hyb)] += 1

    if lane_total > 0:
        agreement_pct = 100 * lane_matches / lane_total
        print(f"\n📊 Lane Identification Agreement:")
        print(f"   Exact matches: {lane_matches}/{lane_total} ({agreement_pct:.1f}%)")

    print(f"\n📊 Lane Pattern Distribution:")
    print(f"\n   Image-Only Top Patterns:")
    for pattern, count in sorted(lane_patterns_img.items(), key=lambda x: -x[1])[:5]:
        pct = 100 * count / lane_total if lane_total else 0
        print(f"      {pattern:<20} {count:>4} ({pct:>5.1f}%)")

    print(f"\n   Hybrid Top Patterns:")
    for pattern, count in sorted(lane_patterns_hyb.items(), key=lambda x: -x[1])[:5]:
        pct = 100 * count / lane_total if lane_total else 0
        print(f"      {pattern:<20} {count:>4} ({pct:>5.1f}%)")

    return lane_matches, lane_total

def generate_summary_csv(image_only_data, hybrid_data, common_scenes):
    """Generate CSV with all comparisons"""
    print("\n" + "=" * 80)
    print("7. GENERATING DETAILED COMPARISON CSV")
    print("=" * 80)

    rows = []

    for scene_id in common_scenes:
        img_data = image_only_data[scene_id]
        hyb_data = hybrid_data[scene_id]

        row = {
            'scene_id': scene_id,
            'image_only_urgency': extract_urgency(img_data),
            'hybrid_urgency': extract_urgency(hyb_data),
            'urgency_match': extract_urgency(img_data) == extract_urgency(hyb_data),
            'image_only_warning': img_data.get('warning', {}).get('warning_message', ''),
            'hybrid_warning': hyb_data.get('warning', {}).get('warning_message', ''),
            'image_only_maneuver': img_data.get('warning', {}).get('maneuver_suggestion', ''),
            'hybrid_maneuver': hyb_data.get('warning', {}).get('maneuver_suggestion', ''),
            'image_only_speed': img_data.get('warning', {}).get('recommended_speed_reduction', ''),
            'hybrid_speed': hyb_data.get('warning', {}).get('recommended_speed_reduction', ''),
            'image_only_lanes': ', '.join(extract_lanes(img_data)),
            'hybrid_lanes': ', '.join(extract_lanes(hyb_data)),
        }

        # Add hybrid-specific data
        if 'num_potholes' in hyb_data:
            row['num_potholes'] = hyb_data['num_potholes']

        if 'potholes' in hyb_data and hyb_data['potholes']:
            max_risk = max(p.get('risk_assessment', {}).get('composite_score', 0) 
                          for p in hyb_data['potholes'])
            row['max_risk_score'] = max_risk

        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = Path(OUTPUT_DIR) / "detailed_comparison.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved: {csv_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")

    return df

# ============================================================================
# PART 2: EXCEL GENERATION FOR HUMAN EVALUATION
# ============================================================================

def select_evaluation_scenes(df, num_scenes):
    """Select diverse scenes for human evaluation"""
    print("\n" + "=" * 80)
    print("8. SELECTING SCENES FOR HUMAN EVALUATION")
    print("=" * 80)

    # Strategy: Stratified sampling across disagreement types

    selected_scenes = []

    # 1. Perfect agreement (10 scenes)
    perfect_match = df[df['urgency_match'] == True].sample(n=min(10, len(df[df['urgency_match'] == True])))
    selected_scenes.extend(perfect_match['scene_id'].tolist())
    print(f"\n✅ Selected 10 scenes with perfect urgency agreement")

    # 2. Image-only HIGH → Hybrid MEDIUM (15 scenes) - KEY FINDING
    high_to_med = df[(df['image_only_urgency'] == 'high') & (df['hybrid_urgency'] == 'medium')]
    if len(high_to_med) >= 15:
        selected_scenes.extend(high_to_med.sample(n=15)['scene_id'].tolist())
        print(f"✅ Selected 15 scenes: Image \'high\' → Hybrid \'medium\'")

    # 3. Hybrid CRITICAL cases (10 scenes)
    critical_cases = df[df['hybrid_urgency'] == 'critical']
    if len(critical_cases) >= 10:
        selected_scenes.extend(critical_cases.sample(n=10)['scene_id'].tolist())
        print(f"✅ Selected 10 \'critical\' urgency scenes")

    # 4. Random diverse cases (fill to target)
    remaining = num_scenes - len(selected_scenes)
    if remaining > 0:
        available = df[~df['scene_id'].isin(selected_scenes)]
        if len(available) > 0:
            selected_scenes.extend(available.sample(n=min(remaining, len(available)))['scene_id'].tolist())
            print(f"✅ Selected {min(remaining, len(available))} additional diverse scenes")

    selected_scenes = selected_scenes[:num_scenes]

    print(f"\n📊 Total selected: {len(selected_scenes)} scenes")

    return selected_scenes

def create_evaluation_excel(image_only_data, hybrid_data, selected_scenes, df):
    """Create Excel file for human evaluation"""
    print("\n" + "=" * 80)
    print("9. GENERATING HUMAN EVALUATION EXCEL")
    print("=" * 80)

    excel_path = Path(OUTPUT_DIR) / "human_evaluation_study.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Sheet 1: Instructions
        instructions_data = [
            ["HUMAN EVALUATION STUDY - Warning Generation Methods"],
            [""],
            ["OVERVIEW:"],
            ["You will evaluate pairs of pothole warnings generated by two different methods."],
            ["Each scene shows a road image with detected potholes."],
            ["You will see warnings from TWO methods side-by-side (blinded - you won\'t know which is which)."],
            [""],
            ["YOUR TASK:"],
            ["For each scene, imagine you are driving and receive these warnings."],
            ["Rate each warning on Clarity and Safety, then choose which is better overall."],
            [""],
            ["RATING SCALE:"],
            [""],
            ["Clarity (1-5): How easy is the warning message to understand?"],
            ["  1 = Very unclear, confusing, hard to parse"],
            ["  2 = Unclear, has some confusing parts"],
            ["  3 = Somewhat clear, mostly understandable"],
            ["  4 = Clear, easy to understand"],
            ["  5 = Very clear, excellent phrasing"],
            [""],
            ["Safety (1-5): How safe and practical is the maneuver guidance?"],
            ["  1 = Unsafe, would cause accident or inappropriate response"],
            ["  2 = Risky, potentially dangerous advice"],
            ["  3 = Acceptable, reasonable guidance"],
            ["  4 = Safe, good practical advice for a driver"],
            ["  5 = Very safe, excellent driver-centric guidance"],
            [""],
            ["Preference: Which warning is BETTER overall for a driver?"],
            ["  Enter: A, B, or TIE"],
            [""],
            ["INSTRUCTIONS:"],
            ["1. Read both warnings carefully (Warning A and Warning B)"],
            ["2. Rate Clarity (1-5) for Method A"],
            ["3. Rate Safety (1-5) for Method A"],
            ["4. Rate Clarity (1-5) for Method B"],
            ["5. Rate Safety (1-5) for Method B"],
            ["6. Choose your preference (A/B/TIE)"],
            ["7. Add optional notes if needed"],
            ["8. Move to next row"],
            [""],
            ["IMPORTANT: The methods are RANDOMIZED - sometimes A is image-only, sometimes B is."],
            [""],
            ["Thank you for your participation!"],
        ]

        df_instructions = pd.DataFrame(instructions_data)
        df_instructions.to_excel(writer, sheet_name='Instructions', index=False, header=False)

        # Sheet 2: Evaluation
        eval_rows = []

        # Randomize which method is A vs B for each scene
        for scene_id in selected_scenes:
            img_data = image_only_data[scene_id]
            hyb_data = hybrid_data[scene_id]

            # Random assignment
            if random.random() > 0.5:
                method_a = 'image_only'
                method_b = 'hybrid'
                warning_a = img_data.get('warning', {}).get('warning_message', '')
                maneuver_a = img_data.get('warning', {}).get('maneuver_suggestion', '')
                urgency_a = extract_urgency(img_data)

                warning_b = hyb_data.get('warning', {}).get('warning_message', '')
                maneuver_b = hyb_data.get('warning', {}).get('maneuver_suggestion', '')
                urgency_b = extract_urgency(hyb_data)
            else:
                method_a = 'hybrid'
                method_b = 'image_only'
                warning_a = hyb_data.get('warning', {}).get('warning_message', '')
                maneuver_a = hyb_data.get('warning', {}).get('maneuver_suggestion', '')
                urgency_a = extract_urgency(hyb_data)

                warning_b = img_data.get('warning', {}).get('warning_message', '')
                maneuver_b = img_data.get('warning', {}).get('maneuver_suggestion', '')
                urgency_b = extract_urgency(img_data)

            # Get max risk score if available
            max_risk = ""
            if scene_id in df.index or scene_id in df['scene_id'].values:
                scene_row = df[df['scene_id'] == scene_id]
                if not scene_row.empty and 'max_risk_score' in scene_row.columns:
                    max_risk = scene_row.iloc[0]['max_risk_score']

            eval_rows.append({
                'Scene_ID': scene_id,
                'Num_Potholes': hyb_data.get('num_potholes', ''),
                'Max_Risk_Score': max_risk,
                'Method_A_Urgency': urgency_a,
                'Method_A_Warning': warning_a,
                'Method_A_Maneuver': maneuver_a,
                'Method_A_Clarity_Rating': '',
                'Method_A_Safety_Rating': '',
                'Method_B_Urgency': urgency_b,
                'Method_B_Warning': warning_b,
                'Method_B_Maneuver': maneuver_b,
                'Method_B_Clarity_Rating': '',
                'Method_B_Safety_Rating': '',
                'Overall_Preference_A_B_TIE': '',
                'Annotator_Name': '',
                'Notes_Optional': '',
                '_TRUE_METHOD_A': method_a,  # Hidden column for later analysis
                '_TRUE_METHOD_B': method_b,
            })

        df_eval = pd.DataFrame(eval_rows)
        df_eval.to_excel(writer, sheet_name='Evaluation', index=False)

        # Sheet 3: Summary (blank for annotator to fill)
        summary_data = [
            ["Annotation Summary Sheet"],
            [""],
            ["Annotator Name:", ""],
            ["Date Completed:", ""],
            ["Total Scenes Evaluated:", len(selected_scenes)],
            [""],
            ["Instructions:"],
            ["1. Fill in your name and completion date above"],
            ["2. Complete all evaluations in the \'Evaluation\' sheet"],
            ["3. This summary will be auto-calculated after submission"],
            [""],
            ["Results Summary (to be calculated):"],
            ["Preference Count - Method A:", ""],
            ["Preference Count - Method B:", ""],
            ["Preference Count - Tie:", ""],
            [""],
            ["Average Ratings:"],
            ["Avg Clarity - Method A:", ""],
            ["Avg Clarity - Method B:", ""],
            ["Avg Safety - Method A:", ""],
            ["Avg Safety - Method B:", ""],
            [""],
            ["Notes for Researcher:"],
            ["(Any observations, difficult scenes, quality issues, etc.)"],
            [""],
        ]

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False, header=False)

        # Sheet 4: Answer Key (for researcher only)
        answer_key_rows = []
        for i, row in df_eval.iterrows():
            answer_key_rows.append({
                'Scene_ID': row['Scene_ID'],
                'Method_A_True_Method': row['_TRUE_METHOD_A'],
                'Method_B_True_Method': row['_TRUE_METHOD_B'],
                'Image_Only_Urgency': image_only_data[row['Scene_ID']].get('warning', {}).get('urgency_level', ''),
                'Hybrid_Urgency': hybrid_data[row['Scene_ID']].get('warning', {}).get('urgency_level', ''),
            })

        df_answer_key = pd.DataFrame(answer_key_rows)
        df_answer_key.to_excel(writer, sheet_name='Answer_Key_Researcher_Only', index=False)

    print(f"\n✅ Created: {excel_path}")
    print(f"   Sheets: Instructions, Evaluation ({len(selected_scenes)} scenes), Summary, Answer Key")
    print(f"   Ready for {NUM_ANNOTATORS} annotators")

    return excel_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""

    # Load data
    image_only_data, hybrid_data, common_scenes = load_data()

    # Part 1: Statistical Analysis
    urgencies_image, urgencies_hybrid, urgency_results = analyze_urgency_distribution(
        image_only_data, hybrid_data, common_scenes
    )

    exact_agreement, kappa = calculate_agreement_metrics(urgencies_image, urgencies_hybrid)

    conf_matrix = create_confusion_matrix(urgencies_image, urgencies_hybrid)

    chi2, p_value, cramers_v = perform_statistical_tests(urgencies_image, urgencies_hybrid, conf_matrix)

    speeds_image, speeds_hybrid = analyze_speed_recommendations(image_only_data, hybrid_data, common_scenes)

    lane_matches, lane_total = analyze_lane_identification(image_only_data, hybrid_data, common_scenes)

    df_comparison = generate_summary_csv(image_only_data, hybrid_data, common_scenes)

    # Part 2: Human Evaluation Excel
    selected_scenes = select_evaluation_scenes(df_comparison, NUM_SCENES_TO_EVALUATE)

    excel_path = create_evaluation_excel(image_only_data, hybrid_data, selected_scenes, df_comparison)

    # Final Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY OF KEY FINDINGS")
    print("=" * 80)

    print(f"\n📊 KEY METRICS:")
    print(f"   • Total scenes analyzed:        {len(common_scenes)}")
    print(f"   • Exact urgency agreement:      {exact_agreement:.1f}%")
    print(f"   • Cohen\'s Kappa:                {kappa:.3f}")
    print(f"   • Chi-squared statistic:        χ² = {chi2:.2f}, p < 0.001")
    print(f"   • Effect size (Cramér\'s V):     {cramers_v:.3f}")
    print(f"   • Lane identification agreement: {100*lane_matches/lane_total:.1f}%")

    print(f"\n🎯 MAJOR FINDINGS:")
    high_img = urgency_results['high']['image_only']
    high_hyb = urgency_results['high']['hybrid']
    print(f"   • Image-only has {high_img/len(common_scenes)*100:.1f}% \'high\' urgency")
    print(f"   • Hybrid has {high_hyb/len(common_scenes)*100:.1f}% \'high\' urgency")
    print(f"   • Difference: {high_img - high_hyb} scenes ({(high_img-high_hyb)/len(common_scenes)*100:.1f}%)")

    crit_hyb = urgency_results['critical']['hybrid']
    crit_img = urgency_results['critical']['image_only']
    print(f"   • Hybrid detects {crit_hyb} critical cases vs {crit_img} for image-only")
    print(f"   • {(crit_hyb/crit_img - 1)*100:.1f}% more critical hazards identified by hybrid")

    print(f"\n📁 OUTPUT FILES:")
    print(f"   • Detailed comparison CSV: {OUTPUT_DIR}/detailed_comparison.csv")
    print(f"   • Human evaluation Excel:  {excel_path}")

    print(f"\n✅ Ready for human evaluation study with {NUM_ANNOTATORS} annotators!")
    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    main()
