"""
Calculate human inter-annotator agreement and human-VLM agreement
"""
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import json
from pathlib import Path
from test_config import OUTPUT_DIR
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_human_annotations():
    """
    Load completed human annotations and calculate all metrics
    """
    
    # Load annotated Excel file
    df = pd.read_excel(OUTPUT_DIR / 'human_annotation_template.xlsx', sheet_name='Annotations')
    
    # Remove rows with missing annotations
    df = df.dropna(subset=['Annotator_1_Depth', 'Annotator_2_Depth'])
    
    # Check if we have 3rd annotator
    has_annotator_3 = df['Annotator_3_Depth'].notna().any()
    
    print(f"\n{'='*60}")
    print(f"HUMAN ANNOTATION ANALYSIS")
    print(f"{'='*60}")
    print(f"\nAnalyzing {len(df)} annotated samples...")
    print(f"Annotators: {3 if has_annotator_3 else 2}")
    
    # Extract labels
    human1 = df['Annotator_1_Depth'].str.lower().values
    human2 = df['Annotator_2_Depth'].str.lower().values
    flash = df['Flash_Prediction'].str.lower().values
    pro = df['Pro_Prediction'].str.lower().values
    gpt5 = df['GPT5_Prediction'].str.lower().values
    
    # 1. HUMAN INTER-ANNOTATOR AGREEMENT
    print(f"\n{'='*60}")
    print("1. HUMAN INTER-ANNOTATOR AGREEMENT")
    print(f"{'='*60}")
    
    h1_h2_kappa = cohen_kappa_score(human1, human2)
    h1_h2_agreement = accuracy_score(human1, human2)
    
    print(f"\nAnnotator 1 vs Annotator 2:")
    print(f"  Cohen's Kappa: {h1_h2_kappa:.3f}")
    print(f"  Agreement: {h1_h2_agreement*100:.1f}%")
    
    if has_annotator_3:
        human3 = df['Annotator_3_Depth'].str.lower().values
        h1_h3_kappa = cohen_kappa_score(human1, human3)
        h2_h3_kappa = cohen_kappa_score(human2, human3)
        h1_h3_agreement = accuracy_score(human1, human3)
        h2_h3_agreement = accuracy_score(human2, human3)
        
        print(f"\nAnnotator 1 vs Annotator 3:")
        print(f"  Cohen's Kappa: {h1_h3_kappa:.3f}")
        print(f"  Agreement: {h1_h3_agreement*100:.1f}%")
        
        print(f"\nAnnotator 2 vs Annotator 3:")
        print(f"  Cohen's Kappa: {h2_h3_kappa:.3f}")
        print(f"  Agreement: {h2_h3_agreement*100:.1f}%")
        
        avg_kappa = np.mean([h1_h2_kappa, h1_h3_kappa, h2_h3_kappa])
        print(f"\nAverage Inter-Annotator Kappa: {avg_kappa:.3f}")
    
    # Kappa interpretation
    kappa_interp = (
        "Poor" if h1_h2_kappa < 0.20 else
        "Fair" if h1_h2_kappa < 0.40 else
        "Moderate" if h1_h2_kappa < 0.60 else
        "Substantial" if h1_h2_kappa < 0.80 else
        "Almost Perfect"
    )
    print(f"Interpretation: {kappa_interp} agreement")
    
    # 2. CREATE CONSENSUS HUMAN LABELS
    print(f"\n{'='*60}")
    print("2. HUMAN CONSENSUS LABELS")
    print(f"{'='*60}")
    
    consensus_human = []
    confidence_dist = {'unanimous': 0, 'majority': 0, 'split': 0}
    
    for i in range(len(df)):
        if has_annotator_3:
            votes = [human1[i], human2[i], human3[i]]
        else:
            votes = [human1[i], human2[i]]
        
        # Count votes
        vote_counts = {cat: votes.count(cat) for cat in set(votes)}
        majority = max(vote_counts.items(), key=lambda x: x[1])
        
        consensus_human.append(majority[0])
        
        if majority[1] == len(votes):
            confidence_dist['unanimous'] += 1
        elif majority[1] > 1:
            confidence_dist['majority'] += 1
        else:
            confidence_dist['split'] += 1
    
    consensus_human = np.array(consensus_human)
    
    print(f"\nConsensus Distribution:")
    for level, count in confidence_dist.items():
        print(f"  {level.capitalize()}: {count} ({count/len(df)*100:.1f}%)")
    
    # 3. HUMAN-VLM AGREEMENT
    print(f"\n{'='*60}")
    print("3. HUMAN-VLM AGREEMENT")
    print(f"{'='*60}")
    
    vlm_results = {}
    
    for model_name, model_labels in [('Gemini Flash', flash), 
                                      ('Gemini Pro', pro), 
                                      ('GPT-5', gpt5)]:
        kappa = cohen_kappa_score(consensus_human, model_labels)
        agreement = accuracy_score(consensus_human, model_labels)
        
        print(f"\n{model_name} vs Human Consensus:")
        print(f"  Cohen's Kappa: {kappa:.3f}")
        print(f"  Agreement: {agreement*100:.1f}%")
        
        vlm_results[model_name] = {
            'kappa': float(kappa),
            'agreement_percent': float(agreement * 100)
        }
    
    # 4. CONFUSION MATRICES
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    categories = ['low', 'medium', 'high']
    
    for idx, (model_name, model_labels) in enumerate([('Gemini Flash', flash), 
                                                       ('Gemini Pro', pro), 
                                                       ('GPT-5', gpt5)]):
        cm = confusion_matrix(consensus_human, model_labels, labels=categories)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=categories, yticklabels=categories,
                    ax=axes[idx], cbar_kws={'label': 'Proportion'})
        
        axes[idx].set_title(f'{model_name}\nvs Human Consensus')
        axes[idx].set_ylabel('Human Consensus')
        axes[idx].set_xlabel('VLM Prediction')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'human_vlm_confusion_matrices.png', dpi=300)
    print(f"\n✓ Saved confusion matrices: {OUTPUT_DIR / 'human_vlm_confusion_matrices.png'}")
    
    # 5. SAVE RESULTS
    results = {
        'num_samples': int(len(df)),
        'num_annotators': 3 if has_annotator_3 else 2,
        'human_inter_annotator': {
            'h1_vs_h2': {
                'kappa': float(h1_h2_kappa),
                'agreement_percent': float(h1_h2_agreement * 100)
            }
        },
        'consensus_distribution': {k: int(v) for k, v in confidence_dist.items()},
        'human_vlm_agreement': vlm_results
    }
    
    if has_annotator_3:
        results['human_inter_annotator'].update({
            'h1_vs_h3': {'kappa': float(h1_h3_kappa), 'agreement_percent': float(h1_h3_agreement * 100)},
            'h2_vs_h3': {'kappa': float(h2_h3_kappa), 'agreement_percent': float(h2_h3_agreement * 100)},
            'average_kappa': float(avg_kappa)
        })
    
    with open(OUTPUT_DIR / 'human_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {OUTPUT_DIR / 'human_validation_results.json'}")
    
    # 6. RECOMMENDATION
    print(f"\n{'='*60}")
    print("MODEL RECOMMENDATION")
    print(f"{'='*60}")
    
    best_model = max(vlm_results.items(), key=lambda x: x[1]['kappa'])
    print(f"\nBest VLM (highest human agreement):")
    print(f"  {best_model[0]}: Kappa={best_model[1]['kappa']:.3f}, Agreement={best_model[1]['agreement_percent']:.1f}%")
    
    return results

if __name__ == "__main__":
    analyze_human_annotations()
