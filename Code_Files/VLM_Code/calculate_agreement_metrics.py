"""
Calculate inter-model agreement metrics
"""
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from test_config import OUTPUT_DIR
  

def load_predictions(model_name):
    """Load model predictions"""
    filename = OUTPUT_DIR / f"{model_name.replace(':', '_').replace('.', '_')}_predictions.json"
    with open(filename) as f:
        return json.load(f)

def calculate_pairwise_kappa():
    """Calculate Cohen's Kappa between all model pairs"""
    
    # Load predictions
    flash_preds = load_predictions("gemini-2.5-flash")
    pro_preds = load_predictions("gemini-2.5-pro")
    gpt5_preds = load_predictions("gpt-5")
    
    # Get common pothole IDs
    common_ids = set(flash_preds.keys()) & set(pro_preds.keys()) & set(gpt5_preds.keys())
    common_ids = sorted(common_ids)
    
    print(f"Common samples: {len(common_ids)}")
    
    # Extract labels
    flash_labels = [flash_preds[pid]['depth_category'] for pid in common_ids]
    pro_labels = [pro_preds[pid]['depth_category'] for pid in common_ids]
    gpt5_labels = [gpt5_preds[pid]['depth_category'] for pid in common_ids]
    
    # Calculate pairwise Cohen's Kappa
    results = {}
    
    pairs = [
        ("Gemini Flash", "Gemini Pro", flash_labels, pro_labels),
        ("Gemini Flash", "GPT-5", flash_labels, gpt5_labels),
        ("Gemini Pro", "GPT-5", pro_labels, gpt5_labels)
    ]
    
    for model1, model2, labels1, labels2 in pairs:
        kappa = cohen_kappa_score(labels1, labels2)
        agreement = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2) / len(labels1)
        
        results[f"{model1} vs {model2}"] = {
            "Cohen's Kappa": kappa,
            "Agreement %": agreement * 100
        }
        
        print(f"\n{model1} vs {model2}:")
        print(f"  Cohen's Kappa: {kappa:.3f}")
        print(f"  Agreement: {agreement*100:.1f}%")
    
    # Save results
    with open(OUTPUT_DIR / "inter_model_agreement.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return flash_labels, pro_labels, gpt5_labels, common_ids

def create_confusion_matrices(flash_labels, pro_labels, gpt5_labels):
    """Create and save confusion matrices"""
    
    categories = ['low', 'medium', 'high']
    
    pairs = [
        ("Gemini Flash", "Gemini Pro", flash_labels, pro_labels),
        ("Gemini Flash", "GPT-5", flash_labels, gpt5_labels),
        ("Gemini Pro", "GPT-5", pro_labels, gpt5_labels)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model1, model2, labels1, labels2) in enumerate(pairs):
        cm = confusion_matrix(labels1, labels2, labels=categories)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=categories, yticklabels=categories,
                    ax=axes[idx], cbar_kws={'label': 'Proportion'})
        
        axes[idx].set_title(f'{model1}\nvs\n{model2}')
        axes[idx].set_ylabel(model1)
        axes[idx].set_xlabel(model2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved confusion matrices: {OUTPUT_DIR / 'confusion_matrices.png'}")

def majority_voting(flash_labels, pro_labels, gpt5_labels, common_ids):
    """Create majority vote labels"""
    
    majority_labels = {}
    
    for i, pid in enumerate(common_ids):
        votes = [flash_labels[i], pro_labels[i], gpt5_labels[i]]
        
        # Count votes
        vote_counts = {cat: votes.count(cat) for cat in set(votes)}
        
        # Majority vote (if tie, use Flash as default)
        majority = max(vote_counts.items(), key=lambda x: (x[1], votes.index(x[0])))[0]
        
        agreement_level = max(vote_counts.values())
        confidence = "high" if agreement_level == 3 else "medium" if agreement_level == 2 else "low"
        
        majority_labels[pid] = {
            'depth_category': majority,
            'confidence': confidence,
            'votes': {
                'flash': flash_labels[i],
                'pro': pro_labels[i],
                'gpt5': gpt5_labels[i]
            }
        }
    
    # Save
    with open(OUTPUT_DIR / "majority_vote_labels.json", 'w') as f:
        json.dump(majority_labels, f, indent=2)
    
    print(f"\nSaved majority vote labels: {OUTPUT_DIR / 'majority_vote_labels.json'}")
    
    # Statistics
    confidence_dist = {}
    for v in majority_labels.values():
        conf = v['confidence']
        confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
    
    print("\nMajority Vote Confidence Distribution:")
    for conf, count in confidence_dist.items():
        print(f"  {conf}: {count} ({count/len(majority_labels)*100:.1f}%)")

if __name__ == "__main__":
    print("="*60)
    print("INTER-MODEL AGREEMENT ANALYSIS")
    print("="*60)
    
    flash_labels, pro_labels, gpt5_labels, common_ids = calculate_pairwise_kappa()
    create_confusion_matrices(flash_labels, pro_labels, gpt5_labels)
    majority_voting(flash_labels, pro_labels, gpt5_labels, common_ids)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
