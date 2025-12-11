"""
Generate Dataset statistics and visualizations
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
FLAT_FILE = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/final_risk_scores_flat.json")
OUTPUT_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
with open(FLAT_FILE) as f:
    data = json.load(f)

# Convert to DataFrame for analysis
records = []
for pid, record in data.items():
    records.append({
        'pothole_id': pid,
        'scene_id': record['scene_id'],
        'risk_score': record['risk_assessment']['composite_score'],
        'severity': record['risk_assessment']['severity_level'],
        'depth_category': record['depth']['category'],
        'depth_confidence': record['depth']['confidence'],
        'area': record['geometric']['area'],
        'circularity': record['geometric'].get('circularity'),
        'solidity': record['geometric'].get('solidity'),
        'distance': record['positional'].get('Distance_Category'),
        'lateral_position': record['positional'].get('Lateral_Position')
    })

df = pd.DataFrame(records)

# Save CSV for easy inspection
df.to_csv(OUTPUT_DIR / "dataset_full.csv", index=False)

print("="*60)
print("DATASET STATISTICS FOR PAPER")
print("="*60)

# 1. Overall Statistics
print("\n--- Overall Dataset ---")
print(f"Total potholes: {len(df)}")
print(f"Total scenes: {df['scene_id'].nunique()}")
print(f"Average potholes per scene: {len(df) / df['scene_id'].nunique():.2f}")

# 2. Depth Distribution
print("\n--- Depth Category Distribution ---")
depth_counts = df['depth_category'].value_counts()
for cat, count in depth_counts.items():
    pct = count / len(df) * 100
    print(f"  {cat:8s}: {count:4d} ({pct:5.1f}%)")

# 3. Risk Severity Distribution
print("\n--- Risk Severity Distribution ---")
severity_counts = df['severity'].value_counts()
for sev, count in severity_counts.items():
    pct = count / len(df) * 100
    print(f"  {sev:8s}: {count:4d} ({pct:5.1f}%)")

# 4. Confidence Distribution
print("\n--- VLM Confidence Distribution ---")
conf_counts = df['depth_confidence'].value_counts()
for conf, count in conf_counts.items():
    pct = count / len(df) * 100
    print(f"  {conf:8s}: {count:4d} ({pct:5.1f}%)")

# 5. Geometric Statistics
print("\n--- Geometric Features (Mean ± Std) ---")
print(f"  Area (pixels):   {df['area'].mean():.0f} ± {df['area'].std():.0f}")
print(f"  Circularity:     {df['circularity'].mean():.3f} ± {df['circularity'].std():.3f}")
print(f"  Solidity:        {df['solidity'].mean():.3f} ± {df['solidity'].std():.3f}")

# 6. Distance Distribution
print("\n--- Distance Category Distribution ---")
dist_counts = df['distance'].value_counts()
for dist, count in dist_counts.items():
    pct = count / len(df) * 100
    print(f"  {dist:15s}: {count:4d} ({pct:5.1f}%)")

# 7. Lateral Position Distribution
print("\n--- Lateral Position Distribution ---")
lat_counts = df['lateral_position'].value_counts()
for lat, count in lat_counts.items():
    pct = count / len(df) * 100
    print(f"  {lat:20s}: {count:4d} ({pct:5.1f}%)")

print("\n" + "="*60)

# ==================== VISUALIZATIONS ====================

sns.set_style("whitegrid")
sns.set_palette("muted")

# Figure 1: Depth Category Distribution
fig, ax = plt.subplots(figsize=(8, 5))
depth_counts = df['depth_category'].value_counts()
ax.bar(depth_counts.index, depth_counts.values, color=['#2ca02c', '#ff7f0e', '#d62728'])
ax.set_xlabel('Depth Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('VLM Depth Classification Distribution (n=3,092)', fontsize=14, fontweight='bold')
for i, (cat, count) in enumerate(depth_counts.items()):
    ax.text(i, count + 50, f'{count}\n({count/len(df)*100:.1f}%)', 
            ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_depth_distribution.png", dpi=300)
print(f"✓ Saved: fig1_depth_distribution.png")

# Figure 2: Risk Severity Distribution
fig, ax = plt.subplots(figsize=(8, 5))
severity_order = ['Low', 'Medium', 'High', 'Critical']
severity_counts = df['severity'].value_counts().reindex(severity_order)
colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
ax.bar(severity_counts.index, severity_counts.values, color=colors)
ax.set_xlabel('Risk Severity Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Composite Risk Score Distribution (n=3,092)', fontsize=14, fontweight='bold')
for i, (sev, count) in enumerate(severity_counts.items()):
    ax.text(i, count + 50, f'{count}\n({count/len(df)*100:.1f}%)', 
            ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_risk_severity.png", dpi=300)
print(f"✓ Saved: fig2_risk_severity.png")

# Figure 3: Risk Score Histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['risk_score'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(df['risk_score'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean = {df["risk_score"].mean():.1f}')
ax.set_xlabel('Composite Risk Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Composite Risk Scores (0-100)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_risk_histogram.png", dpi=300)
print(f"✓ Saved: fig3_risk_histogram.png")

# Figure 4: Depth vs Risk (Boxplot)
fig, ax = plt.subplots(figsize=(8, 6))
depth_order = ['low', 'medium', 'high']
df_ordered = df[df['depth_category'].isin(depth_order)]
sns.boxplot(data=df_ordered, x='depth_category', y='risk_score', 
            order=depth_order, palette=['#2ca02c', '#ff7f0e', '#d62728'], ax=ax)
ax.set_xlabel('Depth Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Composite Risk Score', fontsize=12, fontweight='bold')
ax.set_title('Risk Score by Depth Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig4_depth_vs_risk.png", dpi=300)
print(f"✓ Saved: fig4_depth_vs_risk.png")

# Figure 5: Confidence vs Depth
fig, ax = plt.subplots(figsize=(10, 6))
conf_depth = df.groupby(['depth_confidence', 'depth_category']).size().unstack(fill_value=0)
conf_depth.plot(kind='bar', stacked=False, ax=ax, color=['#2ca02c', '#ff7f0e', '#d62728'])
ax.set_xlabel('VLM Confidence Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('VLM Confidence Distribution by Depth Category', fontsize=14, fontweight='bold')
ax.legend(title='Depth', fontsize=10)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig5_confidence_by_depth.png", dpi=300)
print(f"✓ Saved: fig5_confidence_by_depth.png")

print("\n" + "="*60)
print("ALL FIGURES GENERATED")
print(f"Location: {OUTPUT_DIR}")
print("="*60)
