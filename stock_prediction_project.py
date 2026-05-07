"""
Stock Price Prediction using Machine Learning
==============================================
A comprehensive ML project for predicting stock price movements
using technical indicators and macroeconomic factors.

Features:
- Exploratory Data Analysis (EDA)
- Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, SVM)
- Model evaluation and comparison
- Feature importance analysis
- Visualizations and performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================

print("=" * 70)
print("STOCK PRICE PREDICTION - ML PROJECT")
print("=" * 70)

# Load the data
df = pd.read_csv('stock_data.csv')

print("\n📊 DATA OVERVIEW")
print("-" * 70)
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]:,}")
print(f"Total Features: {df.shape[1]}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum().sum()} missing values found")

# Statistical summary
print("\n📈 STATISTICAL SUMMARY")
print("-" * 70)
print(df.describe().round(4))

# Class distribution
print("\n🎯 TARGET VARIABLE DISTRIBUTION")
print("-" * 70)
class_dist = df['Target'].value_counts()
print(f"Class 0 (Price Down): {class_dist[0]:,} ({class_dist[0]/len(df)*100:.2f}%)")
print(f"Class 1 (Price Up): {class_dist[1]:,} ({class_dist[1]/len(df)*100:.2f}%)")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Correlation analysis
print("\n🔗 FEATURE CORRELATION WITH TARGET")
print("-" * 70)
correlations = df.corr()['Target'].sort_values(ascending=False)
print(correlations)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Class Distribution
ax1 = axes[0, 0]
df['Target'].value_counts().plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Target (0=Down, 1=Up)')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['Down (0)', 'Up (1)'], rotation=0)

# 2. Correlation heatmap
ax2 = axes[0, 1]
top_features = correlations.abs().nlargest(10).index
sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            ax=ax2, cbar_kws={'label': 'Correlation'})
ax2.set_title('Top 10 Features Correlation Matrix', fontsize=14, fontweight='bold')

# 3. Feature distributions
ax3 = axes[1, 0]
df[['Open', 'Close', 'High', 'Low']].boxplot(ax=ax3)
ax3.set_title('Price Metrics Distribution', fontsize=14, fontweight='bold')
ax3.set_ylabel('Normalized Price')

# 4. Target vs Key Features
ax4 = axes[1, 1]
feature_corr = correlations.drop('Target').abs().nlargest(5)
feature_corr.plot(kind='barh', ax=ax4, color='#95E1D3')
ax4.set_title('Top 5 Features Correlated with Target', fontsize=14, fontweight='bold')
ax4.set_xlabel('Absolute Correlation')

plt.tight_layout()
plt.savefig('01_EDA_Analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: 01_EDA_Analysis.png")
plt.close()

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================

print("\n\n" + "=" * 70)
print("DATA PREPARATION & MODEL TRAINING")
print("=" * 70)

# Separate features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📂 DATA SPLIT")
print("-" * 70)
print(f"Training Set: {X_train.shape[0]:,} samples")
print(f"Testing Set: {X_test.shape[0]:,} samples")
print(f"Features: {X_train.shape[1]}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
}

results = {}

print("\n\n🤖 TRAINING MODELS")
print("-" * 70)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train on appropriate data
    if model_name == 'Random Forest':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Logistic Regression and SVM work better with scaled data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"✅ {model_name}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

# ============================================================================
# 5. MODEL COMPARISON VISUALIZATION
# ============================================================================

print("\n\n📊 MODEL COMPARISON")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Performance metrics comparison
metrics_df = pd.DataFrame({
    model_name: {
        'Accuracy': results[model_name]['accuracy'],
        'Precision': results[model_name]['precision'],
        'Recall': results[model_name]['recall'],
        'F1-Score': results[model_name]['f1'],
        'ROC-AUC': results[model_name]['roc_auc']
    }
    for model_name in results.keys()
}).T

# 1. Metrics comparison
ax1 = axes[0, 0]
metrics_df.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_xlabel('Model')
ax1.legend(loc='lower right')
ax1.set_ylim([0.5, 1.0])
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. ROC Curves
ax2 = axes[0, 1]
for model_name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_pred_proba'])
    ax2.plot(fpr, tpr, label=f"{model_name} (AUC={results[model_name]['roc_auc']:.3f})", linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3)

# 3. Confusion Matrices
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]

ax3 = axes[1, 0]
cm = confusion_matrix(y_test, best_model['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar_kws={'label': 'Count'})
ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# 4. F1-Score comparison
ax4 = axes[1, 1]
f1_scores = {name: results[name]['f1'] for name in results.keys()}
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
ax4.barh(list(f1_scores.keys()), list(f1_scores.values()), color=colors)
ax4.set_xlabel('F1-Score')
ax4.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax4.set_xlim([0.5, 1.0])
for i, (name, score) in enumerate(f1_scores.items()):
    ax4.text(score + 0.01, i, f'{score:.4f}', va='center')

plt.tight_layout()
plt.savefig('02_Model_Comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: 02_Model_Comparison.png")
plt.close()

# ============================================================================
# 6. FEATURE IMPORTANCE (Random Forest)
# ============================================================================

print("\n\n🎯 FEATURE IMPORTANCE ANALYSIS")
print("-" * 70)

rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature importance bar plot
ax1 = axes[0]
top_n = 10
ax1.barh(feature_importance['feature'][:top_n][::-1], 
         feature_importance['importance'][:top_n][::-1], color='#4ECDC4')
ax1.set_xlabel('Importance Score')
ax1.set_title(f'Top {top_n} Important Features (Random Forest)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Cumulative importance
ax2 = axes[1]
cumsum = feature_importance['importance'].cumsum() / feature_importance['importance'].sum() * 100
ax2.plot(range(1, len(cumsum) + 1), cumsum, marker='o', linewidth=2, markersize=4)
ax2.axhline(y=80, color='r', linestyle='--', label='80% Cumulative Importance')
ax2.axhline(y=90, color='orange', linestyle='--', label='90% Cumulative Importance')
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Cumulative Importance (%)')
ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('03_Feature_Importance.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: 03_Feature_Importance.png")
plt.close()

# ============================================================================
# 7. DETAILED CLASSIFICATION REPORT
# ============================================================================

print("\n\n📋 DETAILED CLASSIFICATION REPORT")
print("-" * 70)
print(f"\nBest Performing Model: {best_model_name}")
print("\n" + classification_report(y_test, best_model['y_pred'], 
                                 target_names=['Price Down (0)', 'Price Up (1)']))

# ============================================================================
# 8. PREDICTIONS ON NEW DATA
# ============================================================================

print("\n\n🔮 SAMPLE PREDICTIONS")
print("-" * 70)

sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_predictions = pd.DataFrame({
    'Actual': y_test.iloc[sample_indices].values,
    'Predicted': best_model['y_pred'][sample_indices],
    'Probability (Up)': best_model['y_pred_proba'][sample_indices].round(4),
    'Confidence': np.abs(best_model['y_pred_proba'][sample_indices] - 0.5) * 2
}).round(4)

print(sample_predictions.to_string())

# ============================================================================
# 9. SUMMARY REPORT
# ============================================================================

print("\n\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
print(f"""
✅ ANALYSIS COMPLETED SUCCESSFULLY

📊 Dataset Information:
   - Total Records: {len(df):,}
   - Features: {len(X.columns)}
   - Target Classes: 2 (Binary Classification)
   - Class Balance: {class_dist[0]/len(df)*100:.1f}% vs {class_dist[1]/len(df)*100:.1f}%

🤖 Models Trained:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine

🏆 Best Model: {best_model_name}
   - Accuracy:  {best_model['accuracy']:.4f}
   - Precision: {best_model['precision']:.4f}
   - Recall:    {best_model['recall']:.4f}
   - F1-Score:  {best_model['f1']:.4f}
   - ROC-AUC:   {best_model['roc_auc']:.4f}

📈 Visualizations Generated:
   1. 01_EDA_Analysis.png - Data exploration
   2. 02_Model_Comparison.png - Model performance
   3. 03_Feature_Importance.png - Feature analysis

💡 Key Insights:
   - Top Feature: {feature_importance['feature'].iloc[0]}
   - Features needed for 80% importance: {(cumsum <= 80).sum()}
   - Model predicts price movements with {best_model['accuracy']*100:.2f}% accuracy

🚀 Next Steps:
   - Deploy best model for live predictions
   - Fine-tune hyperparameters
   - Collect more recent data
   - Monitor model performance over time
""")

print("=" * 70)
print("✅ All processes completed successfully!")
print("=" * 70)
