# train_model.py - Train URL tracking detection model (Decision Tree Edition 🌳✨)

import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
)

# Load data
df = pd.read_csv("../data/urls.csv")

# Preprocess URLs
def preprocess_url(url):
    return re.sub(r"https?://(www\.)?", "", url)

df['processed_url'] = df['url'].apply(preprocess_url)

# Feature extraction - TF-IDF on character n-grams
vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
X = vectorizer.fit_transform(df['processed_url'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
max_depth = 5
model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Decision Tree Accuracy (depth={max_depth}): {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("📊 Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    filled=True,
    feature_names=vectorizer.get_feature_names_out(),
    class_names=[str(cls) for cls in model.classes_],
    max_depth=5
)
plt.title("🌳 Decision Tree Visualization (Top 2 Levels)")
plt.tight_layout()
plt.show()

# F1 Score vs Depth Curve
depths = list(range(1, 21))
f1_scores = []

for d in depths:
    temp_model = DecisionTreeClassifier(max_depth=d, random_state=42)
    temp_model.fit(X_train, y_train)
    y_pred_temp = temp_model.predict(X_test)
    score = f1_score(y_test, y_pred_temp, average='weighted')  # supports multiclass
    f1_scores.append(score)

plt.figure(figsize=(10, 5))
sns.lineplot(x=depths, y=f1_scores, marker="o")
plt.title("📈 F1 Score vs Decision Tree Depth")
plt.xlabel("Tree Depth")
plt.ylabel("Weighted F1 Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save model & vectorizer
joblib.dump(model, "../models/tracker_detection_tree_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")
print("✅ Decision Tree model and vectorizer saved successfully.")


param_grid = {
    "max_depth": list(range(2, 21)),  # Try trees from depth 2 to 20
    "min_samples_split": [2, 5, 10]   # Try stricter node splits
}

# Initialize base model
dt = DecisionTreeClassifier(random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    scoring='f1_weighted',  # or use 'accuracy' if that's your target
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Run grid search
grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
print("🌟 Best Parameters found via GridSearchCV:")
print(grid_search.best_params_)
print(f"✅ Best F1 Score on training (CV): {grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred_best = best_model.predict(X_test)
f1_best = f1_score(y_test, y_pred_best, average='weighted')
print(f"🎯 Test Set F1 Score with best model: {f1_best:.4f}")

# Save best model
joblib.dump(best_model, "../models/decision_tree_best_model.pkl")


results_df = pd.DataFrame(grid_search.cv_results_)

# Pivot to matrix for heatmap
heatmap_data = results_df.pivot(
    index="param_max_depth",
    columns="param_min_samples_split",
    values="mean_test_score"
)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("F1 Score Heatmap (GridSearchCV)")
plt.xlabel("min_samples_split")
plt.ylabel("max_depth")
plt.tight_layout()
plt.show()