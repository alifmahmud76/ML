
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore

from sklearn.svm import SVC  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore


# ==========================================
# 1. SETUP
# ==========================================

csv_file_path = "crop_recommendation.csv"
target_col = "label"

svm_c = 1.0
log_c = 1.0
knn_neighbors = 5
kmeans_clusters = 22


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading dataset...")

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print("ERROR: Dataset not found!")
    print("Place crop_recommendation.csv in the same folder as this Python file.")
    raise SystemExit

df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Number of crop classes:", y.nunique())


# ==========================================
# 3. PREPROCESSING
# ==========================================

def build_preprocessor(X_data):

    numeric_cols = X_data.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X_data.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols
        ))

    return ColumnTransformer(transformers=transformers)


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. INITIALIZE MODELS
# ==========================================

models = {

    "SVM": SVC(
        kernel="linear",
        C=svm_c,
        probability=True,
        random_state=0
    ),

    "Logistic Regression": LogisticRegression(
        C=log_c,
        max_iter=2000,
        random_state=0
    ),

    "Random Forest": RandomForestClassifier(
        random_state=0
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=knn_neighbors
    )
}


# ==========================================
# 6. TRAIN MODELS AND CALCULATE ACCURACY
# ==========================================

results = {}
pipelines = {}

print("\nTraining machine learning models...\n")

for name, model in models.items():

    print("Training:", name)

    pipe = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", model)
    ])

    pipe.fit(X_train, y_train)

    predictions = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    pipelines[name] = pipe
    results[name] = accuracy


# ==========================================
# 7. K-MEANS CLUSTERING
# ==========================================

print("Training K-Means...")

kmeans_prep = build_preprocessor(X_train)

X_train_trans = kmeans_prep.fit_transform(X_train)

kmeans = KMeans(
    n_clusters=kmeans_clusters,
    random_state=0,
    n_init=10
)

train_clusters = kmeans.fit_predict(X_train_trans)

cluster_map = {}

for cid in range(kmeans_clusters):

    cluster_labels = y_train.iloc[
        train_clusters == cid
    ]

    if len(cluster_labels) > 0:
        cluster_map[cid] = cluster_labels.mode()[0]
    else:
        cluster_map[cid] = y_train.mode()[0]


X_test_trans = kmeans_prep.transform(X_test)

test_clusters = kmeans.predict(X_test_trans)

kmeans_predictions = [
    cluster_map[c] for c in test_clusters
]

results["K-Means"] = accuracy_score(
    y_test,
    kmeans_predictions
)


# ==========================================
# 8. DISPLAY ACCURACY RESULTS
# ==========================================

print("\n" + "=" * 45)
print("       MACHINE LEARNING MODEL ACCURACY")
print("=" * 45)

for name, accuracy in results.items():

    print(
        f"{name:25s}: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

print("=" * 45)

# ==========================================
# 9. MODEL ACCURACY COMPARISON CHART
# ==========================================

plt.figure(figsize=(10, 6))

model_names = list(results.keys())
accuracies = [value * 100 for value in results.values()]

bars = plt.bar(model_names, accuracies)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.xticks(rotation=15)

# Display accuracy values above bars
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=300)
plt.show()


# ==========================================
# 9. INDIVIDUAL CROP PREDICTION
# ==========================================

# ==========================================
# 9. MODEL ACCURACY COMPARISON CHART
# ==========================================

plt.figure(figsize=(10, 6))

model_names = list(results.keys())
accuracies = [value * 100 for value in results.values()]

bars = plt.bar(model_names, accuracies)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.xticks(rotation=15)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=300)

plt.show(block=True)


# ==========================================
# 10. K-MEANS PREDICTION
# ==========================================

# Use one held-out sample as the input for the example prediction.
new_data = X_test.iloc[[0]]

km_clusters = kmeans.predict(
    kmeans_prep.transform(new_data)
)

km_prediction = cluster_map[km_clusters[0]]

print(f"{'K-Means':25s}: {km_prediction}")


# ==========================================
# 11. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": list(results.values())
})

results_df["Accuracy (%)"] = (
    results_df["Accuracy"] * 100
)

results_df.to_csv(
    "crop_model_accuracy_results.csv",
    index=False
)

print("\nResults saved in crop_model_accuracy_results.csv")

print("\nPROGRAM COMPLETED SUCCESSFULLY!")
