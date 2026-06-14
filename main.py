import pandas as pd
import matplotlib.pyplot as plt

# =========================
# AGENTS
# =========================

from agents.data_agent import DataAgent
from agents.analysis_agent import AnalysisAgent
from agents.ml_agent import MLAgent
from agents.evaluation_agent import EvaluationAgent
from agents.report_agent import ReportAgent

# =========================
# SKLEARN
# =========================

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# =========================
# XGBOOST
# =========================

from xgboost import XGBClassifier

# =========================
# INITIALIZE AGENTS
# =========================

data_agent = DataAgent()
analysis_agent = AnalysisAgent()
ml_agent = MLAgent()
evaluation_agent = EvaluationAgent()
report_agent = ReportAgent()

# =========================
# LOAD DATA
# =========================

df = data_agent.load_data(
    "data/Crimes_-_2001_to_Present.csv"
)

print("Data Loaded Successfully")

# =========================
# SAMPLE DATA
# =========================

df = df.sample(
    50000,
    random_state=42
)

# =========================
# IMPORTANT COLUMNS
# =========================

df = df[
    [
        'Date',
        'Primary Type',
        'Latitude',
        'Longitude',
        'Arrest',
        'Location Description',
        'Domestic',
        'Beat',
        'District',
        'Ward',
        'Community Area'
    ]
]

# =========================
# CLEAN DATA
# =========================

df = df.dropna()

# =========================
# DATE CONVERSION
# =========================

df['Date'] = pd.to_datetime(
    df['Date'],
    errors='coerce'
)

df = df.dropna()

# =========================
# FEATURE ENGINEERING
# =========================

df['Hour'] = df['Date'].dt.hour

df['Month'] = df['Date'].dt.month

df['DayOfWeek'] = df['Date'].dt.dayofweek

df['Year'] = df['Date'].dt.year

df['IsWeekend'] = df['DayOfWeek'].apply(
    lambda x: 1 if x >= 5 else 0
)

# =========================
# TOP 3 CRIME CLASSES
# =========================

top_crimes = (
    df['Primary Type']
    .value_counts()
    .nlargest(3)
    .index
)

df = df[
    df['Primary Type'].isin(top_crimes)
]

print("\nTop Crime Types:")
print(df['Primary Type'].value_counts())

# =========================
# ENCODING
# =========================

# Encode Location
loc_encoder = LabelEncoder()

df['LocationCode'] = loc_encoder.fit_transform(
    df['Location Description'].astype(str)
)

# Encode Target
encoder = LabelEncoder()

df['Crime_Type'] = encoder.fit_transform(
    df['Primary Type']
)

# =========================
# ANALYSIS
# =========================

analysis_agent.analyze_crime_types(df)

analysis_agent.analyze_by_hour(df)

# =========================
# FEATURES & TARGET
# =========================

X = df[
    [
        'Latitude',
        'Longitude',
        'Hour',
        'Month',
        'DayOfWeek',
        'Year',
        'Arrest',
        'LocationCode',
        'IsWeekend',
        'Domestic',
        'Beat',
        'District',
        'Ward',
        'Community Area'
    ]
]

y = df['Crime_Type']

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODELS
# =========================

models = {

    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
}

# =========================
# MODEL TRAINING
# =========================

best_model = None

best_accuracy = 0

model_names = []

accuracies = []

print("\n=== MODEL COMPARISON ===")

for name, model in models.items():

    model.fit(X_train, y_train)

    pred_temp = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred_temp
    )

    print(f"{name} Accuracy: {acc}")

    model_names.append(name)

    accuracies.append(acc * 100)

    if acc > best_accuracy:

        best_accuracy = acc

        best_model = model

# =========================
# BEST MODEL
# =========================

print("\nBest Model Selected:")
print(best_model)

print("\nBest Accuracy:")
print(best_accuracy)

# =========================
# CROSS VALIDATION
# =========================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=skf
)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nAverage CV Accuracy:")
print(cv_scores.mean())

# =========================
# FINAL PREDICTION
# =========================

pred = best_model.predict(X_test)

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n=== CLASSIFICATION REPORT ===\n")

print(
    classification_report(
        y_test,
        pred,
        target_names=encoder.classes_
    )
)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(cmap='Blues')

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()

plt.show()

# =========================
# ACCURACY BAR CHART
# =========================

plt.figure(figsize=(8,5))

plt.bar(
    model_names,
    accuracies
)

plt.ylabel("Accuracy (%)")

plt.title(
    "Model Accuracy Comparison"
)

plt.tight_layout()

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================

if hasattr(best_model, 'feature_importances_'):

    importance = best_model.feature_importances_

    features = [
        'Latitude',
        'Longitude',
        'Hour',
        'Month',
        'DayOfWeek',
        'Year',
        'Arrest',
        'LocationCode',
        'IsWeekend',
        'Domestic',
        'Beat',
        'District',
        'Ward',
        'Community Area'
    ]

    plt.figure(figsize=(8,5))

    pd.Series(
        importance,
        index=features
    ).sort_values().plot(
        kind='barh'
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.show()

# =========================
# REPORT GENERATION
# =========================

report_agent.generate_report(df)

print("\nPROJECT COMPLETED SUCCESSFULLY")