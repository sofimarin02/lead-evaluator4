import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


# ------------------------------
# 1) CARGA DE DATOS
# ------------------------------
url = "https://raw.githubusercontent.com/selva86/datasets/master/bank-additional-full.csv"
df = pd.read_csv(url, sep=';')
print(df.head(10))
print("Columnas:", df.columns.tolist())

# ------------------------------
# 2) PREPROCESAMIENTO
# ------------------------------
# Binning de edad
df['age_group'] = pd.cut(
    df['age'],
    bins=[17, 29, 49, 100],
    labels=['<30', '30-50', '>50']
)

# Selección de features coincidiendo con el formulario
feature_cols = [
    'age_group', 'job', 'marital', 'education', 'default',
    'housing', 'loan', 'poutcome',
    'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',
    'duration', 'emp.var.rate', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'nr.employed'
]

X = pd.get_dummies(df[feature_cols], drop_first=True)
y = (df['y'] == 'yes').astype(int)

# ------------------------------
# 3) SPLIT TRAIN / TEST
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 4) GRID SEARCH CV
# ------------------------------
param_grid = {
    'n_estimators':    [100, 300, 500],
    'max_depth':       [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight':    ['balanced', None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("Mejor combo de hiperparámetros:", grid.best_params_)

model = grid.best_estimator_

# ------------------------------
# 5) SMOTE PARA DESEQUILIBRIO
# ------------------------------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

model.fit(X_res, y_res)

# ------------------------------
# 6) PREDICCIÓN Y MÉTRICAS
# ------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# DEBUG: ejemplo real positivo
yes_example = X_test[y_test == 1].iloc[0].values.reshape(1, -1)
print("Ejemplo positivo (feature vector):")
print(dict(zip(X.columns, yes_example.flatten())))
print("Proba ejemplo:", model.predict_proba(yes_example)[0, 1])

# Cálculo de precision–recall y umbrales
prec, rec, ths = precision_recall_curve(y_test, y_proba)
# F1 para cada umbral (omitimos el último par que no corresponde a un threshold)
f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = ths[best_idx]
print(f"Mejor umbral según F1: {best_threshold:.3f}")

# Estadísticas de probabilidad
print("Min proba:", y_proba.min())
print("Max proba:", y_proba.max())
print("Percentiles (50,75,90,95,99):", np.percentile(y_proba, [50, 75, 90, 95, 99]))

# Métricas clásicas
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# ------------------------------
# 7) CURVAS Y PLOTS
# ------------------------------
# Curva ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Importancia de variables
importances = model.feature_importances_
top10 = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:10]
plt.figure(figsize=(8, 6))
sns.barplot(x=top10.values, y=top10.index)
plt.title("Top 10 Variables más importantes")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.show()

# ------------------------------
# 8) GUARDAR MODELO
# ------------------------------
os.makedirs("model", exist_ok=True)
with open("model/lead_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo guardado en model/lead_model.pkl")
