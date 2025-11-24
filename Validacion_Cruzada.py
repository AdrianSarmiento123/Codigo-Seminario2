import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("cicids2017_cleaned.csv")

label_col = "Attack Type"
df[label_col] = df[label_col].astype(str)
df["y"] = df[label_col].apply(lambda x: 0 if x.upper().strip() in ["BENIGN", "NORMAL", "NORMAL TRAFFIC"] else 1)

features = [
 "Destination Port","Flow Duration",
 "Total Fwd Packets","Total Length of Fwd Packets","Fwd Packet Length Max","Fwd Packet Length Min",
 "Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min",
 "Bwd Packet Length Mean","Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s",
 "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
 "Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
 "Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
 "Fwd Packets/s","Bwd Packets/s",
 "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
 "FIN Flag Count","PSH Flag Count","ACK Flag Count",
 "Average Packet Size","Active Mean","Active Max","Active Min","Idle Mean","Idle Max","Idle Min"
]

features = [f for f in features if f in df.columns]
X = df[features].fillna(0).astype(float)
y = df["y"].astype(int)

# Bateria de modelos
# La bateria de modelos :
# El presente código fue generado usando la IA ChatGPT, el prompt fue el siguiente "como probar todos los modelos a la vez en un solo codigo".

modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=100, n_jobs=-1, random_state=42, class_weight="balanced"),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True),
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(max_depth=50, random_state=42, class_weight="balanced"),
    "DNN (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=100, random_state=42)
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
detallado = []

for nombre, modelo in modelos.items():
    print(f"\n Validación cruzada: {nombre}")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_ts = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_ts = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", modelo)])

        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        tiempo = time.time() - t0

        y_pred = pipe.predict(X_ts)
        y_prob = pipe.predict_proba(X_ts)[:,1] if hasattr(pipe, "predict_proba") else y_pred

        detallado.append({
            "Modelo": nombre,
            "Fold": fold,
            "Accuracy": accuracy_score(y_ts, y_pred),
            "Precision": precision_score(y_ts, y_pred),
            "Recall": recall_score(y_ts, y_pred),
            "F1-score": f1_score(y_ts, y_pred),
            "ROC-AUC": roc_auc_score(y_ts, y_prob),
            "Tiempo (s)": tiempo
        })

df_detallado = pd.DataFrame(detallado)

resumen = df_detallado.groupby("Modelo").agg({
    "Accuracy": ['mean', 'std'],
    "Precision": ['mean', 'std'],
    "Recall": ['mean', 'std'],
    "F1-score": ['mean', 'std'],
    "ROC-AUC": ['mean', 'std'],
    "Tiempo (s)": ['mean', 'std']
}).reset_index()

resumen.columns = [
    "Modelo",
    "Accuracy_mean", "Accuracy_std",
    "Precision_mean", "Precision_std",
    "Recall_mean", "Recall_std",
    "F1-score_mean", "F1-score_std",
    "ROC-AUC_mean", "ROC-AUC_std",
    "Tiempo_mean_s", "Tiempo_std_s"
]

with pd.ExcelWriter("validacion_cruzada_5fold.xlsx") as writer:
    df_detallado.to_excel(writer, sheet_name="Folds", index=False)
    resumen.to_excel(writer, sheet_name="Resumen", index=False)

print("\n Archivo generado: validacion_cruzada_5fold.xlsx")
print(resumen)
