import os
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

def train_and_save():
    print("[INFO] Chargement des donnees...")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # 1. Sélection des 4 variables cibles
    features = ['Contract', 'tenure', 'InternetService', 'PaymentMethod']
    X = df[features].copy()
    y = df['Churn'].map({'Yes': 1, 'No': 0}).values

    # Nettoyage
    X['tenure'] = pd.to_numeric(X['tenure'], errors='coerce').fillna(0)

    # 2. Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Pipeline de transformation
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod']
    num_cols = ['tenure']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_proc = preprocessor.transform(X_test).astype(np.float32)

    # Sauvegarder le préprocesseur
    os.makedirs("app", exist_ok=True)
    joblib.dump(preprocessor, os.path.join("app", "preprocessor_4f.joblib"))
    print("[OK] Preprocesseur sauvegarde (app/preprocessor_4f.joblib)")

    # 4. Calcul des poids de classe pour le déséquilibre
    weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}

    # 5. Architecture MLP adaptée aux 4 variables
    input_dim = X_train_proc.shape[1]
    tf.random.set_seed(42)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # 6. Entraînement
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )

    model.fit(
        X_train_proc, y_train,
        validation_split=0.2,
        epochs=35,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Évaluation rapide
    test_loss, test_acc, test_auc = model.evaluate(X_test_proc, y_test, verbose=0)
    print(f"[EVAL] Test Accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

    # 8. Sauvegarder le modèle Keras
    model.save(os.path.join("app", "churn_mlp_4f.keras"))
    print("[OK] Modele Keras sauvegarde (app/churn_mlp_4f.keras)")

if __name__ == "__main__":
    train_and_save()
