import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

class ChurnModelBackend:
    """Service d'inférence et logique métier pour la prédiction de churn."""

    def __init__(self, model_path=None, preprocessor_path=None):
        app_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(app_dir)
        
        # Chercher dans app/ en priorité, sinon à la racine
        default_model = os.path.join(app_dir, "churn_mlp_4f.keras")
        if not os.path.exists(default_model):
            default_model = os.path.join(base_dir, "churn_mlp_4f.keras")

        default_preproc = os.path.join(app_dir, "preprocessor_4f.joblib")
        if not os.path.exists(default_preproc):
            default_preproc = os.path.join(base_dir, "preprocessor_4f.joblib")

        self.model_path = model_path or default_model
        self.preprocessor_path = preprocessor_path or default_preproc
        
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Charge le modèle et le préprocesseur s'ils existent, sinon les entraîne."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            import train_mlp_4features
            train_mlp_4features.train_and_save()

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, contract: str, tenure: float, internet_service: str, payment_method: str) -> dict:
        """
        Prend les 4 variables discriminantes et retourne le diagnostic complet.
        """
        # 1. Préparation des données d'entrée
        input_df = pd.DataFrame([{
            'Contract': contract,
            'tenure': float(tenure),
            'InternetService': internet_service,
            'PaymentMethod': payment_method
        }])

        # 2. Transformation via ColumnTransformer
        X_proc = self.preprocessor.transform(input_df).astype(np.float32)

        # 3. Prédiction via le MLP
        raw_proba = float(self.model.predict(X_proc, verbose=0)[0][0])
        proba_percent = round(raw_proba * 100, 1)

        # 4. Diagnostic de niveau de risque
        if raw_proba >= 0.50:
            risk_level = "HIGH"
            badge_label = f" RISQUE ÉLEVÉ ({proba_percent}%)"
            message = "Forte propension à résilier dans les prochaines semaines. Action de rétention immédiate requise."
        elif raw_proba >= 0.35:
            risk_level = "MEDIUM"
            badge_label = f" RISQUE MODÉRÉ ({proba_percent}%)"
            message = "Signaux faibles de désengagement détectés. Profil à surveiller attentivement."
        else:
            risk_level = "LOW"
            badge_label = f" CLIENT FIDÈLE ({proba_percent}%)"
            message = "Profil stable avec un attachement solide à l'opérateur. Faible risque de départ."

        # 5. Leviers d'action & recommandations
        recommendations = self._generate_recommendations(contract, tenure, internet_service, payment_method)

        return {
            "probability": raw_proba,
            "probability_percent": proba_percent,
            "risk_level": risk_level,
            "badge_label": badge_label,
            "message": message,
            "recommendations": recommendations
        }

    def _generate_recommendations(self, contract: str, tenure: float, internet: str, payment: str) -> list[str]:
        """Génère des actions concrètes basées sur les règles métier identifiées lors de l'EDA."""
        recos = []
        if contract == "Month-to-month":
            recos.append(" **Contrat sans engagement :** Proposer une formule d'engagement annuel avec -15% ou 1 mois offert.")
        else:
            recos.append(" **Contrat sécurisé :** Le client est engagé, son statut contractuel est protecteur.")

        if tenure <= 12:
            recos.append(" **Période d'intégration critique (< 1 an) :** Planifier un appel de bienvenue pour vérifier la satisfaction.")

        if internet == "Fiber optic":
            recos.append(" **Fibre Optique sensible :** Effectuer un audit technique proactif de la ligne et s'assurer de la stabilité du débit.")

        if payment == "Electronic check":
            recos.append("**Friction de paiement :** Inciter au prélèvement automatique (ex: offrir 10€ de crédit de bienvenue sur facture).")

        return recos

# Instance singleton réutilisable
_backend_instance = None

def get_backend_service() -> ChurnModelBackend:
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = ChurnModelBackend()
    return _backend_instance
