import streamlit as st
from app.backend import get_backend_service

def inject_custom_css():
    """Injecte le style CSS pour l'application."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #64748B;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }
        .badge-high {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }
        .badge-med {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }
        .badge-low {
            background-color: #D1FAE5;
            color: #065F46;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }
        .reco-box {
            background: #F8FAFC;
            border-left: 4px solid #3B82F6;
            padding: 0.8rem 1.1rem;
            border-radius: 0 8px 8px 0;
            margin-top: 0.6rem;
            font-size: 0.95rem;
        }
    </style>
    """, unsafe_allow_html=True)

def render_app():
    """Point d'entrée principal du composant Frontend Streamlit."""
    st.set_page_config(
        page_title="Telco Churn Predictor — MLP",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    # En-tête
    st.markdown('<div class="main-header"> Telco Customer Churn Diagnostic Prédictif</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Modèle Réseau de Neurones (MLP) basé sur les 4 facteurs discriminants issus de notre EDA.</div>', unsafe_allow_html=True)

    col_inputs, col_results = st.columns([1, 1], gap="large")

    # Formulaire d'entrée
    with col_inputs:
        st.subheader(" Profil du Client")
        st.write("Renseignez les 4 paramètres clés :")

        contract = st.selectbox(
            "1. Type de Contrat (Driver #1)",
            options=["Month-to-month", "One year", "Two year"],
            index=0,
            help="Month-to-month présente le risque de départ le plus fort."
        )

        tenure = st.slider(
            "2. Ancienneté (Driver #2) : en mois",
            min_value=0,
            max_value=72,
            value=3,
            step=1,
            help="Le risque de churn est maximal les 12 premiers mois."
        )

        internet = st.selectbox(
            "3. Type d'abonnement Internet (Driver #3)",
            options=["Fiber optic", "DSL", "No"],
            index=0,
            help="La fibre optique enregistre le taux de résiliation le plus élevé (~42%)."
        )

        payment = st.selectbox(
            "4. Mode de Paiement (Driver #4)",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=0,
            help="Le chèque électronique est fortement associé aux départs (~45%)."
        )

    # Résultats & Inférence
    with col_results:
        st.subheader(" Diagnostic du Modèle")

        # Appel au backend
        backend = get_backend_service()
        result = backend.predict(
            contract=contract,
            tenure=tenure,
            internet_service=internet,
            payment_method=payment
        )

        # Badge de risque
        risk_class = {
            "HIGH": "badge-high",
            "MEDIUM": "badge-med",
            "LOW": "badge-low"
        }.get(result["risk_level"], "badge-med")

        st.markdown(f'<div class="{risk_class}">{result["badge_label"]}</div>', unsafe_allow_html=True)
        st.write("")
        st.write(result["message"])

        # Barre de progression
        st.progress(min(max(result["probability"], 0.0), 1.0))
        st.caption(f"Score calculé par le MLP : **{result['probability']:.3f}** (seuil de décision = 0.50)")

        # Recommandations
        st.markdown("####  Leviers d'action recommandés")
        for reco in result["recommendations"]:
            st.markdown(f'<div class="reco-box">{reco}</div>', unsafe_allow_html=True)
