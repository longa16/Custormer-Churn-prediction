# Telco Customer Churn Prediction & Diagnostic Web App

An end-to-end Machine Learning project to detect, predict, and prevent customer churn for a telecom company using an exploratory data-driven approach and a **Multi-Layer Perceptron** neural network.

---

##  Key Highlights

- **Data-Driven Feature Selection**: In-depth exploratory data analysi identified the **top 4 most discriminant drivers** of churn:
  1. `Contract`: Month-to-month contracts vs. long-term commitments.
  2. `tenure`: Customer seniority in months, highest risk in the first 12 months.
  3. `InternetService`: Fiber optic vs. DSL or no internet.
  4. `PaymentMethod`: Electronic check friction vs. automated recurring billing.
- **Deep Learning Model**: Built with TensorFlow/Keras, featuring Dense and Dropout layers trained with balanced class weights to prioritize churner detection.
- **Performance**:
  - **ROC-AUC Score**: ~`0.84` on unseen test data.
  - **Recall on Churners**: ~`78%` intercepts nearly 4 out of 5 departing customers.
- **Clean Architecture**: Strict separation of concerns between Frontend Streamlit and Backend (inference service & business rules).

---

## Project Structure

```text
Telco_clientsChurn/
├── app/
│   ├── __init__.py
│   ├── backend.py            # Inference engine, artifact loader & recommendation rules
│   ├── frontend.py           # Streamlit UI components, CSS styling & form controls
│   ├── churn_mlp_4f.keras    # Serialized Keras neural network model
│   └── preprocessor_4f.joblib# Scikit-learn ColumnTransformer
│
├── main.py                   # Single entry point for the Streamlit web application
├── train_mlp_4features.py    # Training & serialization pipeline
├── telco.ipynb               # Step-by-step exploratory data analysis notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── pyproject.toml            # Project dependencies and configuration
```

---

## Quickstart

### 1. Prerequisites
- Python `>= 3.13`
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### 2. Install Dependencies

```bash
uv sync
```

*(or using standard pip: `pip install -r requirements.txt` if exported)*

### 3. Launch the Application

Run the unique application entry point:

```bash
streamlit run main.py
```

*Alternatively, you can run directly:*
```bash
uv run python main.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## Optional Re-train the Model

To re-run the training pipeline and save fresh artifacts into `app/`:

```bash
uv run python train_mlp_4features.py
```

---

## Business Recommendations Engine

The web application doesn't just output a risk percentage; it provides actionable retention suggestions based on customer attributes:
- **Month-to-month customer**: Propose annual plan discount incentives (-15%).
- **First-year customer (`tenure <= 12`)**: Trigger an onboarding check-in call.
- **Fiber optic subscriber**: Schedule a proactive connection quality check.
- **Electronic check payer**: Offer a billing credit incentive to switch to automated bank debit.
