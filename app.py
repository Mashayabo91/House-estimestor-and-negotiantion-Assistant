
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import gradio as gr
from groq import Groq
import os
import kagglehub

# ── 1. Download & load data ──────────────────────────────────────
path = kagglehub.dataset_download("iamsouravbanerjee/house-rent-prediction-dataset")
df = pd.read_csv(path + "/House_Rent_Dataset.csv")

# ── 2. Clean data ────────────────────────────────────────────────
lower = df["Rent"].quantile(0.01)
upper = df["Rent"].quantile(0.99)
df = df[(df["Rent"] >= lower) & (df["Rent"] <= upper)]

# ── 3. Feature engineering ───────────────────────────────────────
df["Price_Per_SqFt"] = df["Rent"] / df["Size"]
df["Room_Ratio"] = df["BHK"] / df["Bathroom"].replace(0, 1)
df = df.drop(columns=["Posted On", "Area Locality", "Floor", "Point of Contact"])

# ── 4. Encode categorical columns ────────────────────────────────
le_dict = {}
categorical_cols = ["Area Type", "City", "Furnishing Status", "Tenant Preferred"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Save class lists for dropdown menus
city_classes = list(le_dict["City"].classes_)
furnishing_classes = list(le_dict["Furnishing Status"].classes_)
area_type_classes = list(le_dict["Area Type"].classes_)
tenant_classes = list(le_dict["Tenant Preferred"].classes_)

# ── 5. Train XGBoost model ───────────────────────────────────────
df["Log_Rent"] = np.log1p(df["Rent"])
X = df.drop(columns=["Rent", "Log_Rent"])
y = df["Log_Rent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

# ── 6. Calculate threshold for classifier ────────────────────────
X_full = df.drop(columns=["Rent", "Log_Rent"])
y_full_pred_log = xgb_model.predict(X_full)
y_full_pred_actual = np.expm1(y_full_pred_log)

df["Predicted_Rent"] = y_full_pred_actual
df["Pct_Diff"] = ((df["Rent"] - df["Predicted_Rent"]) / df["Predicted_Rent"]) * 100
threshold = df["Pct_Diff"].std() * 0.5

# ── 7. Core functions ────────────────────────────────────────────
def predict_fair_rent(bhk, size, area_type, city,
                      furnishing_status, tenant_preferred, bathroom):
    try:
        area_type_enc = le_dict["Area Type"].transform([area_type])[0]
    except:
        area_type_enc = 0
    try:
        city_enc = le_dict["City"].transform([city])[0]
    except:
        city_enc = 0
    try:
        furnishing_enc = le_dict["Furnishing Status"].transform([furnishing_status])[0]
    except:
        furnishing_enc = 0
    try:
        tenant_enc = le_dict["Tenant Preferred"].transform([tenant_preferred])[0]
    except:
        tenant_enc = 0

    room_ratio = bhk / max(bathroom, 1)

    input_data = np.array([[
        bhk, size, area_type_enc, city_enc,
        furnishing_enc, tenant_enc, bathroom,
        0, room_ratio
    ]])

    log_pred = xgb_model.predict(input_data)[0]
    return round(np.expm1(log_pred), 2)


def classify_listing(actual_rent, predicted_rent):
    pct_diff = ((actual_rent - predicted_rent) / predicted_rent) * 100
    if pct_diff > threshold:
        label = "Overpriced"
    elif pct_diff < -threshold:
        label = "Underpriced"
    else:
        label = "Fairly Priced"
    return label, round(pct_diff, 1)


def generate_negotiation_points(bhk, size, city, furnishing_status,
                                 actual_rent, predicted_rent, label, pct_diff):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
    You are a professional real estate negotiation advisor.

    A renter is considering this property:
    - Location: {city}
    - Size: {size} sq ft, {bhk} BHK
    - Furnishing: {furnishing_status}
    - Listed Rent: ₹{actual_rent:,} per month
    - Fair Market Rent (ML predicted): ₹{predicted_rent:,} per month
    - Price Assessment: {label} by {abs(pct_diff):.1f}%

    Based on this analysis, provide:
    1. A clear one-sentence summary of the situation
    2. Three specific negotiation talking points the renter can use
    3. A suggested offer price with justification
    4. One thing the renter should watch out for

    Be specific, practical, and professional. Use Indian rupee context.
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are an expert real estate negotiation advisor specializing in Indian rental markets."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content


def analyze_rental(bhk, size, area_type, city,
                   furnishing_status, tenant_preferred,
                   bathroom, actual_rent):
    try:
        predicted_rent = predict_fair_rent(
            bhk, size, area_type, city,
            furnishing_status, tenant_preferred, bathroom
        )
        label, pct_diff = classify_listing(actual_rent, predicted_rent)
        advice = generate_negotiation_points(
            bhk, size, city, furnishing_status,
            actual_rent, predicted_rent, label, pct_diff
        )
        report = f"""
╔══════════════════════════════════════════╗
║      RENTAL PRICE ANALYSIS REPORT       ║
╚══════════════════════════════════════════╝

📍 Property : {bhk} BHK | {size} sq ft | {city}
🛋️  Furnishing: {furnishing_status}

💰 Listed Rent      : ₹{actual_rent:,}
🎯 Fair Market Rent : ₹{predicted_rent:,}
📊 Assessment       : {label} by {abs(pct_diff):.1f}%

══════════════════════════════════════════
🤖 AI NEGOTIATION ADVISOR (Powered by Groq)
══════════════════════════════════════════
{advice}
        """
        return report
    except Exception as e:
        return f"Error: {str(e)}"


# ── 8. Gradio Interface ──────────────────────────────────────────
demo = gr.Interface(
    fn=analyze_rental,
    inputs=[
        gr.Slider(1, 6, value=2, step=1, label="Number of BHK"),
        gr.Number(value=800, label="Size (sq ft)"),
        gr.Dropdown(choices=area_type_classes, value=area_type_classes[0], label="Area Type"),
        gr.Dropdown(choices=city_classes, label="City"),
        gr.Dropdown(choices=furnishing_classes, label="Furnishing Status"),
        gr.Dropdown(choices=tenant_classes, label="Tenant Preferred"),
        gr.Slider(1, 5, value=1, step=1, label="Number of Bathrooms"),
        gr.Number(value=20000, label="Listed Rent (₹)")
    ],
    outputs=gr.Textbox(label="Analysis Report", lines=25),
    title="🏠 Rental Price Estimator & Negotiation Assistant",
    description="Enter property details to get fair rent prediction and AI-powered negotiation advice.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
