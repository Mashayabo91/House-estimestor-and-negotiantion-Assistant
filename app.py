import os

import gradio as gr
import kagglehub
import numpy as np
import pandas as pd
import xgboost as xgb
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. DOWNLOAD AND LOAD DATASET
# ============================================================

print("Downloading rental dataset...")

dataset_path = kagglehub.dataset_download(
    "iamsouravbanerjee/house-rent-prediction-dataset"
)

csv_path = os.path.join(dataset_path, "House_Rent_Dataset.csv")

df = pd.read_csv(csv_path)

print(f"Dataset loaded successfully: {df.shape}")


# ============================================================
# 2. CLEAN DATA
# ============================================================

lower = df["Rent"].quantile(0.01)
upper = df["Rent"].quantile(0.99)

df = df[
    (df["Rent"] >= lower)
    & (df["Rent"] <= upper)
].copy()


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df["Price_Per_SqFt"] = df["Rent"] / df["Size"].replace(0, 1)

df["Room_Ratio"] = (
    df["BHK"] / df["Bathroom"].replace(0, 1)
)

columns_to_drop = [
    "Posted On",
    "Area Locality",
    "Floor",
    "Point of Contact",
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "Area Type",
    "City",
    "Furnishing Status",
    "Tenant Preferred",
]

label_encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    label_encoders[column] = encoder


# Save dropdown choices
area_type_classes = list(
    label_encoders["Area Type"].classes_
)

city_classes = list(
    label_encoders["City"].classes_
)

furnishing_classes = list(
    label_encoders["Furnishing Status"].classes_
)

tenant_classes = list(
    label_encoders["Tenant Preferred"].classes_
)


# ============================================================
# 5. TRAIN XGBOOST MODEL
# ============================================================

df["Log_Rent"] = np.log1p(df["Rent"])

X = df.drop(
    columns=["Rent", "Log_Rent"]
)

y = df["Log_Rent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


print("Training XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)

model.fit(
    X_train,
    y_train,
)

print("Model training completed.")


# ============================================================
# 6. CALCULATE PRICING THRESHOLD
# ============================================================

X_full = df.drop(
    columns=["Rent", "Log_Rent"]
)

predicted_log_rent = model.predict(X_full)

predicted_rent = np.expm1(
    predicted_log_rent
)

df["Predicted_Rent"] = predicted_rent

df["Pct_Diff"] = (
    (df["Rent"] - df["Predicted_Rent"])
    / df["Predicted_Rent"]
) * 100

threshold = df["Pct_Diff"].std() * 0.5


# ============================================================
# 7. PREDICT FAIR RENT
# ============================================================

def predict_fair_rent(
    bhk,
    size,
    area_type,
    city,
    furnishing_status,
    tenant_preferred,
    bathroom,
):

    area_type_encoded = label_encoders[
        "Area Type"
    ].transform([str(area_type)])[0]

    city_encoded = label_encoders[
        "City"
    ].transform([str(city)])[0]

    furnishing_encoded = label_encoders[
        "Furnishing Status"
    ].transform([str(furnishing_status)])[0]

    tenant_encoded = label_encoders[
        "Tenant Preferred"
    ].transform([str(tenant_preferred)])[0]

    room_ratio = (
        float(bhk)
        / max(float(bathroom), 1)
    )

    input_data = pd.DataFrame(
        [[
            float(bhk),
            float(size),
            area_type_encoded,
            city_encoded,
            furnishing_encoded,
            tenant_encoded,
            float(bathroom),
            0,
            room_ratio,
        ]],
        columns=X.columns,
    )

    predicted_log_rent = model.predict(
        input_data
    )[0]

    predicted_rent = np.expm1(
        predicted_log_rent
    )

    return round(
        float(predicted_rent),
        2,
    )


# ============================================================
# 8. CLASSIFY PROPERTY
# ============================================================

def classify_listing(
    actual_rent,
    predicted_rent,
):

    if predicted_rent <= 0:
        return "Unable to classify", 0.0

    percentage_difference = (
        (actual_rent - predicted_rent)
        / predicted_rent
    ) * 100

    if percentage_difference > threshold:
        label = "Overpriced"

    elif percentage_difference < -threshold:
        label = "Underpriced"

    else:
        label = "Fairly Priced"

    return (
        label,
        round(
            float(percentage_difference),
            1,
        ),
    )


# ============================================================
# 9. GROQ NEGOTIATION ASSISTANT
# ============================================================

def generate_negotiation_advice(
    bhk,
    size,
    city,
    furnishing_status,
    actual_rent,
    predicted_rent,
    label,
    percentage_difference,
):

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        return (
            "Groq is not configured. "
            "Please add GROQ_API_KEY to the "
            "Hugging Face Space Secrets."
        )

    client = Groq(
        api_key=api_key
    )

    prompt = f"""
You are a professional real estate
negotiation advisor specializing in
Indian rental properties.

Property information:

Location: {city}
Size: {size} sq ft
Bedrooms: {bhk} BHK
Furnishing: {furnishing_status}

Listed monthly rent:
₹{actual_rent:,.0f}

Machine-learning estimated fair rent:
₹{predicted_rent:,.0f}

Price assessment:
{label}

Difference:
{abs(percentage_difference):.1f}%

Provide:

1. A one-sentence assessment.
2. Three practical negotiation points.
3. A suggested offer price.
4. A short justification for the offer.
5. One important thing the renter should verify.

Keep the response practical,
professional, concise, and easy to understand.
Use Indian rupee context.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert "
                        "Indian real estate "
                        "negotiation advisor."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.7,
            max_tokens=500,
        )

        return response.choices[
            0
        ].message.content

    except Exception as error:

        return (
            "Groq negotiation assistant error: "
            f"{str(error)}"
        )


# ============================================================
# 10. MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_rental(
    bhk,
    size,
    area_type,
    city,
    furnishing_status,
    tenant_preferred,
    bathroom,
    actual_rent,
):

    try:

        predicted_rent = predict_fair_rent(
            bhk,
            size,
            area_type,
            city,
            furnishing_status,
            tenant_preferred,
            bathroom,
        )

        label, percentage_difference = (
            classify_listing(
                actual_rent,
                predicted_rent,
            )
        )

        negotiation_advice = (
            generate_negotiation_advice(
                bhk,
                size,
                city,
                furnishing_status,
                actual_rent,
                predicted_rent,
                label,
                percentage_difference,
            )
        )

        report = f"""
# 🏠 Rental Price Analysis

### Property

**Location:** {city}

**Property:** {bhk} BHK

**Size:** {size:,.0f} sq ft

**Furnishing:** {furnishing_status}

---

### 💰 Price Analysis

**Listed Rent:** ₹{actual_rent:,.0f} / month

**Estimated Fair Rent:** ₹{predicted_rent:,.0f} / month

**Assessment:** {label}

**Difference:** {abs(percentage_difference):.1f}%

---

### 🤖 AI Negotiation Advisor

{negotiation_advice}

---

*This estimate is generated using a machine-learning model and should be used as a decision-support tool, not as a guaranteed market valuation.*
"""

        return report

    except Exception as error:

        return (
            "### ❌ Analysis Error\n\n"
            f"{str(error)}"
        )


# ============================================================
# 11. GRADIO INTERFACE
# ============================================================

default_city = (
    city_classes[0]
    if city_classes
    else None
)

default_area_type = (
    area_type_classes[0]
    if area_type_classes
    else None
)

default_furnishing = (
    furnishing_classes[0]
    if furnishing_classes
    else None
)

default_tenant = (
    tenant_classes[0]
    if tenant_classes
    else None
)


demo = gr.Interface(
    fn=analyze_rental,

    inputs=[
        gr.Slider(
            minimum=1,
            maximum=6,
            value=2,
            step=1,
            label="Number of BHK",
        ),

        gr.Number(
            value=800,
            label="Size (sq ft)",
        ),

        gr.Dropdown(
            choices=area_type_classes,
            value=default_area_type,
            label="Area Type",
        ),

        gr.Dropdown(
            choices=city_classes,
            value=default_city,
            label="City",
        ),

        gr.Dropdown(
            choices=furnishing_classes,
            value=default_furnishing,
            label="Furnishing Status",
        ),

        gr.Dropdown(
            choices=tenant_classes,
            value=default_tenant,
            label="Tenant Preferred",
        ),

        gr.Slider(
            minimum=1,
            maximum=5,
            value=1,
            step=1,
            label="Number of Bathrooms",
        ),

        gr.Number(
            value=20000,
            label="Listed Rent (₹)",
        ),
    ],

    outputs=gr.Markdown(),

    title=(
        "🏠 Rental Price Estimator "
        "& Negotiation Assistant"
    ),

    description=(
        "Enter property details to estimate "
        "fair rental value and receive "
        "AI-powered negotiation advice."
    ),

    theme=gr.themes.Soft(),
)


# ============================================================
# 12. START APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
    )