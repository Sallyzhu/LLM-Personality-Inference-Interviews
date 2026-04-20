import pandas as pd
from openai import OpenAI
import time
import re
import openai
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# === load test data ===
df = pd.read_csv("Your Data Path")
os.environ['OPENAI_API_KEY'] =""
# Retrieve the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')


# === your fine-tuned model id ===
MODEL_NAME = "gpt-5-mini-2025-08-07"

def predict_personality(text):
    prompt = f"""
You are a psychology expert trained in personality assessment using the Big Five model (Five-Factor Model, FFM).

Task:
Based on the following interview transcript, estimate the individual's Big Five personality traits.

Instructions:
- Infer each trait based on observable behavioral, emotional, and linguistic patterns in the text.
- Use the full range of the scale.
- Scores must be continuous values between 1.0 (very low) and 5.0 (very high).
- Do not round to integers unless strongly justified.

Important:
- Base your judgment only on evidence from the text.
- Do not assume traits without textual support.
- Be consistent across traits.

Return ONLY in this exact format (no extra text, no explanation):
Openness: X.X
Conscientiousness: X.X
Extraversion: X.X
Agreeableness: X.X
Neuroticism: X.X

{text}
"""

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        
    )

    return response.choices[0].message.content


# === run prediction ===
results = []

for i, row in df.iterrows():
    try:
        pred = predict_personality(row["Responses"])
        print(pred)
        results.append(pred)
        print(f"Done {i}")
        time.sleep(0.5)  # avoid rate limit
    except Exception as e:
        print(e)
        results.append(None)

#df["prediction_raw"] = results


def parse_prediction(text):
    if pd.isna(text):
        return [np.nan]*5
    
    try:
        # 用正则提取数值（更稳）
        openness = re.search(r'Openness:\s*([0-9.]+)', text)
        conscientiousness = re.search(r'Conscientiousness:\s*([0-9.]+)', text)
        extraversion = re.search(r'Extraversion:\s*([0-9.]+)', text)
        agreeableness = re.search(r'Agreeableness:\s*([0-9.]+)', text)
        neuroticism = re.search(r'Neuroticism:\s*([0-9.]+)', text)

        return [
            float(openness.group(1)) if openness else np.nan,
            float(extraversion.group(1)) if extraversion else np.nan,
            float(agreeableness.group(1)) if agreeableness else np.nan,
            float(conscientiousness.group(1)) if conscientiousness else np.nan,
            float(neuroticism.group(1)) if neuroticism else np.nan
        ]
    except:
        return [np.nan]*5
parsed = df["prediction_raw"].apply(parse_prediction)

df[[
    "Pred_Openness",
    "Pred_Extraversion",
    "Pred_Agreeableness",
    "Pred_Conscientiousness",
    "Pred_Neuroticism"
]] = pd.DataFrame(parsed.tolist(), index=df.index)
df = pd.concat([df, parsed_df], axis=1)
df.to_csv("", index=False)
import numpy as np
from scipy.stats import pearsonr

def correlation_stats(y_true, y_pred):
    # remove nan
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    n = len(y_true)

    # Pearson r + p-value
    r, p = pearsonr(y_true, y_pred)

    # Fisher Z transform
    z = np.arctanh(r)

    # standard error
    se = 1 / np.sqrt(n - 3)

    # 95% CI in z space
    z_low = z - 1.96 * se
    z_high = z + 1.96 * se

    # back transform
    r_low = np.tanh(z_low)
    r_high = np.tanh(z_high)

    return r, p, se, r_low, r_high
from sklearn.metrics import mean_absolute_error

results = []

for trait in gt_cols:
    y_true = df[gt_cols[trait]].values
    y_pred = df[f"Pred_{trait}"].values

    r, p, se, r_low, r_high = correlation_stats(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    results.append({
        "Trait": trait,
        "Pearson_r": r,
        "p_value": p,
        "SE": se,
        "CI_lower": r_low,
        "CI_upper": r_high,
        "MAE": mae
    })

results_df = pd.DataFrame(results)
print(results_df)
