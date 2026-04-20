import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv("PersonalityBFP_binarygroudtruh.csv")
print(f"Total: {len(df)} samples")
print(df.columns.tolist())

# ground truth
traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
gt_cols = ["GT_Openness", "GT_Conscientiousness", "GT_Extraversion", "GT_Agreeableness", "GT_Neuroticism"]

texts = df["Responses"].astype(str).tolist()
labels = df[gt_cols].values  # shape: (N, 5)

print(f"\nGround truth stats:")
print(df[gt_cols].describe())
from sklearn.model_selection import train_test_split

gt_cols = ["GT_Extraversion", "GT_Agreeableness", "GT_Conscientiousness", "GT_Neuroticism", "GT_Openness"]
traits =  ["Extraversion",    "Agreeableness",    "Conscientiousness",    "Neuroticism",    "Openness"]

texts = df["Responses"].astype(str).tolist()
labels = df[gt_cols].values

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

encoder = SentenceTransformer("intfloat/e5-large-v2"/"all-mpnet-base-v2")
print("\nEncoding train...")
train_emb = encoder.encode(X_train, batch_size=16, show_progress_bar=True)
print("Encoding test...")
test_emb = encoder.encode(X_test, batch_size=16, show_progress_bar=True)
print(f"Train embedding shape: {train_emb.shape}")
print(f"Test embedding shape:  {test_emb.shape}")
scaler = StandardScaler()
train_emb_scaled = scaler.fit_transform(train_emb)
test_emb_scaled = scaler.transform(test_emb)
print(f"\nScaler mean (first 5 dims): {scaler.mean_[:5]}")

regressor = MultiOutputRegressor(Ridge(alpha=1.0))
regressor.fit(train_emb_scaled, y_train)
print(f"\nTrained {len(regressor.estimators_)} Ridge regressors")
for i, (trait, est) in enumerate(zip(traits, regressor.estimators_)):
    print(f"  {trait:20s} coef norm: {np.linalg.norm(est.coef_):.4f}")
predictions = regressor.predict(test_emb_scaled)
predictions = np.clip(predictions, 1.0, 5.0) 
print(f"\nPrediction shape: {predictions.shape}")
print(f"Prediction sample (first 3 rows):")
for j in range(min(3, len(predictions))):
    pred_str = "  ".join([f"{traits[k]}: {predictions[j,k]:.2f}" for k in range(5)])
    gt_str   = "  ".join([f"{traits[k]}: {y_test[j,k]:.2f}"   for k in range(5)])
    print(f"  Row {j} PRED: {pred_str}")
    print(f"  Row {j} GT:   {gt_str}")
    print()

print("=== Evaluation Results (all-mpnet-base-v2) ===")
for i, trait in enumerate(traits):
    mae  = mean_absolute_error(y_test[:, i], predictions[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))
    print(f"  {trait:20s}  MAE: {mae:.4f}  RMSE: {rmse:.4f}")

overall_mae  = mean_absolute_error(y_test, predictions)
overall_rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"\n  {'Overall':20s}  MAE: {overall_mae:.4f}  RMSE: {overall_rmse:.4f}")

df_final = pd.DataFrame(y_test, columns=[f"GT_{t}" for t in traits])
for i, trait in enumerate(traits):
    df_final[f"Pred_{trait}"] = predictions[:, i]
df_final.to_csv("E5_prediction_result.csv",index=False)

from scipy.stats import pearsonr
def correlation_stats(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 3:
        return None, None, None, None, None
    r, p = pearsonr(y_true, y_pred)
    z = np.arctanh(r)
    se = 1 / np.sqrt(len(y_true) - 3)
    z_low = z - 1.96 * se
    z_high = z + 1.96 * se
    r_low = np.tanh(z_low)
    r_high = np.tanh(z_high)
    return r, p, se, r_low, r_high

from sklearn.metrics import mean_absolute_error

results_stats = []
for i, trait in enumerate(traits):
    y_true = df_final[f"GT_{trait}"].values.astype(float)
    y_pred = df_final[f"Pred_{trait}"].values.astype(float)
    r, p, se, low, high = correlation_stats(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    results_stats.append([trait, r, p, se, low, high, mae, rmse])

stats_df = pd.DataFrame(results_stats, columns=[
    "Trait", "Pearson_r", "p_value", "SE", "CI_lower", "CI_upper", "MAE", "RMSE"
])

print("\n=== Evaluation Stats ===")
print(stats_df.to_string(index=False))

stats_df.to_csv("E5_correlation_stats.csv", index=False)
print("\n✅ Saved to E5_correlation_stats.csv")

from sklearn.model_selection import train_test_split
from openai import OpenAI
import numpy as np

client = OpenAI()  
# gt_cols顺序和CSV一致
gt_cols = ["GT_Extraversion", "GT_Agreeableness", "GT_Conscientiousness", "GT_Neuroticism", "GT_Openness"]
traits =  ["Extraversion",    "Agreeableness",    "Conscientiousness",    "Neuroticism",    "Openness"]

texts = df["Responses"].astype(str).tolist()
labels = df[gt_cols].values

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

from openai import OpenAI
import numpy as np
# switch the model name
def get_openai_embeddings(texts, model=("text-embedding-3-small"/"text-embedding-3-large"), batch_size=10):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [t[:2048] for t in batch] 
        print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        batch_emb = [item.embedding for item in response.data]
        all_embeddings.extend(batch_emb)
    return np.array(all_embeddings)

print("Encoding train...")
train_emb = get_openai_embeddings(X_train)
print("Encoding test...")
test_emb = get_openai_embeddings(X_test)
print(f"Train embedding shape: {train_emb.shape}")
print(f"Test embedding shape:  {test_emb.shape}")


print("\nEncoding train with text-embedding-3-small...")
train_emb = get_openai_embeddings(X_train)
print("Encoding test...")
test_emb = get_openai_embeddings(X_test)
print(f"Train embedding shape: {train_emb.shape}")
print(f"Test embedding shape:  {test_emb.shape}")


scaler = StandardScaler()
train_emb_scaled = scaler.fit_transform(train_emb)
test_emb_scaled = scaler.transform(test_emb)
print(f"\nScaler mean (first 5 dims): {scaler.mean_[:5]}")


regressor = MultiOutputRegressor(Ridge(alpha=1.0))
regressor.fit(train_emb_scaled, y_train)
print(f"\nTrained {len(regressor.estimators_)} Ridge regressors")
for i, (trait, est) in enumerate(zip(traits, regressor.estimators_)):
    print(f"  {trait:20s} coef norm: {np.linalg.norm(est.coef_):.4f}")

predictions = regressor.predict(test_emb_scaled)
predictions = np.clip(predictions, 1.0, 5.0) 
print(f"\nPrediction shape: {predictions.shape}")
print(f"Prediction sample (first 3 rows):")
for j in range(min(3, len(predictions))):
    pred_str = "  ".join([f"{traits[k]}: {predictions[j,k]:.2f}" for k in range(5)])
    gt_str   = "  ".join([f"{traits[k]}: {y_test[j,k]:.2f}"   for k in range(5)])
    print(f"  Row {j} PRED: {pred_str}")
    print(f"  Row {j} GT:   {gt_str}")
    print()


print("=== Evaluation Results (all-mpnet-base-v2) ===")
for i, trait in enumerate(traits):
    mae  = mean_absolute_error(y_test[:, i], predictions[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))
    print(f"  {trait:20s}  MAE: {mae:.4f}  RMSE: {rmse:.4f}")

overall_mae  = mean_absolute_error(y_test, predictions)
overall_rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"\n  {'Overall':20s}  MAE: {overall_mae:.4f}  RMSE: {overall_rmse:.4f}")


# 构建df_final（包含GT和Pred）
df_final = pd.DataFrame(y_test, columns=[f"GT_{t}" for t in traits])
for i, trait in enumerate(traits):
    df_final[f"Pred_{trait}"] = predictions[:, i]
df_final.to_csv("OpenAI_small_prediction_result.csv", index=False)


from scipy.stats import pearsonr

def correlation_stats(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 3:
        return None, None, None, None, None
    r, p = pearsonr(y_true, y_pred)
    z = np.arctanh(r)
    se = 1 / np.sqrt(len(y_true) - 3)
    z_low = z - 1.96 * se
    z_high = z + 1.96 * se
    r_low = np.tanh(z_low)
    r_high = np.tanh(z_high)
    return r, p, se, r_low, r_high

from sklearn.metrics import mean_absolute_error

results_stats = []
for i, trait in enumerate(traits):
    y_true = df_final[f"GT_{trait}"].values.astype(float)
    y_pred = df_final[f"Pred_{trait}"].values.astype(float)
    r, p, se, low, high = correlation_stats(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    results_stats.append([trait, r, p, se, low, high, mae, rmse])

stats_df = pd.DataFrame(results_stats, columns=[
    "Trait", "Pearson_r", "p_value", "SE", "CI_lower", "CI_upper", "MAE", "RMSE"
])

print("\n=== Evaluation Stats ===")
print(stats_df.to_string(index=False))


stats_df.to_csv("OpenAI_small_correlation_stats.csv", index=False)
print("\n✅ Saved to OpenAI_small_correlation_stats.csv")
