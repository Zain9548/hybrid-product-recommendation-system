from flask import Flask, request, jsonify
import pickle

# ===== Import project modules =====
from src.data_preprocessing import (
    load_and_clean_data,
    prepare_cf_data,
    prepare_cb_data
)
from src.hybrid_recommender import hybrid_recommendations

# ===== Flask App =====
app = Flask(__name__)

# ===== Load Data =====
DATA_PATH = "data/7817_1.csv"

df = load_and_clean_data(DATA_PATH)
df_cf = prepare_cf_data(df)
df_cb = prepare_cb_data(df)

# ===== Load CBF Models (SAFE) =====
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/similarity_matrix.pkl", "rb") as f:
    similarity_matrix = pickle.load(f)

print("✅ Data and CBF models loaded successfully")

# ===== Home Route =====
@app.route("/")
def home():
    return jsonify({
        "message": "Hybrid Product Recommendation API is running 🚀",
        "model_type": "Popularity + Content-Based",
        "endpoints": {
            "/recommend": "POST → get recommendations"
        }
    })

# ===== Recommendation Route =====
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    if not data or "username" not in data:
        return jsonify({"error": "username is required"}), 400

    username = data["username"]
    n = int(data.get("n", 5))

    # ===== Hybrid Recommendations =====
    recommendations = hybrid_recommendations(
        username=username,
        df_cf=df_cf,
        df_cb=df_cb,
        similarity_matrix=similarity_matrix,
        n=n
    )

    response = [
        {"product_id": item, "score": round(score, 3)}
        for item, score in recommendations
    ]

    return jsonify({
        "username": username,
        "recommendations": response
    })


# ===== Run App (RENDER SAFE) =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
