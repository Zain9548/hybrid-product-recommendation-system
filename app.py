from flask import Flask, request, jsonify
import pickle
import pandas as pd

# ===== Import your project modules =====
from src.data_preprocessing import load_and_clean_data, prepare_cf_data, prepare_cb_data
from src.content_based_filtering import build_tfidf_model
from src.hybrid_recommender import hybrid_recommendations

# ===== Flask app =====
app = Flask(__name__)

# ===== Load Data =====
DATA_PATH = "data/7817_1.csv"

df = load_and_clean_data(DATA_PATH)
df_cf = prepare_cf_data(df)
df_cb = prepare_cb_data(df)

# ===== Load Models =====
with open("models/cf_model.pkl", "rb") as f:
    model_cf = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/similarity_matrix.pkl", "rb") as f:
    similarity_matrix = pickle.load(f)

print("✅ Models and data loaded successfully")

# ===== Home Route =====
@app.route("/")
def home():
    return jsonify({
        "message": "Hybrid Product Recommendation API is running 🚀",
        "endpoints": {
            "/recommend": "POST → get recommendations"
        }
    })

# ===== Recommendation Route =====
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Input JSON:
    {
        "username": "john_doe",
        "n": 10
    }
    """

    data = request.get_json()

    if not data or "username" not in data:
        return jsonify({"error": "username is required"}), 400

    username = data["username"]
    n = data.get("n", 10)

    # Check if user exists
    if username not in df_cf['reviews.username'].values:
        return jsonify({
            "error": "User not found",
            "message": "Cold-start user. Add fallback logic."
        }), 404

    # Get recommendations
    recommendations = hybrid_recommendations(
        username=username,
        model_cf=model_cf,
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


# ===== Run App =====
if __name__ == "__main__":
    app.run(debug=True)
