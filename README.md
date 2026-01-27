# 🛒 Personalized Product Recommendation Engine (Hybrid)

A **production-ready Hybrid Recommendation System** that provides personalized product recommendations by combining **Collaborative Filtering (CF)** and **Content-Based Filtering (CBF)**.  
The system is deployed as a **Flask REST API** and can be consumed by any frontend (Streamlit / React / Mobile App).

---

## 🚀 Project Overview

Modern e-commerce platforms rely heavily on recommendation systems to improve user experience and increase sales.  
This project implements a **Hybrid Recommendation Engine** that:

- Learns from **user behavior** (ratings, interactions)
- Understands **product content** (text reviews)
- Combines both approaches to overcome **cold-start problems**

---

## 🧠 Recommendation Techniques Used

### 1️⃣ Collaborative Filtering (CF)
- Technique: **Matrix Factorization (SVD)**
- Library: `scikit-surprise`
- Idea:  
  > Users with similar preferences tend to like similar products.

### 2️⃣ Content-Based Filtering (CBF)
- Technique: **TF-IDF + Cosine Similarity**
- Library: `scikit-learn`
- Idea:  
  > Recommend products similar to what the user liked before.

### 3️⃣ Hybrid Recommendation
- Final score =  0.6 × CF score + 0.4 × CBF score

  ---
  





**RUN the Run the Backend (Flask API)**
python app.py



Server will start 
cpp
http://127.0.0.1:5000/




Deployement For API



