import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def build_tfidf_model(df_cb, max_features=5000):
    """
    Build TF-IDF model and similarity matrix
    """
    df_cb['clean_text'] = df_cb['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features
    )

    tfidf_matrix = vectorizer.fit_transform(df_cb['clean_text'])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return vectorizer, similarity_matrix


def get_similar_products(product_id, df_cb, similarity_matrix, n=10):
    """
    Get similar products using cosine similarity
    """
    if product_id not in df_cb['id'].values:
        return []

    idx = df_cb[df_cb['id'] == product_id].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in scores[1:n+1]]
    return df_cb.iloc[top_indices]['id'].values
