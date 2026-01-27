from src.content_based_filtering import get_similar_products

def hybrid_recommendations(
    username,
    df_cf,
    df_cb,
    similarity_matrix,
    n=10
):
    final_scores = {}

    # Popularity-based (CF replacement)
    popular_items = (
        df_cf['id']
        .value_counts()
        .head(20)
        .index
        .tolist()
    )

    for item in popular_items:
        final_scores[item] = final_scores.get(item, 0) + 1.0

    # Content-Based part
    user_history = df_cf[df_cf['reviews.username'] == username]['id']

    if len(user_history) > 0:
        last_item = user_history.iloc[-1]
        cb_items = get_similar_products(
            last_item, df_cb, similarity_matrix, n=20
        )

        for item in cb_items:
            final_scores[item] = final_scores.get(item, 0) + 2.0

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n]
