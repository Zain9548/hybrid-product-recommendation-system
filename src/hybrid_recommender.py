from src.collaborative_filtering import get_cf_recommendations
from src.content_based_filtering import get_similar_products


def hybrid_recommendations(
    username,
    model_cf,
    df_cf,
    df_cb,
    similarity_matrix,
    n=10,
    cf_weight=0.6,
    cbf_weight=0.4
):
    """
    Hybrid recommendation using CF + CBF
    """
    final_scores = {}

    # Collaborative Filtering part
    try:
        cf_recs = get_cf_recommendations(
            username, model_cf, df_cf, n=20
        )
        for item, score in cf_recs:
            final_scores[item] = final_scores.get(item, 0) + cf_weight * score
    except:
        pass

    # Content-Based Filtering part
    user_history = df_cf[df_cf['reviews.username'] == username]['id']

    if len(user_history) > 0:
        last_item = user_history.iloc[-1]
        cb_items = get_similar_products(
            last_item, df_cb, similarity_matrix, n=20
        )

        for item in cb_items:
            final_scores[item] = final_scores.get(item, 0) + cbf_weight

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n]
