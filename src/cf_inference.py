def get_cf_recommendations(username, model, df_cf, n=10):
    all_items = df_cf['id'].unique()
    rated_items = df_cf[df_cf['reviews.username'] == username]['id'].values

    items_to_predict = [item for item in all_items if item not in rated_items]

    predictions = [
        (item, model.predict(username, item).est)
        for item in items_to_predict
    ]

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]
