from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


def train_cf_model(df_cf):
    """
    Train SVD collaborative filtering model
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df_cf[['reviews.username', 'id', 'reviews.rating']],
        reader
    )

    trainset, testset = train_test_split(
        data, test_size=0.2, random_state=42
    )

    model = SVD(random_state=42)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)

    return model, rmse


def get_cf_recommendations(username, model, df_cf, n=10):
    """
    Generate CF recommendations for a user
    """
    all_items = df_cf['id'].unique()
    rated_items = df_cf[df_cf['reviews.username'] == username]['id'].values

    items_to_predict = [i for i in all_items if i not in rated_items]

    preds = [
        (item, model.predict(username, item).est)
        for item in items_to_predict
    ]

    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]
