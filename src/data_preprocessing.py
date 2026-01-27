import pandas as pd

DROP_COLS = [
    'asins', 'colors', 'dateAdded', 'dateUpdated', 'dimension',
    'ean', 'keys', 'manufacturer', 'manufacturerNumber',
    'sizes', 'upc', 'weight',
    'reviews.date', 'reviews.sourceURLs',
    'reviews.userCity', 'reviews.userProvince'
]

def load_and_clean_data(csv_path):
    """
    Load CSV and drop unnecessary columns
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=DROP_COLS, errors='ignore')
    return df


def prepare_cf_data(df):
    """
    Prepare data for Collaborative Filtering
    """
    df_cf = df.drop(columns=[
        'brand','categories','name','prices',
        'reviews.doRecommend','reviews.numHelpful',
        'reviews.text','reviews.title'
    ], errors='ignore')

    df_cf = df_cf.dropna(subset=['reviews.rating'])
    df_cf = df_cf.drop_duplicates(['id', 'reviews.username'])

    user_counts = df_cf['reviews.username'].value_counts()
    item_counts = df_cf['id'].value_counts()

    df_cf = df_cf[
        df_cf['reviews.username'].isin(user_counts[user_counts >= 3].index) &
        df_cf['id'].isin(item_counts[item_counts >= 3].index)
    ]

    return df_cf


def prepare_cb_data(df):
    """
    Prepare data for Content-Based Filtering
    """
    df_cb = df.dropna(subset=['reviews.text']).copy()
    df_cb['text'] = df_cb['reviews.title'].fillna('') + " " + df_cb['reviews.text']
    return df_cb
