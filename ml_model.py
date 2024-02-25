import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from snowflake_connection import get_snowflake_connection

def setup_module(module):
    # Create a temporary table in Snowflake
    conn = get_snowflake_connection()
    create_temp_table(conn)

    # Create a temporary dataset
    data = {
        'DATE': pd.date_range(start='1/1/2022', periods=10),
        'PRODUCT': [f'Product {i}' for i in range(1, 11)],
        'CLOSE_PRICE': [random.randint(50, 150) for _ in range(10)],
        'VOLATILITY_INDEX': [random.random() for _ in range(10)]
    }
    df = pd.DataFrame(data)

    insert_data(conn, df)

def teardown_module(module):
    # Drop the temporary table
    conn = get_snowflake_connection()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS TEMP_TABLE")

def create_temp_table(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE TEMPORARY TABLE TEMP_TABLE (DATE DATE, PRODUCT STRING, CLOSE_PRICE NUMBER, VOLATILITY_INDEX FLOAT)")


def insert_data(conn, df):
    with conn.cursor() as cur:
        for row in df.itertuples(index=False):
            cur.execute("INSERT INTO TEMP_TABLE VALUES (%s, %s, %s, %s)", row)


def fetch_data(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM TEMP_TABLE")
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=['DATE', 'PRODUCT', 'CLOSE_PRICE', 'VOLATILITY_INDEX'])


def prepare_data(df):
    required_columns = ['VOLATILITY_INDEX', 'CLOSE_PRICE']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[['VOLATILITY_INDEX']]
    y = df['CLOSE_PRICE']
    return X, y

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    predictions = model.predict(X_test)
    return pd.Series(predictions, name='CLOSE_PRICE')

def evaluate(y_test, predictions):
    return mean_squared_error(y_test, predictions)

def generate_report(mse):
    return f"Our model has been trained to predict the 'CLOSE PRICE' of a product based on its 'VOLATILITY INDEX'. After testing the model on some data that it had never seen before, we found that its predictions were, on average, {mse:.2f} units away from the actual prices. This means that if our model predicts a 'CLOSE PRICE' of 100 units, you can expect the actual price to be between {100-mse:.2f} and {100+mse:.2f} units, most of the time."
