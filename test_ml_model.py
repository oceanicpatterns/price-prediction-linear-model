import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
import fakesnow
import snowflake.connector
from ml_model import evaluate, generate_report, predict, prepare_data, teardown_module, train_model

class TestSnowflakeConnection(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.df = pd.DataFrame([(1, 'Product 1', 50, 0.123456),
                            (2, 'Product 2', 51, 0.234567),
                            (3, 'Product 3', 52, 0.345678),
                            (4, 'Product 4', 53, 0.456789),
                            (5, 'Product 5', 54, 0.567890),
                            (6, 'Product 6', 55, 0.678901),
                            (7, 'Product 7', 56, 0.789012),
                            (8, 'Product 8', 57, 0.890123),
                            (9, 'Product 9', 58, 0.901234),
                            (10, 'Product 10', 59, 1.012345)],
                            columns=['DATE', 'PRODUCT', 'CLOSE_PRICE', 'VOLATILITY_INDEX'])
        
        with fakesnow.patch():
            cls.fake_conn = snowflake.connector.connect()

    def fetch_data(fake_conn):
        fake_cursor = fake_conn.cursor
        fake_conn.cursor().execute.return_value = None
        fake_conn.cursor().fetchall.return_value = [(1, 'Product 1', 50, 0.123456),
                                        (2, 'Product 2', 51, 0.234567),
                                        (3, 'Product 3', 52, 0.345678),
                                        (4, 'Product 4', 53, 0.456789),
                                        (5, 'Product 5', 54, 0.567890),
                                        (6, 'Product 6', 55, 0.678901),
                                        (7, 'Product 7', 56, 0.789012),
                                        (8, 'Product 8', 57, 0.890123),
                                        (9, 'Product 9', 58, 0.901234),
                                        (10, 'Product 10', 59, 1.012345)]
        df = pd.DataFrame(fake_cursor.fetchall(), columns=['DATE', 'PRODUCT', 'CLOSE_PRICE', 'VOLATILITY_INDEX'])
        return df

    def test_fetch_data(self):
        self.assertIsNotNone(self.df.to_numpy())

    def test_prepare_data(self):
        X, y = prepare_data(self.df)
        self.assertEqual(X.shape, (10, 1))
        self.assertEqual(y.shape, (10,))

    def test_train_model(self):
        X_train, y_train = prepare_data(self.df)
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, LinearRegression)

    def test_predict(self):
        X_train, y_train = prepare_data(self.df)
        model = train_model(X_train, y_train)
        X_test = pd.DataFrame([[0.5]], columns=['VOLATILITY_INDEX'])
        prediction = predict(model, X_test)
        self.assertIsInstance(prediction, pd.Series)

    def test_evaluate(self):
        X_train, y_train = prepare_data(self.df)
        model = train_model(X_train, y_train)
        X_test, y_test = prepare_data(self.df)
        prediction = predict(model, X_test)
        mse = evaluate(y_test, prediction)
        self.assertIsInstance(mse, float)

    def test_generate_report(self):
        X_train, y_train = prepare_data(self.df)
        model = train_model(X_train, y_train)
        X_test, y_test = prepare_data(self.df)
        prediction = predict(model, X_test)
        mse = evaluate(y_test, prediction)
        report = generate_report(mse)
        self.assertIsInstance(report, str)


    @classmethod
    def tearDownClass(cls):
       pass

if __name__ == '__main__':
    unittest.main()