import pandas as pd
import random
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Establish a connection to Snowflake
conn = snowflake.connector.connect(
    user='<username>',
    password='<password>',
    account='<account_url>',
    warehouse='<warehouse>',
    database='<database>',
    schema='<schema>'
)

# Create a cursor object
cur = conn.cursor()

# Create a temporary dataset
data = {
    'DATE': pd.date_range(start='1/1/2022', periods=10),
    'PRODUCT': [f'Product {i}' for i in range(1, 11)],
    'CLOSE PRICE': [random.randint(50, 150) for _ in range(10)],
    'VOLATILITY INDEX': [random.random() for _ in range(10)]
}
df = pd.DataFrame(data)

# Convert the DataFrame to a list of tuples
data_tuples = list(df.itertuples(index=False, name=None))

# Create a temporary table in Snowflake
cur.execute("CREATE TEMPORARY TABLE TEMP_TABLE (DATE DATE, PRODUCT STRING, CLOSE_PRICE NUMBER, VOLATILITY_INDEX FLOAT)")

# Insert the data into the temporary table
cur.executemany("INSERT INTO TEMP_TABLE VALUES (%s, %s, %s, %s)", data_tuples)

# Commit the transaction
conn.commit()

# Fetch the data from the Snowflake table
cur.execute("SELECT * FROM TEMP_TABLE")
rows = cur.fetchall()

# Convert the data into a DataFrame
df = pd.DataFrame(rows, columns=['DATE', 'PRODUCT', 'CLOSE_PRICE', 'VOLATILITY_INDEX'])

# Prepare the data for the model
X = df[['VOLATILITY_INDEX']]
y = df['CLOSE_PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Fetch the data from the Snowflake table
cur.execute("SELECT * FROM TEMP_TABLE")
rows = cur.fetchall()

# Convert the data into a DataFrame
df = pd.DataFrame(rows, columns=['DATE', 'PRODUCT', 'CLOSE_PRICE', 'VOLATILITY_INDEX'])

# Prepare the data for the model
X = df[['VOLATILITY_INDEX']]
y = df['CLOSE_PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)

# Generate a report
print(f"Our model has been trained to predict the 'CLOSE PRICE' of a product based on its 'VOLATILITY INDEX'.")
print(f"After testing the model on some data that it had never seen before, we found that its predictions were, on average, {mse:.2f} units away from the actual prices.")
print(f"This means that if our model predicts a 'CLOSE PRICE' of 100 units, you can expect the actual price to be between {100-mse:.2f} and {100+mse:.2f} units, most of the time.")

# Commit the transaction
conn.commit()
