import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv(r"C:\Users\anuli\OneDrive\Desktop\Fossil Age Estimation\Datasets\fossil_data.csv")

print(data.columns)
print(data.head())

numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
print(correlation_matrix['Age'].sort_values(ascending=False))

most_correlated_feature = 'Longitude'  

categorical_cols = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

X = data_encoded.drop(['Age'], axis=1)
y = data_encoded['Age']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

isotonic = IsotonicRegression()
isotonic.fit(X_train[most_correlated_feature].values.reshape(-1, 1), y_train)

for name, dataset, y_true in [("Training Set", X_train, y_train), 
                               ("Validation Set", X_val, y_val), 
                               ("Test Set", X_test, y_test)]:
    y_pred = isotonic.predict(dataset[most_correlated_feature].values.reshape(-1, 1))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_variance = explained_variance_score(y_true, y_pred)

    print(f"Results on {name}:")
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared:", r2)
    print("Explained Variance:", explained_variance)
    print("-" * 20)
