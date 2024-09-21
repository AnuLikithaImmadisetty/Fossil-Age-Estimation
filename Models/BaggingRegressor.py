import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

data = pd.read_csv(r"C:\\Users\\anuli\\OneDrive\\Desktop\\Fossil Age Estimation\\Datasets\\fossil_data.csv")

X = data.drop(['Age'], axis=1)
y = data['Age']

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

ct = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('scaler', StandardScaler(), numerical_cols)
])

X = ct.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # First split
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Second split for validation and test

bagging = BaggingRegressor(n_estimators=100)  
bagging.fit(X_train, y_train)

def print_results(set_name, y_true, y_pred):
    print(f"Results on {set_name} Set:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred)}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred)}")
    print(f"R-squared: {r2_score(y_true, y_pred)}")
    print(f"Explained Variance: {explained_variance_score(y_true, y_pred)}")
    print("--------------------")

y_train_pred = bagging.predict(X_train)
print_results("Training", y_train, y_train_pred)

y_val_pred = bagging.predict(X_val)
print_results("Validation", y_val, y_val_pred)

y_test_pred = bagging.predict(X_test)
print_results("Test", y_test, y_test_pred)
