import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

data = pd.read_csv(r"C:\Users\anuli\OneDrive\Desktop\Fossil Age Estimation\Datasets\fossil_data.csv")
X = data.drop(['Age'], axis=1)
y = data['Age']

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

ct = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('scaler', StandardScaler(), numerical_cols)
])

X = ct.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

huber = HuberRegressor(max_iter=500)  
huber.fit(X_train, y_train)

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    explained_variance = explained_variance_score(y, y_pred)

    print(f"Results on {dataset_name}:")
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared:", r2)
    print("Explained Variance:", explained_variance)
    print("-" * 20)

evaluate_model(huber, X_train, y_train, "Training Set")
evaluate_model(huber, X_val, y_val, "Validation Set")
evaluate_model(huber, X_test, y_test, "Test Set")
