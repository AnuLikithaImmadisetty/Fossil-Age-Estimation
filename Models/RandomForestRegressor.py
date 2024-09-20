import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df = pd.read_csv(r'C:\Users\anuli\OneDrive\Desktop\Fossil Age Estimation\Datasets\fossil_data.csv')
    return pd.read_csv(data_path)

def preprocess_data(df):
    """Prepares the data for model training."""
    X = df.drop(['Age'], axis=1)
    y = df['Age']

    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    ct = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('scaler', StandardScaler(), numerical_cols)
    ], remainder='passthrough')

    X = ct.fit_transform(X)

    return X, y

def train_evaluate_validate(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    for name, X_set, y_true in zip(["Training", "Validation", "Testing"], [X_train, X_val, X_test], [y_train, y_val, y_test]):
        y_pred = rf_model.predict(X_set)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_variance = explained_variance_score(y_true, y_pred)  # Added


        print(f"Results on {X_set.shape[0]} samples ({name}):")
        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)
        print("R-squared:", r2)
        print("Explained Variance:", explained_variance)
        print("-"*20)

    return rf_model

data_path = 'C:\\Users\\anuli\\OneDrive\\Desktop\\Fossil Age Estimation\\Datasets\\fossil_data.csv'
fossil_data = load_data(data_path)

X, y = preprocess_data(fossil_data)
model = train_evaluate_validate(X, y)