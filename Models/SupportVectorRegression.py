import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(data_path):
    return pd.read_csv(data_path)

def preprocess_data(df):
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

def train_evaluate_validate_svr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svr_model = SVR(kernel='rbf', C=1, gamma='scale')
    svr_model.fit(X_train, y_train)

    y_pred = svr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)

    print("Model Performance:")
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared:", r2)
    print("Explained Variance:", explained_variance)

    return svr_model

data_path = 'C:\\Users\\anuli\\OneDrive\\Desktop\\Fossil Age Estimation\\Datasets\\fossil_data.csv'
fossil_data = load_data(data_path)
X, y = preprocess_data(fossil_data)
best_svr_model = train_evaluate_validate_svr(X, y)
