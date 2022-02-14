import joblib
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from utils import data_transform, ColumnSelector

date_transformer = FunctionTransformer(data_transform)
filename = 'model/pipeline.pkl'
X_test = pd.read_csv('data_test.csv')
X_test['target'] = joblib.load(filename).predict(X_test)
with open('answers_test.csv', 'w') as file:
    X_test[['id', 'buy_time', 'vas_id', 'target']].to_csv(file, index=False)
