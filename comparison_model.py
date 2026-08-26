import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble  import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data_file = "employee_salary_regression.csv"
data_path = os.path.join(os.path.dirname(__file__), data_file)
data = pd.read_csv(data_file)


data['remote_work'] = data['remote_work'].fillna(data['remote_work'].mean())
data['age'] = data['age'].ffill()
encoder = LabelEncoder()
data['education_level'] = encoder.fit_transform(data['education_level'])
scaler = MinMaxScaler()
sacled_data =  scaler.fit_transform(data[['annual_salary_usd']])
sacled_data.min(), sacled_data.max() 
standard_scaler = StandardScaler()
data['annual_salary_usd'] = standard_scaler.fit_transform(data[['annual_salary_usd']])

x = data[['age','years_experience','education_level','performance_score','num_skills','remote_work']]
y = data['annual_salary_usd']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)
linear_regression_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()
model1 =linear_regression_model.fit(x_train,y_train)
model2 = decision_tree_model.fit(x_train,y_train)
model3 = random_forest_model.fit(x_train,y_train)
y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)
mean_sqaured_error_model1 = mean_squared_error(y_test,y_pred1)
mean_sqaured_error_model2 = mean_squared_error(y_test,y_pred2)
mean_sqaured_error_model3 = mean_squared_error(y_test,y_pred3)
rms_model = np.sqrt(mean_squared_error(y_test,y_pred1))
rms_model2 = np.sqrt(mean_squared_error(y_test,y_pred2))
rms_model3 = np.sqrt(mean_squared_error(y_test,y_pred3))

r2_score1 = r2_score(y_test,y_pred1)
r2_score2 = r2_score(y_test,y_pred2)    
r2_score3 = r2_score(y_test,y_pred3)

model_results_dir = os.path.join(os.path.dirname(__file__), 'model_results')
if not os.path.exists(model_results_dir):
    os.makedirs(model_results_dir)
joblib.dump(model1, os.path.join(model_results_dir, 'linear_regression_model.pkl'))
joblib.dump(model2, os.path.join(model_results_dir, 'decision_tree_model.pkl'))
joblib.dump(model3, os.path.join(model_results_dir, 'random_forest_model.pkl')) 
joblib.dump(encoder, os.path.join(model_results_dir, 'label_encoder.pkl'))
joblib.dump(standard_scaler, os.path.join(model_results_dir, 'standard_scaler.pkl'))
joblib.dump(scaler, os.path.join(model_results_dir, 'min_max_scaler.pkl'))
print("Model Saved Successfully")


