import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Load the dataset
data = pd.read_csv('DSP II_data.csv')

# Set the title of the app
st.title('Predicting Electric Consumption in Malaysia')

# Add a sidebar for user input
st.sidebar.header('User Input') 
st.sidebar.subheader('Please enter the values of the features to predict electricity generation (TWh)')

# Define the user input function
def user_input():
    temp_mean = st.sidebar.slider('Temperature Mean', data['Temperature Mean'].min(), data['Temperature Mean'].max(), float(data['Temperature Mean'].mean()))
    gdp = st.sidebar.slider('GDP', data['GDP'].min(), data['GDP'].max(), float(data['GDP'].mean()))
    population = st.sidebar.slider('Population', int(data['Population'].min()), int(data['Population'].max()), int(data['Population'].mean()))
    features = pd.DataFrame({'Temperature Mean': [temp_mean], 'GDP': [gdp], 'Population': [population]})
    return features

# Get the user input and display it on the app
input_df = user_input()
st.subheader('User Input')
st.write(input_df)

# Split the data into features and target
features = data[['Temperature Mean', 'GDP', 'Population']]
target = data['Electricity generation (TWh)']

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

# Train the SVR model using the training data
param_grid_svr = {'svr__C': [0.1, 1, 10, 100], 
                  'svr__epsilon': [0.01, 0.1, 1, 10],
                  'svr__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))}
pipeline_svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='linear'))])
regressor_svr = GridSearchCV(pipeline_svr, param_grid_svr, cv=5, n_jobs=-1) 
regressor_svr.fit(features_train, target_train)

# Train the DT model using the training data
param_grid_dt = {'decisiontreeregressor__max_depth': [2, 4, 6, 8], 
                 'decisiontreeregressor__min_samples_split': [2, 4, 6], 
                 'decisiontreeregressor__min_samples_leaf': [1, 2, 3]}
pipeline_dt = Pipeline([('scaler', StandardScaler()), ('decisiontreeregressor', DecisionTreeRegressor())])
regressor_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=5, n_jobs=-1)
regressor_dt.fit(features_train, target_train)

# Train the KNN model
param_grid_knn = {'kneighborsregressor__n_neighbors': [3, 5, 7], 
                  'kneighborsregressor__weights': ['uniform', 'distance'], 
                  'kneighborsregressor__p': [1, 2]}
pipeline_knn = Pipeline([('scaler', StandardScaler()), ('kneighborsregressor', KNeighborsRegressor())])
regressor_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, n_jobs=-1)
regressor_knn.fit(features_train, target_train)

# Make predictions using your trained models and the user input
prediction_svr = regressor_svr.predict(input_df) 
prediction_dt = regressor_dt.predict(input_df)
prediction_knn = regressor_knn.predict(input_df)

# Display the predictions on the app
st.subheader('Predictions')
st.write(f'Support Vector Regression: {float(prediction_svr[0]):.2f} TWh')
st.write(f'Decision Tree Regression: {prediction_dt[0]:.2f} TWh')
st.write(f'K-Nearest Neighbors Regression: {prediction_knn[0]:.2f} TWh')

# Create a DataFrame with the actual and predicted values
results = pd.DataFrame({'Actual': target_test.values,
                        'Regressor Predicted': regressor_svr.predict(features_test),
                        'DT Predicted': regressor_dt.predict(features_test),
                        'KNN Predicted': regressor_knn.predict(features_test)})

# Create an interactive bar chart
fig = go.Figure()
fig.add_trace(go.Bar(x=results.index, y=results['Actual'], name='Actual'))
fig.add_trace(go.Bar(x=results.index, y=results['Regressor Predicted'], name='Regressor Predicted'))
fig.add_trace(go.Bar(x=results.index, y=results['DT Predicted'], name='DT Predicted'))
fig.add_trace(go.Bar(x=results.index, y=results['KNN Predicted'], name='KNN Predicted'))
fig.update_layout(barmode='group', xaxis_title='Index', yaxis_title='Value', title='Actual vs. Predicted')
st.plotly_chart(fig)

# Create an interactive scatter plot with a diagonal line
fig = go.Figure()
fig.add_trace(go.Scatter(x=results['Actual'], y=results['Actual'], name='Actual', mode='lines'))
fig.add_trace(go.Scatter(x=results['Actual'], y=results['Regressor Predicted'], name='Regressor Predicted', mode='markers'))
fig.add_trace(go.Scatter(x=results['Actual'], y=results['DT Predicted'], name='DT Predicted', mode='markers'))
fig.add_trace(go.Scatter(x=results['Actual'], y=results['KNN Predicted'], name='KNN Predicted', mode='markers'))
fig.update_layout(xaxis_title='Actual', yaxis_title='Predicted', title='Scatter Plot for Actual vs. Predicted')
st.plotly_chart(fig)

# Define a custom function to calculate the MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate the error metrics for each model
SVR_mse = mean_squared_error(target_test, regressor_svr.predict(features_test))
DT_mse = mean_squared_error(target_test, regressor_dt.predict(features_test))
KNN_mse = mean_squared_error(target_test, regressor_knn.predict(features_test))

SVR_rmse = mean_squared_error(target_test, regressor_svr.predict(features_test), squared=False)
DT_rmse = mean_squared_error(target_test, regressor_dt.predict(features_test), squared=False)
KNN_rmse = mean_squared_error(target_test, regressor_knn.predict(features_test), squared=False)

SVR_mape = mean_absolute_percentage_error(target_test, regressor_svr.predict(features_test))
DT_mape = mean_absolute_percentage_error(target_test, regressor_dt.predict(features_test))
KNN_mape = mean_absolute_percentage_error(target_test, regressor_knn.predict(features_test))

# Define the lists of error metrics
models = ['SVR', 'Decision Tree', 'KNN']
mse_values = [SVR_mse, DT_mse, KNN_mse]
rmse_values = [SVR_rmse, DT_rmse, KNN_rmse]
mape_values = [SVR_mape, DT_mape, KNN_mape]

x = np.arange(len(models)) 
width = 0.2

fig2, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mse_values, width, label='MSE')
rects2 = ax.bar(x + width/2, rmse_values, width, label='RMSE')
rects3 = ax.bar(x + 1.5*width, mape_values, width, label='MAPE')

ax.set_ylabel('Error')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_title('Model Comparison for Predicting Electric Consumption in Malaysia')

fig2.tight_layout()
st.pyplot(fig2)

# Add Feedback
st.subheader('Feedback')
st.write('Please rate your experience with this app and leave any comments or suggestions.')
rating = st.slider('Rating', 1, 5, 3)
comments = st.text_area('Comments', 'Enter your comments here')
submit = st.button('Submit')
if submit:
    st.write(f'Thank you for your feedback. You rated this app {rating} stars and wrote: {comments}')



