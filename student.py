from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid GUI-related issues
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-model', methods=['POST'])
def run_model():
    # Load dataset with error handling for file upload
    file = request.files.get('file')
    if file:
        data = pd.read_csv(file)
    else:
        # Default file path (replace with your own or handle error)
        data = pd.read_csv("student_scores.csv")
    
    # Handling missing values
    data.dropna(subset=['Hours', 'Scores'], inplace=True)  # Remove rows with missing values
    
    # Scatter plot
    plt.scatter(data['Hours'], data['Scores'], color='blue', label='Actual Data')
    plt.title('Regression Line')
    plt.xlabel('Hours Studied')
    plt.ylabel('Marks Obtained')

    # Features and target variable
    X = data[['Hours']]
    y = data['Scores']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Plot regression line
    plt.plot(data['Hours'], model.predict(data[['Hours']]), color='red', label='Regression Line')
    plt.legend()

    # Define the path to save the plot image in the 'static' folder
    plot_path = os.path.join('static', 'regression_line.png')
    
    # Save the plot to the static folder
    plt.savefig(plot_path)
    plt.close()  # Close the plot after saving it

    # Add row numbers to the results dataframe
    results = pd.DataFrame({'Row Number': data.index + 1,  # Adding row numbers starting from 1
                            'Hours': data['Hours'], 
                            'Scores': data['Scores'],
                            'Predicted': np.concatenate([model.predict(X_train), y_pred])})
    
    # Convert results to HTML table
    results_html = results.to_html(classes='table table-bordered', index=False)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Return the rendered HTML page with results and the plot image
    return render_template('index.html', table=results_html, mse=mse, r2=r2, plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
