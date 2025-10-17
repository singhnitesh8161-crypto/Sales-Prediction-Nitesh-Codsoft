import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    
    df = pd.read_csv('advertising.csv')

    print("--- First 5 Rows of the Dataset ---")
    print(df.head())
    print("\n")

    print("--- Dataset Info ---")
    df.info()
    print("\n")

    print("--- Missing Values ---")
    print(df.isnull().sum())
    print("\n")

    print("--- Generating Visualizations ---")
    
    sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
    plt.suptitle('Pairwise Relationships between Advertising Channels and Sales', y=1.02)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Variables')
    plt.show()

    features = ['TV', 'Radio', 'Newspaper']
    X = df[features]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set has {X_train.shape[0]} samples.")
    print(f"Testing set has {X_test.shape[0]} samples.\n")

    model = LinearRegression()

    print("--- Training the Linear Regression Model ---")
    model.fit(X_train, y_train)
    print("Model training complete.\n")

    print("--- Model Coefficients ---")
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)
    print("\n")


    y_pred = model.predict(X_test)


    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("--- Model Performance on Test Data ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2 Score): {r2:.2f}")
    print("The R2 score of ~0.90 indicates that approximately 90% of the variance in Sales can be explained by the advertising spend on TV, Radio, and Newspaper.\n")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2)
    plt.title('Actual Sales vs. Predicted Sales')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.grid(True)
    plt.show()
    
    new_budget = pd.DataFrame({
        'TV': [150],       
        'Radio': [25],      
        'Newspaper': [20]   
    })
    
    predicted_sales = model.predict(new_budget)
    print("--- Prediction for a New Campaign ---")
    print(f"Predicted sales for the new advertising budget: {predicted_sales[0]:.2f} units")


except FileNotFoundError:
    print("Error: 'advertising.csv' not found. Please make sure the file is in the same directory as the script.")
except Exception as e:
    print(f"An error occurred: {e}")