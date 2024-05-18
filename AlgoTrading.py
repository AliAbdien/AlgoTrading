import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_explore_data(filepath):
    data = pd.read_excel(filepath)
    print("Data info:")
    print("Data description:")

    # Setting up a subplot grid with 2 rows and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig.suptitle('Data Exploration', fontsize=16)

    # Scatter plot to explore relationships
    sns.scatterplot(x='Volume', y='Close', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Volume vs Close Price Scatter Plot')

    # Box plot for outliers
    sns.boxplot(x=data['Close'], ax=axes[0, 1])
    axes[0, 1].set_title('Box Plot of Closing Prices')

    # Time series trend of 'Close' prices
    axes[1, 0].plot(data['Open time'], data['Close'], marker='o', linestyle='-')
    axes[1, 0].set_title('Time Series Trend of Closing Prices')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Close Price')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Histogram for 'Volume'
    sns.histplot(data['Volume'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Trading Volume')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return data


def visualize_data_dashboard(data):
    
    
    # Prepare category data for bar and pie plots
    if 'Category' not in data.columns:
        data['Category'] = pd.qcut(data['Volume'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

    # Set up the figure for subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 18), constrained_layout=True)
    fig.suptitle('Data Visualization Dashboard')

    # Line Plot
    axes[0, 0].plot(data['Open time'], data['Close'], label='Closing Prices')
    axes[0, 0].set_title('Time Series - Closing Prices')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Closing Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Area Plot
    axes[0, 1].fill_between(data['Open time'], data['Close'], color="skyblue", alpha=0.4)
    axes[0, 1].set_title('Area Plot - Closing Prices Over Time')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Closing Price')
    axes[0, 1].grid(True)

    # Bar Chart
    sns.barplot(x='Category', y='Close', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Bar Chart - Closing Prices by Volume Category')
    axes[1, 0].set_xlabel('Volume Category')
    axes[1, 0].set_ylabel('Closing Price')

    # Distribution Plot
    sns.histplot(data['Close'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Closing Prices')
    axes[1, 1].set_xlabel('Closing Price')
    axes[1, 1].set_ylabel('Frequency')

    # Pie Chart
    result = data.groupby('Category')['Close'].sum()
    result.plot(kind='pie', autopct='%1.1f%%', ax=axes[2, 0])
    axes[2, 0].set_title('Pie Chart - Market Share by Price Category')
    axes[2, 0].set_ylabel('')  # Hide the y-label

    # Bubble Plot
    bubble_size = data['Volume'] / data['Volume'].max() * 1000  # Normalize size
    scatter = axes[2, 1].scatter(data['Volume'], data['Close'], s=bubble_size, alpha=0.5, cmap='viridis', edgecolors='w', linewidths=0.5)
    axes[2, 1].set_title('Bubble Plot - Volume vs Close Price')
    axes[2, 1].set_xlabel('Volume')
    axes[2, 1].set_ylabel('Close Price')
    axes[2, 1].grid(True)
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5, num=3, color='cyan')
    axes[2, 1].legend(handles, ['Low Volume', 'Medium Volume', 'High Volume'], title="Volume Categories")

    plt.show()
    data.drop(columns=['Category'], inplace=True)
# Path to the data file

def clean_data(data):
    """
    Clean the dataset by filling missing values, removing duplicates, and handling outliers.
    Args:
        data (pd.DataFrame): The data to clean.
    Returns:
        pd.DataFrame: The cleaned data.
    """
    # Fill numerical missing values with the median
    for column in data.select_dtypes(include=['float64', 'int64']):
        median = data[column].median()
        data[column].fillna(median, inplace=True)
    
    # Fill categorical missing values with the mode
    for column in data.select_dtypes(include=['object']):
        mode = data[column].mode()[0]
        data[column].fillna(mode, inplace=True)
    
    print("Data after filling missing values:\n", data.head())

    # Remove duplicates
    initial_count = len(data)
    data.drop_duplicates(inplace=True)
    print(f"Removed {initial_count - len(data)} duplicates")

    # Handle outliers using IQR
    for column in data.select_dtypes(include=['float64', 'int64']):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
        data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    
    print("Data after handling outliers:\n", data.head())

    return data

def analyze_correlations(data, columns=None, strong_threshold=0.5, weak_threshold=0.5, visualize_pca=True):
    """
    Analyze and visualize correlations between numeric features and perform PCA if specified.
    Args:
        data (pd.DataFrame): The dataset containing the features.
        columns (list, optional): List of columns to include in the correlation matrix. Defaults to None.
        strong_threshold (float, optional): Threshold for considering correlations as strong. Defaults to 0.5.
        weak_threshold (float, optional): Threshold for considering correlations as weak. Defaults to 0.5.
        visualize_pca (bool, optional): Whether to perform and visualize PCA analysis. Defaults to True.
    """
    if columns is not None:
        data = data[columns]
    
    correlation_matrix = data.corr()

    # Identify strong and weak correlations with 'Close'
    close_correlations = correlation_matrix['Close']
    strong_correlations = close_correlations[(close_correlations.abs() >= strong_threshold) & (close_correlations.index != 'Close')]
    weak_correlations = close_correlations[(close_correlations.abs() < weak_threshold) & (close_correlations.index != 'Close')]

    print("Strong Correlations with 'Close':")
    print(strong_correlations)
    print("\nWeak Correlations with 'Close':")
    print(weak_correlations)

    # Plot the correlation matrix heatmap
    plt.figure(figsize=(30, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix')
    plt.show()

    # PCA Analysis for variable importance if required
    if visualize_pca and data.dropna().shape[0] > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data.dropna().select_dtypes(include=[np.number]))
        print("Explained variance by component: %s" % pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
        plt.title('PCA Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()

def select_strong_correlations(data, target_column='Close', threshold=0.5):
    """
    Select columns that have a strong correlation with the target column for further analysis.
    Args:
        data (pd.DataFrame): The dataset containing the features.
        target_column (str): The target column to check correlations against.
        threshold (float): Threshold for considering correlations as strong.
    Returns:
        pd.DataFrame: A DataFrame containing only the columns with strong correlations to the target column.
    """
    correlation_matrix = data.corr()
    strong_correlations = correlation_matrix[target_column][correlation_matrix[target_column].abs() >= threshold]
    strong_columns = strong_correlations.index.tolist()
    print("Columns with strong correlation to '{}': {}".format(target_column, strong_columns))
    return data[strong_columns]

def split_data(data, test_size=0.2, validation_size=0.1):
    """
    Split the data into training, validation, and testing sets.
    Args:
        data (pd.DataFrame): The dataset to be split.
        test_size (float): The proportion of the dataset to include in the test split.
        validation_size (float): The proportion of the training dataset to include in the validation split.
    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """
    # First, separate out the test set
    train_val, test = train_test_split(data, test_size=test_size, random_state=42)
    
    # Second, separate out the validation set
    val_size_adjusted = validation_size / (1 - test_size)  # Adjust validation size
    train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=42)
    
    return train, val, test

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Fits a model and evaluates its performance on both the training and test sets.
    Returns evaluation metrics including MSE, MAE, MedAE, and R^2.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'MSE': mse, 'MAE': mae, 'MedAE': medae, 'R2': r2}

def perform_clustering(data, features):
    """
    Perform clustering using multiple methods (K-means and Agglomerative Clustering with 4 clusters) and evaluate using Silhouette Score.
    
    Args:
        data (DataFrame): The dataset to cluster.
        features (list): List of columns to use for clustering.
    """
    # Extract the relevant features
    data_clustering = data[features].dropna()
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clustering)
    
    # Dictionary to store silhouette scores
    silhouette_scores = {}

    # K-means clustering with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels_kmeans = kmeans.fit_predict(data_scaled)
    score_kmeans = silhouette_score(data_scaled, labels_kmeans)
    silhouette_scores['K-means 4 clusters'] = score_kmeans
    print(f"Silhouette Score for K-means with 4 clusters: {score_kmeans}")
    
    # Plotting K-means clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(data_clustering.iloc[:, 0], data_clustering.iloc[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.5)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('K-means Clustering with 4 Clusters')
    plt.colorbar(label='Cluster Label')
    plt.show()

    # Agglomerative clustering with 4 clusters using 'ward' and 'average' linkage
    for linkage in ['ward', 'average']:
        agglom = AgglomerativeClustering(n_clusters=4, linkage=linkage)
        labels_agglom = agglom.fit_predict(data_scaled)
        score_agglom = silhouette_score(data_scaled, labels_agglom)
        silhouette_scores[f'Agglomerative {linkage} linkage'] = score_agglom
        print(f"Silhouette Score for Agglomerative Clustering with {linkage} linkage: {score_agglom}")
        
        # Plotting Agglomerative clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(data_clustering.iloc[:, 0], data_clustering.iloc[:, 1], c=labels_agglom, cmap='viridis', alpha=0.5)
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(f'Agglomerative Clustering with {linkage} Linkage')
        plt.colorbar(label='Cluster Label')
        plt.show()


def main():
    filepath = 'BTCUSDT_1d_data_Regression Processed.xlsx'
    data = load_and_explore_data(filepath)
    visualize_data_dashboard(data)

    data = clean_data(data)
    analyze_correlations(data)

    strong_data = select_strong_correlations(data, 'Close', 0.5)
    strong_data = strong_data.drop(columns=['Open time'])

    train, val, test = split_data(strong_data)
    target_column='Close'
    # Prepare features and target for training
    X_train = train.drop(target_column, axis=1)
    y_train = train[target_column]
    X_val = val.drop(target_column, axis=1)
    y_val = val[target_column]
    X_test = test.drop(target_column, axis=1)
    y_test = test[target_column]



    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        RandomForestRegressor(n_estimators=100),
        GradientBoostingRegressor(n_estimators=100),
        SVR(kernel='rbf'),
        KNeighborsRegressor(n_neighbors=5),
        DecisionTreeRegressor(),
        SVR(kernel='linear'),
        RandomForestRegressor(n_estimators=50)
    ]
    results = {}
    for i, model in enumerate(models):
        result = evaluate_model(model, X_train, y_train, X_test, y_test)
        results[f'Model_{i+1}'] = result
        print(f"Results for {model.__class__.__name__}:")
        for metric, value in result.items():
            print(f"  {metric}: {value:.4f}")
        print("----------------------------------------------------------")    
    # Find the best model based on R2 score
    best_model = max(results, key=lambda x: results[x]['R2'])
    print(f"\nBest Model: {best_model}, with R2: {results[best_model]['R2']:.4f}")
    
    # Visualization of model performance
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))
    fig.suptitle('Comparison of Model Performance Metrics', fontsize=16)
    
    for i, metric in enumerate(['MSE', 'MAE', 'MedAE', 'R2']):
        values = [results[model][metric] for model in results]
        axes[i].bar(results.keys(), values, color='skyblue')
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)
        for j, value in enumerate(values):
            axes[i].text(j, value, f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



    features = ['Close', 'Volume']  # Specify the features you want to use for clustering
    perform_clustering(data, features)

if __name__ == "__main__":
    main()
