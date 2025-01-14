import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_customer_gender_distribution(customers_file_path):
    # Load customers dataset
    customers = pd.read_csv(customers_file_path)

    # Gender Distribution
    gender_distribution = customers['Gender'].value_counts()
    labels = ['Male', 'Female']
    colors = ['skyblue', 'orange']
    explode = [0.1, 0]  # 강조 효과 (Male 약간 분리)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.pie(
        gender_distribution,
        labels=labels,
        autopct='%1.1f%%',  # 각 항목에 % 표기
        startangle=90,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 12}
    )
    plt.title('Customer Gender Distribution', fontsize=16)

    # Save and show plot
    os.makedirs('./graphs', exist_ok=True)
    plt.savefig('./graphs/customer_gender_distribution_pie.png')
    plt.show()
    
def plot_customer_age_distribution(customers_file_path):
    # Load customers dataset
    customers = pd.read_csv(customers_file_path)

    # Age Distribution
    age_bins = [0, 20, 30, 40, 50, 60, 70, 100]
    age_labels = ['18-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    customers['AgeGroup'] = pd.cut(customers['Age'], bins=age_bins, labels=age_labels, right=False)
    age_distribution = customers['AgeGroup'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(8, 6))
    age_distribution.plot(kind='bar', color='lightgreen', rot=0)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)

    # Save and show plot
    os.makedirs('./graphs', exist_ok=True)
    plt.savefig('./graphs/customer_age_distribution.png')
    plt.show()

def plot_product_sales(products_file_path, purchases_file_path):
    # Load datasets
    products = pd.read_csv(products_file_path)
    purchases = pd.read_csv(purchases_file_path)

    # Merge datasets
    purchases_products = purchases.merge(products, on='ProductID')

    # Calculate total sales
    product_sales = purchases_products.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(10)

    # Plot
    plt.figure(figsize=(12, 8))
    product_sales.plot(kind='bar', color='lightblue', rot=45)
    plt.title('Top 10 Product Sales')
    plt.xlabel('Product Name')
    plt.ylabel('Total Sales')

    # Save and show plot
    os.makedirs('./graphs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./graphs/product_sales.png')
    plt.show()

def plot_product_satisfaction(products_file_path, purchases_file_path):
    # Load datasets
    products = pd.read_csv(products_file_path)
    purchases = pd.read_csv(purchases_file_path)

    # Merge datasets
    purchases_products = purchases.merge(products, on='ProductID')

    # Calculate average satisfaction
    product_satisfaction = purchases_products.groupby('ProductName')['Satisfaction'].mean().sort_values(ascending=False).head(10)

    # Plot
    plt.figure(figsize=(12, 8))
    product_satisfaction.plot(kind='bar', color='lightcoral', rot=45)
    plt.title('Top 10 Product Satisfaction')
    plt.xlabel('Product Name')
    plt.ylabel('Average Satisfaction')

    # Save and show plot
    os.makedirs('./graphs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./graphs/product_satisfaction.png')
    plt.show()

def plot_monthly_sales(purchases_file_path):
    # Load purchases dataset
    purchases = pd.read_csv(purchases_file_path)

    # Convert purchaseDate to datetime and extract month-year
    purchases['purchaseDate'] = pd.to_datetime(purchases['purchaseDate'])
    purchases['MonthYear'] = purchases['purchaseDate'].dt.to_period('M').astype(str)

    # Calculate total sales per month
    monthly_sales = purchases.groupby('MonthYear')['Quantity'].sum()

    # Sort by date order
    monthly_sales = monthly_sales.sort_index()

    # Plot
    plt.figure(figsize=(12, 8))
    monthly_sales.plot(kind='line', marker='o', color='teal')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month-Year')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save and show plot
    os.makedirs('./graphs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./graphs/monthly_sales_trend.png')
    plt.show()

def plot_churn_distribution(rfm_file_path):
    # Load RFM data
    rfm = pd.read_csv(rfm_file_path)

    # Churn distribution
    churn_distribution = rfm['Churn'].value_counts()
    labels = ['Not Churned', 'Churned']
    colors = ['lightblue', 'salmon']
    explode = [0, 0.1]  # Highlight Churned

    # Plot
    plt.figure(figsize=(8, 6))
    plt.pie(
        churn_distribution,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 12}
    )
    plt.title('Churn Distribution', fontsize=16)

    # Save and show plot
    os.makedirs('./graphs', exist_ok=True)
    plt.savefig('./graphs/churn_distribution.png')
    plt.show()

def plot_modeling_results(metrics):
    """
    metrics: A dictionary containing results for 'Logistic Regression' and 'PCA + Logistic Regression'.
    Example:
    metrics = {
        "Logistic Regression": {"accuracy": 0.95, "f1_score": 0.94},
        "PCA + Logistic Regression": {"accuracy": 0.85, "f1_score": 0.83}
    }
    """
    categories = list(metrics.keys())
    accuracy = [metrics[cat]['accuracy'] for cat in categories]
    f1_score = [metrics[cat]['f1_score'] for cat in categories]

    # Bar width
    bar_width = 0.35
    index = range(len(categories))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(index, accuracy, bar_width, label='Accuracy', color='lightblue')
    plt.bar(
        [i + bar_width for i in index],
        f1_score,
        bar_width,
        label='F1 Score',
        color='salmon'
    )

    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Modeling Results Comparison', fontsize=16)
    plt.xticks([i + bar_width / 2 for i in index], categories)
    plt.legend()

    # Save and show plot
    os.makedirs('./graphs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./graphs/modeling_results_comparison.png')
    plt.show()

def plot_customer_churn(file_path):
    """
    Plots a pie chart of customer retention vs churn based on the 'Churn' column in a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file containing a 'Churn' column with 0 for retained customers and 1 for churned customers.
    """
    # Load the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Count the number of retained and churned customers
    customer_status = data['Churn'].value_counts()
    retained = customer_status.get(0, 0)
    churned = customer_status.get(1, 0)

    # Calculate the percentages
    total_customers = retained + churned
    retained_percentage = (retained / total_customers) * 100
    churned_percentage = (churned / total_customers) * 100

    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(
        [retained_percentage, churned_percentage],
        labels=['Retained', 'Churned'],
        autopct='%1.1f%%',
        startangle=140
    )

    os.makedirs('./graphs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./graphs/Customer Retention vs Churn.png')
    plt.show()






plot_customer_gender_distribution('./customers.csv')
plot_customer_age_distribution('./customers.csv')
plot_product_sales('./products.csv', './purchases.csv')
plot_product_satisfaction('./products.csv', './purchases.csv')
plot_monthly_sales('./purchases.csv')
plot_customer_churn('./rfm.csv')
plot_churn_distribution('./rfm.csv')