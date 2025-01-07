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


plot_customer_gender_distribution('./customers.csv')
plot_customer_age_distribution('./customers.csv')
plot_product_sales('./products.csv', './purchases.csv')
plot_product_satisfaction('./products.csv', './purchases.csv')
