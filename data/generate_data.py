import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Faker 초기화
fake = Faker('ko_KR')

# 1. 고객 데이터 생성
def generate_customers(num_customers=400):
    customers = []
    for _ in range(num_customers):
        customer_id = fake.uuid4()
        name = fake.name()
        age = random.randint(18, 70)
        gender = random.choice(["M", "F"])
        location = fake.city()
        signup_date = fake.date_between(start_date="-2y", end_date="today")
        customers.append([customer_id, name, age, gender, location, signup_date])
    return pd.DataFrame(customers, columns=["CustomerID", "Name", "Age", "Gender", "Location", "SignupDate"])

# 2. 상품 데이터 생성
def generate_products():
    categories = {
        "Beer": [
            "Cass", "Terra", "Asahi", "Corona", "Tiger", "Sapporo", "Kirin Ichibang", "Hite", "Kelly", "Guinness", "Kozel"
        ],
        "Wine": [
            "Sangiovese", "Pinot Noir", "Syrah", "Merlot", "Cabernet Sauvignon",
            "Chenin Blanc", "Pinot Gris", "Riesling", "Sauvignon Blanc", "Chardonnay"
        ],
        "Whiskey": [
            "Ballantine's 12", "Ballantine's 21", "Ballantine's 30",
            "Glenfiddich 12", "Glenfiddich 21", "Glenfiddich 30",
            "Johnnie Walker Green", "Johnnie Walker Black", "Johnnie Walker Blue", "Johnnie Walker Red"
        ]
    }

    products = []
    for category, product_names in categories.items():
        for product_name in product_names:
            product_id = fake.uuid4()  # 고유 ID 생성
            price = round(random.uniform(5, 100), 2)  # 5에서 100 사이의 랜덤 가격 생성
            products.append([product_id, product_name, category, price])

    return pd.DataFrame(products, columns=["ProductID", "ProductName", "Category", "Price"])

# 3. 구매 데이터 생성
def generate_purchases(customers, products, num_purchases=2500):
    purchases = []
    for _ in range(num_purchases):
        purchase_id = fake.uuid4()
        customer_id = random.choice(customers["CustomerID"])
        product = products.sample(1).iloc[0]
        product_id = product["ProductID"]
        quantity = random.randint(1, 5)
        price = product["Price"]
        total_amount = round(quantity * price, 2)
        purchase_date = fake.date_between(start_date="-1y", end_date="today")
        satisfaction = random.randint(1,10)
        purchases.append([purchase_id, customer_id, product_id, quantity, price, total_amount, purchase_date, satisfaction])
    return pd.DataFrame(purchases, columns=["purchaseID", "CustomerID", "ProductID", "Quantity", "Price", "TotalAmount", "purchaseDate", "Satisfaction"])

# 데이터 생성
customers = generate_customers(200)
products = generate_products()
purchases = generate_purchases(customers, products, 1000)

# 데이터 저장 또는 확인
customers.to_csv("customers.csv", index=False)
products.to_csv("products.csv", index=False)
purchases.to_csv("purchases.csv", index=False)

print("데이터 생성 완료!")
