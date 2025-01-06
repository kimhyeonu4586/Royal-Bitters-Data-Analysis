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
    korean_domains = ["naver.com", "daum.net", "gmail.com", "hanmail.net", "kakao.com"]
    for _ in range(num_customers):
        local_part = fake.user_name()
        domain = random.choice(korean_domains)
        customer_id = f"{local_part}@{domain}"
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
        "Beer": {
            "Cass": 1400, 
            "Terra": 1980, 
            "Asahi": 3000, 
            "Corona": 2650, 
            "Tiger": 3000,
            "Sapporo": 3000, 
            "Kirin Ichibang": 3000, 
            "Hite": 2170, 
            "Kelly": 2170,
            "Guinness": 3000
        },
        "Wine": {
            "Ruffino Lumina Pinot Grigio": 39000,
            "Meiomi Pinot Noir": 90000,
            "Bontara Merlot": 65000,
            "Sonoma Estate Cabernet Sauvignon": 85000,
            "19 Crimes Red Blend": 67200,
            "David's Nadia Chenin Blanc 2017": 61800,
            "100 Hectares Grande Reserva Branco": 60000,
            "Gunderloch Riesling Royal Blue": 85000,
            "Fleur de Mer Rosé": 36000,
            "Louis Jadot Bourgogne Chardonnay": 35000
        },
        "Whiskey": {
            "Ballantine's 12 Year": 49000,
            "Ballantine's 21 Year": 177000,
            "Ballantine's 30 Year": 524900,
            "Glenfiddich 12 Year": 67500,
            "Glenfiddich 21 Year": 279000,
            "Glenfiddich 30 Year": 1250000,
            "Johnnie Walker Green Label": 69000,
            "Johnnie Walker Black Label": 39000,
            "Johnnie Walker Blue Label": 338000,
            "Johnnie Walker Red Label": 22000
        }
    }

    products = []
    for category, products_info in categories.items():
        for product_name, price in products_info.items():
            product_id = fake.uuid4()  # 고유 ID 생성
            products.append([product_id, product_name, category, price])

    return pd.DataFrame(products, columns=["ProductID", "ProductName", "Category", "Price (KRW)"])

# 3. 구매 데이터 생성
def generate_purchases(customers, products, num_purchases=2500):
    purchases = []
    for _ in range(num_purchases):
        purchase_id = fake.uuid4()
        customer_id = random.choice(customers["CustomerID"])
        product = products.sample(1).iloc[0]
        product_id = product["ProductID"]
        quantity = random.randint(1, 5)
        price = product["Price (KRW)"]
        total_amount = round(quantity * price, 2)
        purchase_date = fake.date_between(start_date="-1y", end_date="today")
        satisfaction = random.randint(1,10)
        purchases.append([purchase_id, customer_id, product_id, quantity, price, total_amount, purchase_date, satisfaction])
    return pd.DataFrame(purchases, columns=["purchaseID", "CustomerID", "ProductID", "Quantity", "Price (KRW)", "TotalAmount", "purchaseDate", "Satisfaction"])

# 데이터 생성
customers = generate_customers(400)
products = generate_products()
purchases = generate_purchases(customers, products, 2500)

# 데이터 저장 또는 확인
customers.to_csv("customers.csv", index=False)
products.to_csv("products.csv", index=False)
purchases.to_csv("purchases.csv", index=False)

print("데이터 생성 완료!")
