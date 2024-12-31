import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from customer_analysis.repository.customer_analysis_repository import CustomerRepository


class CustomerRepositoryImpl(CustomerRepository):
    def __init__(self):
        # CSV 파일 로드
        self.products = pd.read_csv("./data/products.csv")
        self.purchases = pd.read_csv("./data/purchases.csv")
        self.customers = pd.read_csv("./data/customers.csv")

    def prepare_data(self):
        # 데이터 결합
        purchases_products = self.purchases.merge(self.products, on="ProductID")
        full_data = purchases_products.merge(self.customers, on="CustomerID")
        full_data["purchaseDate"] = pd.to_datetime(full_data["purchaseDate"])

        # RFM 데이터 생성 및 만족도 평균 추가
        rfm = full_data.groupby("CustomerID").agg({
            "purchaseDate": lambda x: (pd.Timestamp.now() - x.max()).days,  # Recency
            "purchaseID": "count",  # Frequency
            "TotalAmount": "sum",  # Monetary
            "Satisfaction": "mean"  # 평균 만족도
        }).rename(columns={
            "purchaseDate": "Recency",
            "purchaseID": "Frequency",
            "TotalAmount": "Monetary"
        })

        # 이탈 여부 생성
        rfm["Churn"] = (rfm["Recency"] > 90).astype(int)
        return rfm

    def split_data(self, rfm):
        # 학습 및 테스트 데이터 분리
        X = rfm[["Recency", "Frequency", "Monetary", "Satisfaction"]]
        y = rfm["Churn"]
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self, X_train, y_train):
        # Logistic Regression 모델 학습
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        # 모델 평가
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return accuracy, report

    def analyze_trends(self):
        # 구매 데이터 분석
        purchases_products = self.purchases.merge(self.products, on="ProductID")

        # 카테고리별 총 구매량
        category_sales = purchases_products.groupby("Category")["Quantity"].sum().to_dict()

        # 제품별 총 구매량과 평균 만족도
        product_stats = purchases_products.groupby("ProductName").agg({
            "Quantity": "sum",
            "Satisfaction": "mean"
        }).sort_values(by="Satisfaction", ascending=False).head(10)

        # 형식에 맞게 변환
        top_products = {
            product: f"{row['Quantity']} [{row['Satisfaction']:.2f}]"
            for product, row in product_stats.iterrows()
        }

        return {"category_sales": category_sales, "top_products": top_products}