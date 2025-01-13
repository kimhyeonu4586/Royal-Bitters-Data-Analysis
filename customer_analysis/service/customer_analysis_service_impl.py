from customer_analysis.repository.customer_analysis_repository_impl import CustomerRepositoryImpl
from customer_analysis.service.customer_analysis_service import CustomerService
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import json

class CustomerServiceImpl(CustomerService):
    def __init__(self):
        self.__repository = CustomerRepositoryImpl()

    async def predict_churn(self):
        # 데이터 준비
        rfm = self.__repository.prepare_data()

        # 학습 및 테스트 데이터 분리
        X_train, X_test, y_train, y_test = self.__repository.split_data(rfm)

        # 모델 학습
        model = self.__repository.train_model(X_train, y_train)

        # 모델 평가
        accuracy, report = self.__repository.evaluate_model(model, X_test, y_test)
        # f1_score = report["weighted avg"]["f1-score"]

        # with open('./graphs/logistic_regression_metrics.json', 'w') as f:
        #     json.dump({"accuracy": accuracy, "f1_score" : f1_score}, f)


        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    async def analyze_trends(self):
        # 구매 동향 분석
        trends = self.__repository.analyze_trends()
        return trends

    async def predict_churn_with_pca(self):
        """
        PCA 처리(n_components=2 고정)와 고객 이탈 예측 수행.
        """
        X_train, X_test, y_train, y_test = self.__repository.perform_pca_and_split(2)
        
        model = self.__repository.train_model_with_pca(X_train, y_train)
        
        accuracy, report = self.__repository.evaluate_model_with_pca(model, X_test, y_test)
        # f1_score = report["weighted avg"]["f1_score"]

        # with open('./graphs/pca_logistic_regression_metrics.json', 'w') as f:
        #     json.dump({"accuracy": accuracy, "f1_score": f1_score}, f)

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        
        return {
            "accuracy": accuracy, 
            "classification_report": report
        }
