from customer_analysis.repository.customer_analysis_repository_impl import CustomerRepositoryImpl
from customer_analysis.service.customer_analysis_service import CustomerService


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

        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    async def analyze_trends(self):
        # 구매 동향 분석
        trends = self.__repository.analyze_trends()
        return trends