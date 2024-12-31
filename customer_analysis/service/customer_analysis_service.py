from abc import ABC, abstractmethod


class CustomerService(ABC):
    @abstractmethod
    async def predict_churn(self) -> dict:
        """
        고객 이탈 예측 수행.
        """
        pass

    @abstractmethod
    async def analyze_trends(self) -> dict:
        """
        구매 동향 분석 수행.
        """
        pass
