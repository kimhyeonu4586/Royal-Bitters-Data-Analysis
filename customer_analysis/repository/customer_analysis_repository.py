from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator


class CustomerRepository(ABC):
    @abstractmethod
    def prepare_data(self) -> pd.DataFrame:
        """
        데이터를 로드하고 RFM 데이터로 변환.
        """
        pass

    @abstractmethod
    def split_data(self, rfm: pd.DataFrame):
        """
        RFM 데이터를 학습 및 테스트 데이터로 분리.
        """
        pass

    @abstractmethod
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        """
        학습 데이터를 기반으로 모델을 학습.
        """
        pass

    @abstractmethod
    def evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
        """
        테스트 데이터를 사용하여 모델 평가.
        """
        pass

    @abstractmethod
    def analyze_trends(self) -> dict:
        """
        구매 동향을 분석하고 결과를 반환.
        """
        pass

    @abstractmethod
    def perform_pca_and_split(self, n_components: int):
        """
        PCA 처리와 데이터를 학습 및 테스트 세트로 분리.
        """
        pass