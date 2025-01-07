import os
import sys

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

from customer_analysis.service.customer_analysis_service_impl import CustomerServiceImpl

customerRouter = APIRouter()

async def injectCustomerService() -> CustomerServiceImpl:
    return CustomerServiceImpl()

class PCARequest(BaseModel):
    n_components: int
    
@customerRouter.post("/customer-analysis/churn")
async def predict_churn(customerService: CustomerServiceImpl = Depends(injectCustomerService)):
    churn_response = await customerService.predict_churn()
    return churn_response

@customerRouter.post("/customer-analysis/trends")
async def analyze_trends(customerService: CustomerServiceImpl = Depends(injectCustomerService)):
    trend_response = await customerService.analyze_trends()
    return trend_response

@customerRouter.post("/customer-analysis/pca")
async def predict_churn_with_pca(customerService: CustomerServiceImpl = Depends(injectCustomerService)):
    pca_churn_response = await customerService.predict_churn_with_pca()
    return pca_churn_response