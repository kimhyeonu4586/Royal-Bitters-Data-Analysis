import os
import sys

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from customer_analysis.service.customer_analysis_service_impl import CustomerServiceImpl

customerRouter = APIRouter()

async def injectCustomerService() -> CustomerServiceImpl:
    return CustomerServiceImpl()

@customerRouter.post("/customer-analysis/churn")
async def predict_churn(customerService: CustomerServiceImpl = Depends(injectCustomerService)):
    churn_response = await customerService.predict_churn()
    return churn_response

@customerRouter.post("/customer-analysis/trends")
async def analyze_trends(customerService: CustomerServiceImpl = Depends(injectCustomerService)):
    trend_response = await customerService.analyze_trends()
    return trend_response
