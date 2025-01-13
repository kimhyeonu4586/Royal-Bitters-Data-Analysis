import os
import sys
import json

from fastapi import APIRouter, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from customer_analysis.service.customer_analysis_service_impl import CustomerServiceImpl

customerRouter = APIRouter()

async def injectCustomerService() -> CustomerServiceImpl:
    return CustomerServiceImpl()

class PCARequest(BaseModel):
    n_components: int
    
@customerRouter.post("/customer-analysis/churn")
async def predict_churn(
    background_tasks: BackgroundTasks,
    customerService: CustomerServiceImpl = Depends(injectCustomerService)
):
    """
    Perform standard Logistic Regression for churn prediction.
    """
    churn_response = await customerService.predict_churn()

    # Save metrics to JSON
    os.makedirs('./graphs', exist_ok=True)
    with open('./graphs/logistic_regression_metrics.json', 'w') as f:
        json.dump(churn_response, f)

    # Schedule visualization task
    background_tasks.add_task(generate_visualizations)

    return JSONResponse(content=churn_response)

@customerRouter.post("/customer-analysis/trends")
async def analyze_trends(customerService: CustomerServiceImpl = Depends(injectCustomerService)):
    trend_response = await customerService.analyze_trends()
    return trend_response

@customerRouter.post("/customer-analysis/pca")
async def predict_churn_with_pca(
    background_tasks: BackgroundTasks,
    customerService: CustomerServiceImpl = Depends(injectCustomerService)
):
    """
    Perform PCA + Logistic Regression for churn prediction.
    """
    pca_churn_response = await customerService.predict_churn_with_pca()

    # Save metrics to JSON
    os.makedirs('./graphs', exist_ok=True)
    with open('./graphs/pca_logistic_regression_metrics.json', 'w') as f:
        json.dump(pca_churn_response, f)

    # Schedule visualization task
    background_tasks.add_task(generate_visualizations)

    return JSONResponse(content=pca_churn_response)

@customerRouter.get("/customer-analysis/visualize")
async def visualize_results():
    """
    Trigger visualization generation manually.
    """
    try:
        generate_visualizations()
        return {"message": "Visualizations generated successfully."}
    except Exception as e:
        return {"error": str(e)}

# Helper function for visualization
def generate_visualizations():
    from customer_analysis.visualization.visualization import visualize_results
    visualize_results()