from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import os

from customer_analysis.controller.customer_analysis_controller import customerRouter

load_dotenv()

app = FastAPI()

app.include_router(customerRouter)

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv('HOST'), port=int(os.getenv('FASTAPI_PORT')))