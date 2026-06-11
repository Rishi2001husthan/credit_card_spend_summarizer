from fastapi import FastAPI
from src.api.v1.routes import query
app=FastAPI()


@app.get("/")
def read_root():
   return {"Message": "Hello World"}


@app.get("/health")
def health_check():
   return {
       "status":"ok"
   }


app.include_router(query.router,prefix="/api/v1")