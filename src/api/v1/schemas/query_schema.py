
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
   query : str = Field(..., example="What is the minimum CIBIL score for personal loans?")




class QueryResponse(BaseModel):
   query : str
   answer : str
   policy_citations : str
   page_no : str
   document_name : str




class AIResponse(BaseModel):
   query: str = Field(description="The Given query by user must be present here")
   answer: str = Field(description="The generated response")
   policy_citations: str = Field(description="Give the Policy Citation (for document queries)")
   page_no: str = Field(description="The page number in the metadata")
   document_name: str = Field(description="Name of the document used")
   sql_query_executed: Optional[str] = Field(default = None, description = "The SQL query executed (for product/database queries)")

