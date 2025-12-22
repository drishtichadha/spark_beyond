from pydantic import BaseModel

class Problem(BaseModel):
    target: str
    schema: list[str]
    
