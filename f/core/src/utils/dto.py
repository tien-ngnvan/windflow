from pydantic import BaseModel
from typing import Dict, Optional



class Document(BaseModel):
    page_content: str
    metadata: Optional[Dict[str, any]] = {}
    id: Optional[str] = None
