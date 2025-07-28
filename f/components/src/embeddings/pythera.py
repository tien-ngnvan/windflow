from typing import Dict, Any
from pydantic import BaseModel

try:
    from trism import TritonModel

except ImportError as exc:
    raise ImportError(
        "Could not import sentence_transformers python package. "
        "Please install it with `pip install trism`."
    )

from ....core.src.embeddings.base import EmbeddingBase



class PytheraAPI(BaseModel, EmbeddingBase):
    """Sentence transformers embedding models.

    .. code-block:: python
            model_name = {
                'query': 'query_ensemble', 'passage': 'passage_ensemble'
            }
            url = {
                'query':'http://model.pythera.ai/retrieve/v0/query', 
                'passage':'http://model.pythera.ai/retrieve/v0/passage'
            }
            hf = PytheraAPI(
                model_name= model_name,
                url= url,
                version= 0,
                grpc= True,
            )

    """
    model_name: Dict[str, Any]
    url: Dict[str, Any]
    version: int
    grpc: bool = True

    def __init__(self, **kwargs):
        super.__init__(**kwargs)

        for k, v in self.model_name.items():
            if k not in ['query', 'passage']:
                raise KeyError("The key must be `query` or `passage`")
        for k, v in self.url.items():
            if k not in ['query', 'passage']:
                raise KeyError("The key must be `query` or `passage`")
            
            self.query_model = TritonModel(
                model= self.model_name['query'],
                url= self.url['query'],
                version= self.version,
                grpc=self.grpc
            )

            self.passage_model = TritonModel(
                model= self.model_name['passage'],
                url= self.url['passage'],
                version= self.version,
                grpc=self.grpc
            )

    def embed_documents(self, documents):
        """Compute doc embeddings using a Pythera model.

        Args:
            documents: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeded = self.passage_model.run(
            data = documents
        )
        
        return embeded.tolist()
    
    def embed_query(self, text):
        """Compute query embeddings using a Pythera model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeded = self.query_model.run(
            data = text
        )
        return embeded.tolist()