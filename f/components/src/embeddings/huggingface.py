from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    import sentence_transformers

except ImportError as exc:
    raise ImportError(
        "Could not import sentence_transformers python package. "
        "Please install it with `pip install sentence-transformers`."
    )

from ....core.src.embeddings.base import EmbeddingBase



class STEmbeding(BaseModel, EmbeddingBase):
    """Sentence transformers embedding models.

    .. code-block:: python
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = STEmbeding(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    cache_folder: Optional[str] = None

    def __init__(self, **kwargs):
        super.__init__(**kwargs)

        self.model = sentence_transformers.SentenceTransformer(
            self.model_name,
            cache_folder=self.cache_folder,
            **self.model_kwargs,
        )

    def embed_documents(self, documents):
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            documents: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeded = self.model.encode(
            documents, **self.encode_kwargs 
        )
        
        return embeded.tolist()
    
    def embed_query(self, text):
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        return self.embed_documents([text])[0]