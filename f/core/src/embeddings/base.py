from abc import ABC, abstractmethod

from ..utils.dto import Document



class EmbeddingBase(ABC):
    """
    Base class for embedding models.
    """

    @abstractmethod
    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into vectors.

        Args:
            texts (list[str]): List of texts to embed.

        Returns:
            list[list[float]]: List of embedded vectors.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """
        Embed a list of Document objects into vectors.

        Args:
            documents (list[Document]): List of Document objects to embed.

        Returns:
            list[list[float]]: List of embedded vectors.
        """
        pass