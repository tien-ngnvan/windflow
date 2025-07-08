from abc import ABC
from typing import Optional, Any


class VectorStore(ABC):
    def add_texts(
            self,
            texts: list[str],
            metadatas: list[dict] | None = None,
    ):
        """Add texts to the vector store.

        Args:
            texts (list[str]): List of texts to add.
            metadatas (list[dict] | None): Optional list of metadata dictionaries corresponding to the texts.
        """
        pass

    def delete(
            self,
            ids: Optional[list[str]]=None,
            **kwargs: Any
    ):
        """Delete texts from the vector store by their IDs.

        Args:
            ids (list[str]): List of IDs of the texts to delete.
        """
        pass

    def search(
            query: str,
            k: int = 4,
            **kwargs: Any
    ) -> list[dict]:
        """Search for texts in the vector store.

        Args:
            query (str): The search query.
            k (int): The number of results to return.
        Returns:
            list[dict]: List of search results, each containing text and metadata.
        """
        pass