import copy
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any
from ..utils.dto import Document



class BaseTextSplitter(ABC):
    """
    Abstract base class for text splitters.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        add_start_index: bool = False,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._add_start_index = add_start_index

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split plain text into list of chunks.

        Args:
            text (str): Raw input text.

        Returns:
            List[str]: List of text chunks.
        """
        pass

    def split_documents(self, documents: List[Union[str, Document]]) -> List[Document]:
        """
        Split list of documents or raw texts into list of Document chunks.

        Args:
            documents: List of strings or Document instances.

        Returns:
            List[Document]: List of chunked Documents.
        """
        texts, metadata = [], []

        for doc in documents:
            if isinstance(doc, str):
                text = doc
                metadata = {}
            elif isinstance(doc, Document):
                text = doc.page_content
                metadata = doc.metadata or {}
            else:
                raise TypeError(f"Unsupported document type: {type(doc)}")

            texts.append(text)
            metadata.append(metadata)

        return self.create_documents(texts, metadata)

    def create_documents(
            self, texts: List[str], 
            metadatas: Optional[list[dict[Any, Any]]] = None
        ) -> List[Document]:
        """
        Create Document instances from list of text chunks.

        Args:
            texts (List[str]): List of text chunks.
            metadata (dict, optional): Metadata to attach to each Document.

        Returns:
            List[Document]: List of Document instances.
        """
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents