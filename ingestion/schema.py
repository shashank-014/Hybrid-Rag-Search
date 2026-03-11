from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class IngestedDocument:
    source_id: str
    source_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_langchain_document(self) -> Document:
        doc_meta = {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            **self.metadata,
        }
        return Document(page_content=self.content, metadata=doc_meta)



def build_document(
    *,
    source_id: str,
    source_type: str,
    title: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> IngestedDocument:
    return IngestedDocument(
        source_id=source_id,
        source_type=source_type,
        title=title,
        content=content,
        metadata=metadata or {},
    )
