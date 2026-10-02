"""Abstract base class for ICD candidate retrieval engines."""

from abc import ABC, abstractmethod

from medical_coding.schemas.icd import ICDCandidate


class BaseICDRetriever(ABC):
    """Interface for searching authoritative ICD candidates from the local dataset."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[ICDCandidate]:
        """Retrieve top candidate ICD records matching the clinical query.

        Args:
            query: Clinical diagnosis string or expanded context.
            top_k: Maximum number of candidates to return.

        Returns:
            List of ICDCandidate objects sourced strictly from local ICD-10-CM data.
        """
        pass
