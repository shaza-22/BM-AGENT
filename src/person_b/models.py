"""Core data models and schemas for Person B."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union


# ============================================================================
# Enums
# ============================================================================

class SubGoalStatus(str, Enum):
    """Lifecycle states for a sub-goal."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    PARTIAL = "partial"
    NOT_AVAILABLE = "not_available"
    FAILED = "failed"


class ValidationStatus(str, Enum):
    """Outcome status for sub-goal validation against a page."""
    RESOLVED = "resolved"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"
    UNREADABLE = "unreadable"


class ContentType(str, Enum):
    """Content type of fetched or loaded page material."""
    HTML = "html"
    TEXT = "text"
    PDF = "pdf"
    UNKNOWN = "unknown"


class ExtractionLocationType(str, Enum):
    """Location type where an evidence fragment was extracted."""
    TEXT_TABLE = "text_table"
    PDF_TABLE = "pdf_table"
    PROSE = "prose"
    LIST = "list"
    HEADING = "heading"
    METADATA = "metadata"
    OTHER = "other"


class ClaimStatus(str, Enum):
    """Verification status for an individual factual claim."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    PARTIAL = "partial"


class TaskType(str, Enum):
    """Classified category of research task."""
    LOOKUP = "lookup"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    MULTI_HOP = "multi_hop"
    TRAP = "trap"
    GENERAL = "general"


# ============================================================================
# Helper Functions for Serialization
# ============================================================================

def _to_serializable(val: Any) -> Any:
    """Recursively convert dataclasses and enums into JSON-serializable primitives."""
    if isinstance(val, Enum):
        return val.value
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return val.to_dict()
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_to_serializable(v) for v in val]
    if isinstance(val, bytes):
        return f"<bytes: {len(val)} bytes>"
    return val


# ============================================================================
# Core Data Models
# ============================================================================

@dataclass
class Goal:
    """Normalized representation of user task intent."""
    id: str
    statement: str
    original_task: str
    task_type: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "id": self.id,
            "statement": self.statement,
            "original_task": self.original_task,
            "task_type": self.task_type,
            "constraints": self.constraints,
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            id=data["id"],
            statement=data["statement"],
            original_task=data["original_task"],
            task_type=data.get("task_type"),
            constraints=list(data.get("constraints", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SubGoal:
    """Atomic research step towards satisfying a Goal."""
    id: str
    question: str
    status: SubGoalStatus = SubGoalStatus.PENDING
    target_fields: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    created_from: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "id": self.id,
            "question": self.question,
            "status": self.status.value if isinstance(self.status, SubGoalStatus) else self.status,
            "target_fields": self.target_fields,
            "parent_id": self.parent_id,
            "depends_on": self.depends_on,
            "created_from": self.created_from,
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubGoal":
        status_val = data.get("status", SubGoalStatus.PENDING)
        if isinstance(status_val, str):
            try:
                status = SubGoalStatus(status_val)
            except ValueError:
                status = SubGoalStatus.PENDING
        else:
            status = status_val

        return cls(
            id=data["id"],
            question=data["question"],
            status=status,
            target_fields=list(data.get("target_fields", [])),
            parent_id=data.get("parent_id"),
            depends_on=list(data.get("depends_on", [])),
            created_from=data.get("created_from"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Plan:
    """Orchestration plan maintaining active, resolved, and pending sub-goals."""
    id: str
    goal: Goal
    sub_goals: List[SubGoal] = field(default_factory=list)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "id": self.id,
            "goal": self.goal.to_dict(),
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
            "version": self.version,
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            id=data["id"],
            goal=Goal.from_dict(data["goal"]),
            sub_goals=[SubGoal.from_dict(sg) for sg in data.get("sub_goals", [])],
            version=data.get("version", 1),
            metadata=dict(data.get("metadata", {})),
        )

    def get_sub_goal(self, sub_goal_id: str) -> Optional[SubGoal]:
        for sg in self.sub_goals:
            if sg.id == sub_goal_id:
                return sg
        return None

    @property
    def pending_sub_goals(self) -> List[SubGoal]:
        return [sg for sg in self.sub_goals if sg.status == SubGoalStatus.PENDING]

    @property
    def resolved_sub_goals(self) -> List[SubGoal]:
        return [sg for sg in self.sub_goals if sg.status == SubGoalStatus.RESOLVED]


@dataclass
class PageContext:
    """Page artifact passed from Person A or fixture loader into Person B."""
    content: str
    source_url: Optional[str] = None
    content_type: str = "text"
    status_code: Optional[int] = 200
    pdf_bytes: Optional[bytes] = None
    pdf_path: Optional[str] = None
    pdf_tables: Optional[List[List[List[str]]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "content_type": self.content_type,
            "status_code": self.status_code,
            "content_length": len(self.content) if self.content else 0,
            "has_pdf_bytes": self.pdf_bytes is not None,
            "pdf_path": self.pdf_path,
            "has_pdf_tables": self.pdf_tables is not None,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceLocation:
    """Precise provenance of an extracted value within a page/document."""
    type: ExtractionLocationType = ExtractionLocationType.PROSE
    table_name: Optional[str] = None
    row_index: Optional[int] = None
    col_name: Optional[str] = None
    page_number: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "type": self.type.value if isinstance(self.type, ExtractionLocationType) else self.type,
            "table_name": self.table_name,
            "row_index": self.row_index,
            "col_name": self.col_name,
            "page_number": self.page_number,
            "details": self.details,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceLocation":
        loc_type_val = data.get("type", ExtractionLocationType.PROSE)
        if isinstance(loc_type_val, str):
            try:
                loc_type = ExtractionLocationType(loc_type_val)
            except ValueError:
                loc_type = ExtractionLocationType.PROSE
        else:
            loc_type = loc_type_val

        return cls(
            type=loc_type,
            table_name=data.get("table_name"),
            row_index=data.get("row_index"),
            col_name=data.get("col_name"),
            page_number=data.get("page_number"),
            details=dict(data.get("details", {})),
        )


@dataclass
class Evidence:
    """Evidence record tying an extracted factual value to source provenance."""
    field: str
    value: Any
    source_url: Optional[str] = None
    evidence_text: Optional[str] = None
    location: Optional[EvidenceLocation] = None
    confidence: float = 1.0
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "field": self.field,
            "value": self.value,
            "source_url": self.source_url,
            "evidence_text": self.evidence_text,
            "location": self.location.to_dict() if self.location else None,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        loc = EvidenceLocation.from_dict(data["location"]) if data.get("location") else None
        return cls(
            field=data["field"],
            value=data["value"],
            source_url=data.get("source_url"),
            evidence_text=data.get("evidence_text"),
            location=loc,
            confidence=float(data.get("confidence", 1.0)),
            timestamp=data.get("timestamp"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ExtractedField:
    """Individual field with attached evidence provenance."""
    name: str
    value: Any
    evidence: Optional[Evidence] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }


@dataclass
class ExtractionResult:
    """Structured output from extraction modules."""
    status: str = "success"  # 'success', 'partial', 'unreadable', 'empty', 'failed'
    extracted: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "status": self.status,
            "extracted": self.extracted,
            "evidence": [e.to_dict() for e in self.evidence],
            "missing_fields": self.missing_fields,
            "warnings": self.warnings,
            "source_url": self.source_url,
            "metadata": self.metadata,
        })


@dataclass
class ValidationResult:
    """Decision and structured payload produced by sub-goal validation."""
    resolved: bool
    extracted: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    status: ValidationStatus = ValidationStatus.UNRESOLVED
    evidence: List[Evidence] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    source_url: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_contract_dict(self) -> Dict[str, Any]:
        """Return the exact minimum 3-field dictionary required by Person A contract."""
        return {
            "resolved": self.resolved,
            "extracted": self.extracted,
            "reason": self.reason,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return full enriched dictionary including all metadata and evidence."""
        return _to_serializable({
            "resolved": self.resolved,
            "extracted": self.extracted,
            "reason": self.reason,
            "status": self.status.value if isinstance(self.status, ValidationStatus) else self.status,
            "evidence": [e.to_dict() for e in self.evidence],
            "missing_fields": self.missing_fields,
            "source_url": self.source_url,
            "warnings": self.warnings,
            "confidence": self.confidence,
            "metadata": self.metadata,
        })


@dataclass
class ComparisonItem:
    """Entity with attributes and evidence for side-by-side comparison."""
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attributes": self.attributes,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class ComparisonResult:
    """Structured comparison across multiple products/entities."""
    items: List[ComparisonItem] = field(default_factory=list)
    compared_fields: List[str] = field(default_factory=list)
    differences: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "items": [it.to_dict() for it in self.items],
            "compared_fields": self.compared_fields,
            "differences": self.differences,
            "summary": self.summary,
            "metadata": self.metadata,
        })


@dataclass
class Claim:
    """Individual factual assertion generated during synthesis."""
    id: str
    statement: str
    entity: Optional[str] = None
    field: Optional[str] = None
    value: Optional[str] = None
    source_url: Optional[str] = None
    evidence_text: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "entity": self.entity,
            "field": self.field,
            "value": self.value,
            "source_url": self.source_url,
            "evidence_text": self.evidence_text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        return cls(
            id=data["id"],
            statement=data["statement"],
            entity=data.get("entity"),
            field=data.get("field"),
            value=data.get("value"),
            source_url=data.get("source_url"),
            evidence_text=data.get("evidence_text"),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class ClaimCheck:
    """Verification result for a single claim against visited sources."""
    claim: Claim
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    reason: str = ""
    matching_evidence: List[Evidence] = field(default_factory=list)
    verified_source_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "claim": self.claim.to_dict(),
            "status": self.status.value if isinstance(self.status, ClaimStatus) else self.status,
            "reason": self.reason,
            "matching_evidence": [e.to_dict() for e in self.matching_evidence],
            "verified_source_url": self.verified_source_url,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimCheck":
        claim_data = data["claim"]
        claim_obj = Claim.from_dict(claim_data) if isinstance(claim_data, dict) else claim_data
        status_val = data.get("status", ClaimStatus.UNSUPPORTED)
        if isinstance(status_val, str):
            try:
                status = ClaimStatus(status_val)
            except ValueError:
                status = ClaimStatus.UNSUPPORTED
        else:
            status = status_val

        ev_list = []
        for e in data.get("matching_evidence", []):
            if isinstance(e, Evidence):
                ev_list.append(e)
            elif isinstance(e, dict):
                ev_list.append(Evidence.from_dict(e))

        return cls(
            claim=claim_obj,
            status=status,
            reason=data.get("reason", ""),
            matching_evidence=ev_list,
            verified_source_url=data.get("verified_source_url"),
        )


@dataclass
class SynthesisResult:
    """Draft synthesized response before final self-check."""
    task: str
    draft_answer: str
    claims: List[Claim] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "task": self.task,
            "draft_answer": self.draft_answer,
            "claims": [c.to_dict() for c in self.claims],
            "sources": self.sources,
            "missing_info": self.missing_info,
            "metadata": self.metadata,
        })


@dataclass
class FinalAnswer:
    """Final output delivered to user with strict source attribution."""
    answer: str
    source_urls: List[str] = field(default_factory=list)
    not_found: List[str] = field(default_factory=list)
    verified_claims: List[ClaimCheck] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable({
            "answer": self.answer,
            "source_urls": self.source_urls,
            "not_found": self.not_found,
            "verified_claims": [vc.to_dict() for vc in self.verified_claims],
            "metadata": self.metadata,
        })
