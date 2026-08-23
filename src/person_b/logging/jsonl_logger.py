"""Structured JSONL logger for Person B pipeline execution events."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PersonBLogEvent:
    """Single structured log event entry."""
    event_type: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    run_id: Optional[str] = None
    plan_id: Optional[str] = None
    sub_goal_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "sub_goal_id": self.sub_goal_id,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class PersonBLogger:
    """Logger that writes structured JSON Lines entries without crashing on write errors."""

    def __init__(
        self,
        log_dir: str = "logs",
        run_id: Optional[str] = None,
        enabled: bool = True,
        strict_logging: bool = False,
        buffer_only: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.enabled = enabled
        self.strict_logging = strict_logging
        self.buffer_only = buffer_only
        self.log_file = self.log_dir / "person_b_events.jsonl"
        self._events: List[PersonBLogEvent] = []

    def _sanitize_data(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Sanitize payload to avoid leaking secrets or logging huge binary blobs."""
        if not data:
            return {}
        sanitized = {}
        for k, v in data.items():
            k_lower = k.lower()
            if "key" in k_lower or "secret" in k_lower or "password" in k_lower or "auth" in k_lower:
                sanitized[k] = "***MASKED***"
            elif isinstance(v, bytes):
                sanitized[k] = f"<bytes: {len(v)} bytes>"
            elif isinstance(v, str) and len(v) > 2000:
                sanitized[k] = v[:500] + f"... [truncated {len(v)} chars]"
            else:
                sanitized[k] = v
        return sanitized

    def log(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        plan_id: Optional[str] = None,
        sub_goal_id: Optional[str] = None,
    ) -> Optional[PersonBLogEvent]:
        """Emit a structured log event."""
        if not self.enabled:
            return None

        event = PersonBLogEvent(
            event_type=event_type,
            run_id=self.run_id,
            plan_id=plan_id,
            sub_goal_id=sub_goal_id,
            data=self._sanitize_data(data),
        )
        self._events.append(event)

        if not self.buffer_only:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except Exception as e:
                if self.strict_logging:
                    raise e

        return event

    def flush(self) -> None:
        """Flush in-memory events to disk and clear buffer."""
        if not self.enabled:
            return
        if self.buffer_only and self._events:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    for ev in self._events:
                        f.write(ev.to_json() + "\n")
            except Exception as e:
                if self.strict_logging:
                    raise e
        self._events.clear()

    def log_task_understood(self, task: str, goal_id: str, task_type: str) -> None:
        self.log("task_understood", {"task": task, "goal_id": goal_id, "task_type": task_type})

    def log_plan_created(self, plan_id: str, sub_goals_count: int) -> None:
        self.log("plan_created", {"plan_id": plan_id, "sub_goals_count": sub_goals_count}, plan_id=plan_id)

    def log_sub_goal_added(self, plan_id: str, sub_goal_id: str, question: str, parent_id: Optional[str] = None) -> None:
        self.log("sub_goal_added", {"question": question, "parent_id": parent_id}, plan_id=plan_id, sub_goal_id=sub_goal_id)

    def log_plan_expanded(self, plan_id: str, added_count: int, new_total: int) -> None:
        self.log("plan_expanded", {"added_count": added_count, "new_total": new_total}, plan_id=plan_id)

    def log_extraction_result(self, source_url: Optional[str], status: str, tables_count: int, entities_count: int, evidence_count: int) -> None:
        self.log("extraction_result", {
            "source_url": source_url,
            "status": status,
            "tables_count": tables_count,
            "entities_count": entities_count,
            "evidence_count": evidence_count,
        })

    def log_validation_result(self, sub_goal_id: Optional[str], resolved: bool, status: str, reason: str, source_url: Optional[str]) -> None:
        self.log("validation_result", {
            "resolved": resolved,
            "status": status,
            "reason": reason,
            "source_url": source_url,
        }, sub_goal_id=sub_goal_id)

    def log_comparison(self, items_count: int, compared_fields_count: int) -> None:
        self.log("comparison_created", {"items_count": items_count, "compared_fields_count": compared_fields_count})

    def log_recommendation(self, recommended_item: Optional[str], rationale: str) -> None:
        self.log("recommendation_created", {"recommended_item": recommended_item, "rationale": rationale})

    def log_synthesis(self, claims_count: int, sources_count: int) -> None:
        self.log("synthesis_created", {"claims_count": claims_count, "sources_count": sources_count})

    def log_claim_verification(self, total_claims: int, supported_count: int, unsupported_count: int) -> None:
        self.log("claim_verification", {
            "total_claims": total_claims,
            "supported_count": supported_count,
            "unsupported_count": unsupported_count,
        })

    def log_not_found(self, item: str, reason: str) -> None:
        self.log("not_found_decision", {"item": item, "reason": reason})

    def log_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.log("error", {"error_type": error_type, "message": message, "details": details or {}})
