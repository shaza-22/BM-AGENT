"""Configuration settings for Person B."""

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class PersonBConfig:
    """Runtime configuration for Person B components."""

    reasoning_provider: str = "fake"  # 'fake', 'gemini', 'openai', etc.
    reasoning_model: str = ""
    reasoning_api_key: Optional[str] = None
    log_dir: str = "logs"
    strict_verification: bool = True
    confidence_threshold: float = 0.7
    max_expansion_sub_goals: int = 20
    max_table_rows: int = 500
    fixtures_dir: str = "fixtures/live"
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "PersonBConfig":
        """Create configuration loaded from environment variables."""
        return cls(
            reasoning_provider=os.getenv("PERSON_B_REASONING_PROVIDER", "fake"),
            reasoning_model=os.getenv("PERSON_B_REASONING_MODEL", ""),
            reasoning_api_key=os.getenv("PERSON_B_REASONING_API_KEY"),
            log_dir=os.getenv("PERSON_B_LOG_DIR", "logs"),
            strict_verification=os.getenv("PERSON_B_STRICT_VERIFICATION", "true").lower()
            in ("true", "1", "yes"),
            confidence_threshold=float(
                os.getenv("PERSON_B_CONFIDENCE_THRESHOLD", "0.7")
            ),
            max_expansion_sub_goals=int(
                os.getenv("PERSON_B_MAX_EXPANSION_SUB_GOALS", "20")
            ),
            max_table_rows=int(os.getenv("PERSON_B_MAX_TABLE_ROWS", "500")),
            fixtures_dir=os.getenv("PERSON_B_FIXTURES_DIR", "fixtures/live"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a safe dictionary (masking API keys)."""
        data = asdict(self)
        if data.get("reasoning_api_key"):
            data["reasoning_api_key"] = "***MASKED***"
        return data


# Default global configuration instance
default_config = PersonBConfig.from_env()
