from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import DEFAULT_ASSISTANT_REQUEST_PARSER_CONFIDENCE_THRESHOLD as DEFAULT_INTENT_PARSER_CONFIDENCE_THRESHOLD

DEFAULT_INTENT_PARSER_MAX_PARSE_ATTEMPTS = 2
DEFAULT_ASSISTANT_REQUEST_PROMPT_PATH = Path("app/interpretation/prompts/assistant_request_parser.md")
DEFAULT_PENDING_ACTION_PROMPT_PATH = Path("app/interpretation/prompts/pending_action_parser.md")


@dataclass(frozen=True)
class IntentParserModelConfig:
    assistant_request_prompt_path: Path = DEFAULT_ASSISTANT_REQUEST_PROMPT_PATH
    pending_action_prompt_path: Path = DEFAULT_PENDING_ACTION_PROMPT_PATH
    confidence_threshold: float = DEFAULT_INTENT_PARSER_CONFIDENCE_THRESHOLD
    max_parse_attempts: int = DEFAULT_INTENT_PARSER_MAX_PARSE_ATTEMPTS

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
        if int(self.max_parse_attempts) < 1:
            raise ValueError("max_parse_attempts must be at least 1.")
