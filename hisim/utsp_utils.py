"""Utility classes for UTSP connectors."""

from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class UtspConfig:
    """Config class for all UTSP connectors. Contains UTSP connection parameters."""

    url: str
    api_key: str
