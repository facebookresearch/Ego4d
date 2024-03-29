from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class PathSpecification:
    source_path: str
    relative_path: str
    views: Optional[List[str]] = field(
        default_factory=lambda: None, compare=False, hash=False
    )
    universities: Optional[List[str]] = field(
        default_factory=lambda: None, compare=False, hash=False
    )
    file_type: Optional[str] = field(
        default_factory=lambda: None, compare=False, hash=False
    )
    size: Optional[int] = field(default_factory=lambda: None, compare=False, hash=False)
    checksum: Optional[str] = field(
        default_factory=lambda: None, compare=False, hash=False
    )


@dataclass_json
@dataclass(frozen=True)
class ManifestEntry:
    uid: str
    paths: List[PathSpecification]
    splits: Optional[List[str]] = field(
        default_factory=lambda: None, compare=False, hash=False
    )
    benchmarks: Optional[List[str]] = field(
        default_factory=lambda: None, compare=False, hash=False
    )


def manifest_dumps(xs: List[ManifestEntry]) -> str:
    return ManifestEntry.schema().dumps(xs, many=True)  # pyre-ignore


def manifest_loads(data: str) -> List[ManifestEntry]:
    return ManifestEntry.schema().loads(data, many=True)  # pyre-ignore
