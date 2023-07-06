from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class PathSpecification:
    source_path: str
    relative_path: str


@dataclass_json
@dataclass
class ManifestEntry:
    uid: str
    paths: List[PathSpecification]


def manifest_dumps(xs: List[ManifestEntry]) -> str:
    return ManifestEntry.schema().dumps(xs, many=True)  # pyre-ignore


def manifest_loads(data: str) -> List[ManifestEntry]:
    return ManifestEntry.schema().loads(data, many=True)  # pyre-ignore
