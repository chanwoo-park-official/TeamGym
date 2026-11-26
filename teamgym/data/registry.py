"""Domain and construction-command metadata, without importing any pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re


DOMAIN_LABELS = {
    "legal": "Legal",
    "finance": "Finance",
    "medical": "Medical",
    "swe": "Software Engineering",
}
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]*\Z")


@dataclass(frozen=True)
class PipelineSpec:
    """One construction module exposing main(), with optional legacy aliases."""

    domain: str
    dataset: str
    module: str
    aliases: tuple[str, ...] = ()

    @property
    def qualified_name(self) -> str:
        return f"{self.domain}/{self.dataset}"


def build_registry(pipelines: Iterable[PipelineSpec]) -> dict[str, PipelineSpec]:
    """Reject ambiguous command names before loading any construction code."""
    registry: dict[str, PipelineSpec] = {}
    for pipeline in pipelines:
        if pipeline.domain not in DOMAIN_LABELS:
            raise ValueError(f"Unknown domain: {pipeline.domain}")
        if not _IDENTIFIER.fullmatch(pipeline.dataset):
            raise ValueError(f"Invalid dataset name: {pipeline.dataset!r}")
        if not pipeline.module or not all(part.isidentifier() for part in pipeline.module.split(".")):
            raise ValueError(f"Invalid pipeline module: {pipeline.module!r}")
        for alias in pipeline.aliases:
            if not _IDENTIFIER.fullmatch(alias):
                raise ValueError(f"Invalid pipeline alias: {alias!r}")
        for name in (pipeline.qualified_name, *pipeline.aliases):
            if name in registry:
                raise ValueError(f"Duplicate pipeline command: {name}")
            registry[name] = pipeline
    return registry


# Register construction modules when their code is present in this checkout.
# Finance RL preparation is separate from the construction commands below.
PIPELINE_SPECS = tuple(
    PipelineSpec(
        domain="legal",
        dataset=name,
        module=f"teamgym.data.legal.{name}.{module}",
        aliases=(name,),
    )
    for name, module in (
        ("cuad", "pipeline"), ("maud", "pipeline"), ("casebook", "generation"),
        ("mslr", "generation"), ("sara", "pipeline"), ("multilexsum", "pipeline"),
    )
) + (
    PipelineSpec("finance", "findver", "teamgym.data.finance.findver.free_response_funnel", ("findver",)),
    PipelineSpec("finance", "finlongdocqa", "teamgym.data.finance.finlongdocqa.pipeline", ("finlongdocqa",)),
    PipelineSpec("medical", "evidencebench", "teamgym.data.medical.evidencebench.pipeline", ("evidencebench",)),
    PipelineSpec("medical", "mlcr", "teamgym.data.medical.mlcr.cli", ("mlcr",)),
    PipelineSpec("swe", "swe-atlas", "teamgym.data.software.swe_atlas.pipeline", ("swe-atlas",)),
    PipelineSpec("swe", "deep-swe", "teamgym.data.software.deep_swe.pipeline", ("deep-swe",)),
)
PIPELINE_REGISTRY = build_registry(PIPELINE_SPECS)
PIPELINES = {name: pipeline.module for name, pipeline in PIPELINE_REGISTRY.items()}
