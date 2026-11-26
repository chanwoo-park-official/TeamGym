"""Compatibility imports for shared written-answer scoring."""

from __future__ import annotations

from teamgym.scoring.written_answers import (
    Any as Any,
    CACHE_SCHEMA as CACHE_SCHEMA,
    JUDGE_SCHEMA as JUDGE_SCHEMA,
    OpenAIJsonClient as OpenAIJsonClient,
    OpenAIJsonClientConfig as OpenAIJsonClientConfig,
    Path as Path,
    WrittenJudgeConfig as WrittenJudgeConfig,
    _cache_path as _cache_path,
    dataclass as dataclass,
    hashlib as hashlib,
    holistic_score as holistic_score,
    is_written_answer_task as is_written_answer_task,
    json as json,
    judge_one_answer as judge_one_answer,
    judge_written_episode as judge_written_episode,
    os as os,
    uuid4 as uuid4,
    written_answer_from_episode as written_answer_from_episode,
)
