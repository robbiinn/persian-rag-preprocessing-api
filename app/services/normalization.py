from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import html
import json
import math
import re
import time
import unicodedata
import uuid
from typing import Any

import numpy as np
from hazm import Normalizer

from app.config import settings
from app.schemas import NormalizationOptions
from app.services.embeddings import get_embedding_function

FAQ_NAMESPACE = uuid.UUID("2b5d286c-a85d-4bd4-a105-9b8a80b37d69")
DROP_FIELDS: set[str] = set()
PERSIAN_CHAR_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ؤ": "و", "إ": "ا", "أ": "ا"})
DIGIT_MAP = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹" + "٠١٢٣٤٥٦٧٨٩", "0123456789" + "0123456789"
)
CONTRAST_PAIRS = [
    ("آنلاین", "آفلاین"), ("حضوری", "غیرحضوری"), ("خرید", "فروش"),
    ("واریز", "برداشت"), ("حقیقی", "حقوقی"), ("فعال", "غیرفعال"),
    ("ورود", "ثبت نام"), ("پذیرفته", "رد شده"),
]
_normalizer = Normalizer()

# Protect machine-readable tokens while Hazm rewrites punctuation and spacing.
PROTECTED_TOKEN_RE = re.compile(
    r"""(?ix)
    (?:
        https?://[^\s<>"'«»]+                         # URLs
        | www\.[^\s<>"'«»]+                           # www URLs
        | [\w.+-]+@[\w.-]+\.[a-z]{2,}                 # email addresses
        | (?<![\w.])\d+(?:[.-]\d+){2,}(?![\w.])      # version-like numbers
        | (?<![\w])(?:\+?\d[\d()\s.-]{5,}\d)(?![\w]) # phone-like numbers
    )
    """
)


def _protect_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Replace URLs, emails, phone numbers and version strings with safe markers."""
    protected: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        marker = f"ZXQPROTECTEDTOKEN{len(protected)}QXZ"
        protected[marker] = match.group(0)
        return marker

    return PROTECTED_TOKEN_RE.sub(replace, text), protected


def _restore_tokens(text: str, protected: dict[str, str]) -> str:
    """Restore protected values after normalization, tolerating marker spacing."""
    for marker, original in protected.items():
        flexible_marker = r"\s*".join(re.escape(char) for char in marker)
        text = re.sub(flexible_marker, lambda _: original, text, flags=re.IGNORECASE)
        # Fallback to the literal placeholder in case the normalizer changed
        # its internal spacing in an unexpected way.
        text = text.replace(marker, original)
    return text


def is_missing(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    return isinstance(value, float) and math.isnan(value)


def make_json_safe(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def clean_and_normalize(value: Any) -> str:
    if is_missing(value):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    for char in ("\u200e", "\u200f", "\ufeff", "\u2066", "\u2067", "\u2068", "\u2069"):
        text = text.replace(char, "")
    text = text.replace("\u00a0", " ").translate(PERSIAN_CHAR_MAP).translate(DIGIT_MAP)
    text, protected_tokens = _protect_tokens(text)
    text = _normalizer.normalize(text)
    text = _restore_tokens(text, protected_tokens)
    for old, new in {"؟": "?", "…": "...", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "’": "'", "‘": "'", "–": "-", "—": "-"}.items():
        text = text.replace(old, new)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r",{2,}", ",", text)
    return re.sub(r"\s+", " ", text).strip()


def _stable_id(question: str, answer: str) -> str:
    return str(uuid.uuid5(FAQ_NAMESPACE, f"question={question}|answer={answer}"))


def _source_id(record: dict[str, Any], question: str, answer: str) -> str:
    for key in ("faq_id", "id"):
        value = record.get(key)
        if not is_missing(value) and str(value).strip().lower() not in {"nan", "none", "null"}:
            return str(value).strip()
    return _stable_id(question, answer)


def _prepare(records: list[dict[str, Any]], options: NormalizationOptions, source_file: str | None) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"Record {index} must be a JSON object.")
        question = clean_and_normalize(raw.get(options.question_field, raw.get("question_normalized")))
        answer = clean_and_normalize(raw.get(options.answer_field, raw.get("answer_normalized")))
        if not question or not answer:
            continue
        signature = (question, answer)
        if signature in seen:
            continue
        seen.add(signature)
        faq_id = _source_id(raw, question, answer)
        excluded = {options.question_field, options.answer_field, "question_normalized", "answer_normalized"} | DROP_FIELDS
        source_metadata = {str(k): make_json_safe(v) for k, v in raw.items() if k not in excluded and not is_missing(v)}
        prepared.append({
            "chunk_id": faq_id,
            "question_normalized": question,
            "answer_normalized": answer,
            "embedding_text": f"سوال: {question}\nپاسخ: {answer}",
            "metadata": {
                "faq_id": faq_id, "source_type": "json_faq", "source_file": source_file,
                "chunking_method": "whole_qa_pair", "chunk_level": 0, "parent_id": None,
                "language": "fa", "embedding_text_version": "normalized",
                "duplicate_group_id": None, "is_canonical": True,
                "duplicate_status": "unique", "source_metadata": source_metadata,
                "processing_steps": ["persian_normalization", "exact_deduplication"],
            },
        })
    return prepared


def _cosine_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    return matrix @ matrix.T


def _has_contrast(a: str, b: str) -> bool:
    return any((x in a and y in b) or (y in a and x in b) for x, y in CONTRAST_PAIRS)


def _embedding_dedup(records: list[dict[str, Any]], options: NormalizationOptions) -> list[dict[str, Any]]:
    if len(records) < 2:
        return records
    embed = get_embedding_function(options.embedding_provider.value, options.embedding_model)
    q_sim = _cosine_matrix(embed([r["question_normalized"] for r in records]))
    a_sim = _cosine_matrix(embed([r["answer_normalized"] for r in records]))
    removed: set[int] = set()
    for i in range(len(records)):
        if i in removed:
            continue
        group: list[int] = []
        for j in range(i + 1, len(records)):
            if j in removed or _has_contrast(records[i]["question_normalized"], records[j]["question_normalized"]):
                continue
            if q_sim[i, j] >= options.question_similarity_threshold and a_sim[i, j] >= options.answer_similarity_threshold:
                group.append(j)
        if group:
            group_id = str(uuid.uuid5(FAQ_NAMESPACE, f"group={records[i]['question_normalized']}"))
            records[i]["metadata"].update({"duplicate_group_id": group_id, "duplicate_status": "canonical"})
            records[i]["metadata"]["processing_steps"].append("embedding_deduplication")
            removed.update(group)
    return [record for index, record in enumerate(records) if index not in removed]


def _llm_judge(first: dict[str, Any], second: dict[str, Any]) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for llm_dedup.")
    from openai import OpenAI
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    client = OpenAI(**kwargs)
    prompt = (
        "دو FAQ فارسی را طبقه‌بندی کن. DUPLICATE یعنی intent و پاسخ یکسان؛ VARIANT یعنی intent یکسان "
        "ولی روش یا شرط پاسخ متفاوت؛ در غیر این صورت UNRELATED. فقط JSON بده: "
        '{"label":"DUPLICATE|VARIANT|UNRELATED"}\n'
        f"A سؤال: {first['question_normalized']}\nA پاسخ: {first['answer_normalized']}\n"
        f"B سؤال: {second['question_normalized']}\nB پاسخ: {second['answer_normalized']}"
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "system", "content": "You are a strict Persian FAQ deduplication judge."}, {"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            label = str(json.loads(response.choices[0].message.content or "{}").get("label", "UNRELATED")).upper()
            return label if label in {"DUPLICATE", "VARIANT", "UNRELATED"} else "UNRELATED"
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"LLM deduplication failed: {last_error}")


def _llm_dedup(records: list[dict[str, Any]], options: NormalizationOptions) -> list[dict[str, Any]]:
    if len(records) < 2:
        return records
    embed = get_embedding_function(options.embedding_provider.value, options.embedding_model)
    similarities = _cosine_matrix(embed([r["question_normalized"] for r in records]))
    removed: set[int] = set()
    for i in range(len(records)):
        if i in removed:
            continue
        for j in range(i + 1, len(records)):
            if j in removed or similarities[i, j] < options.question_similarity_threshold:
                continue
            if _has_contrast(records[i]["question_normalized"], records[j]["question_normalized"]):
                continue
            label = _llm_judge(records[i], records[j])
            if label == "UNRELATED":
                continue
            group_id = records[i]["metadata"].get("duplicate_group_id") or str(uuid.uuid5(FAQ_NAMESPACE, f"group={records[i]['question_normalized']}"))
            records[i]["metadata"].update({"duplicate_group_id": group_id, "duplicate_status": "canonical"})
            records[i]["metadata"]["processing_steps"].append("llm_deduplication")
            if label == "DUPLICATE":
                removed.add(j)
            else:
                records[j]["metadata"].update({"duplicate_group_id": group_id, "duplicate_status": "variant", "is_canonical": False})
                records[j]["metadata"]["processing_steps"].append("llm_deduplication")
    return [record for index, record in enumerate(records) if index not in removed]


def normalize_records(records: list[dict[str, Any]], options: NormalizationOptions, source_file: str | None = None) -> list[dict[str, Any]]:
    prepared = _prepare(deepcopy(records), options, source_file)
    if options.method.value == "embedding_dedup":
        return _embedding_dedup(prepared, options)
    if options.method.value == "llm_dedup":
        return _llm_dedup(prepared, options)
    return prepared
