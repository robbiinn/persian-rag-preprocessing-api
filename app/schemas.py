from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NormalizationMethod(str, Enum):
    basic = "basic"
    embedding_dedup = "embedding_dedup"
    llm_dedup = "llm_dedup"


class ChunkingMethod(str, Enum):
    faq = "faq"
    question_answer = "question_answer"
    semantic = "semantic"


class EmbeddingProvider(str, Enum):
    local = "local"
    openai = "openai"


class NormalizationOptions(BaseModel):
    method: NormalizationMethod = NormalizationMethod.basic
    question_field: str = "سوال"
    answer_field: str = "پاسخ چت بات"
    embedding_provider: EmbeddingProvider = EmbeddingProvider.local
    embedding_model: str | None = None
    question_similarity_threshold: float = Field(0.90, ge=-1, le=1)
    answer_similarity_threshold: float = Field(0.85, ge=-1, le=1)


class ChunkingOptions(BaseModel):
    method: ChunkingMethod = ChunkingMethod.faq
    max_chars: int = Field(600, ge=50, le=100_000)
    overlap_chars: int = Field(80, ge=0, le=20_000)
    answer_length_threshold: int = Field(500, ge=1, le=1_000_000)
    similarity_threshold: float = Field(0.78, ge=-1, le=1)
    min_chunk_chars: int = Field(120, ge=1, le=100_000)
    embedding_provider: EmbeddingProvider = EmbeddingProvider.local
    embedding_model: str | None = None

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        return self


class NormalizeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "records": [{"سوال": "چگونه ثبت نام كنم؟", "پاسخ چت بات": "از سايت وارد شويد."}],
            "options": {"method": "basic"},
        }
    })
    records: list[dict[str, Any]]
    options: NormalizationOptions = Field(default_factory=NormalizationOptions)
    source_file: str | None = None


class ChunkRequest(BaseModel):
    records: list[dict[str, Any]]
    options: ChunkingOptions = Field(default_factory=ChunkingOptions)


class PipelineRequest(BaseModel):
    records: list[dict[str, Any]]
    normalization: NormalizationOptions = Field(default_factory=NormalizationOptions)
    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    source_file: str | None = None


class ProcessResponse(BaseModel):
    ok: bool = True
    input_records: int
    output_records: int
    normalization_method: str | None = None
    chunking_method: str | None = None
    skipped_count: int = 0
    skipped_records: list[dict[str, Any]] = Field(default_factory=list)
    data: Any


class RetrievedItem(BaseModel):
    """One ranked retrieval result.

    ``faq_ids`` may contain more than one ID when the retriever returns an
    expanded/duplicate group instead of a single FAQ.
    """

    rank: int = Field(..., ge=1)
    faq_ids: list[str] = Field(..., min_length=1)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_faq_ids(self):
        cleaned = list(dict.fromkeys(value.strip() for value in self.faq_ids if value.strip()))
        if not cleaned:
            raise ValueError("faq_ids must contain at least one non-empty ID")
        self.faq_ids = cleaned
        return self


class RetrievalEvaluationRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "sample_id": "eval-001",
            "query": "چطور رمز عبورم را تغییر دهم؟",
            "reference_answer": "از بخش تنظیمات حساب، رمز را تغییر دهید.",
            "llm_answer": "برای تغییر رمز وارد تنظیمات حساب شوید.",
            "gold_faq_ids": ["faq-123"],
            "retrieved_items": [
                {"rank": 1, "faq_ids": ["faq-456"], "score": 0.91},
                {"rank": 2, "faq_ids": ["faq-123"], "score": 0.84},
            ],
            "k": 2,
        }
    })

    sample_id: str | None = None
    query: str | None = None
    reference_answer: str | None = None
    llm_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    gold_faq_ids: list[str] = Field(..., min_length=1)
    retrieved_items: list[RetrievedItem] = Field(default_factory=list)
    k: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_evaluation_input(self):
        gold_ids = list(dict.fromkeys(value.strip() for value in self.gold_faq_ids if value.strip()))
        if not gold_ids:
            raise ValueError("gold_faq_ids must contain at least one non-empty ID")
        self.gold_faq_ids = gold_ids

        ranks = [item.rank for item in self.retrieved_items]
        if len(ranks) != len(set(ranks)):
            raise ValueError("retrieved_items ranks must be unique")
        return self


class RetrievalEvaluationResponse(BaseModel):
    sample_id: str | None = None
    query: str | None = None
    reference_answer: str | None = None
    llm_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    k: int
    recall_at_k: float
    mrr_at_k: float
    hit_rate_at_k: float
    relevant_retrieved_count: int
    total_relevant_count: int
    first_relevant_rank: int | None
    matched_faq_ids: list[str]
    missed_faq_ids: list[str]
    ranked_items: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalBatchEvaluationRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "cases": [RetrievalEvaluationRequest.model_config["json_schema_extra"]["example"]],
            "k_values": [1, 3, 5, 10],
        }
    })

    cases: list[RetrievalEvaluationRequest] = Field(..., min_length=1)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10], min_length=1)

    @model_validator(mode="after")
    def validate_k_values(self):
        if any(k < 1 for k in self.k_values):
            raise ValueError("k_values must contain positive integers")
        self.k_values = sorted(set(self.k_values))
        return self


class RetrievalMetricsAtK(BaseModel):
    k: int
    recall_at_k: float
    mrr_at_k: float
    hit_rate_at_k: float


class RetrievalCaseBatchResult(BaseModel):
    sample_id: str | None = None
    query: str | None = None
    reference_answer: str | None = None
    llm_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    gold_faq_ids: list[str]
    ranked_items: list[dict[str, Any]] = Field(default_factory=list)
    metrics: list[RetrievalMetricsAtK]


class RetrievalBatchEvaluationResponse(BaseModel):
    ok: bool = True
    query_count: int
    k_values: list[int]
    aggregate_metrics: list[RetrievalMetricsAtK]
    cases: list[RetrievalCaseBatchResult]


class RagOutputCaseResult(BaseModel):
    sample_id: str | None = None
    query: str | None = None
    reference_answer: str | None = None
    llm_answer: str | None = None
    gold_faq_ids: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    initial_metrics: list[RetrievalMetricsAtK]
    grouped_metrics: list[RetrievalMetricsAtK]
    initial_sources: list[dict[str, Any]] = Field(default_factory=list)
    final_groups: list[dict[str, Any]] = Field(default_factory=list)


class RagOutputEvaluationResponse(BaseModel):
    ok: bool = True
    source_format: str = "api_llama_grouped_output"
    query_count: int
    k_values: list[int]
    initial_aggregate_metrics: list[RetrievalMetricsAtK]
    grouped_aggregate_metrics: list[RetrievalMetricsAtK]
    cases: list[RagOutputCaseResult]
