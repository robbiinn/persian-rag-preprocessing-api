import json
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.config import settings
from app.file_io import parse_upload, validate_record_count
from app.schemas import (
    ChunkRequest, ChunkingMethod, ChunkingOptions, EmbeddingProvider,
    NormalizationMethod, NormalizationOptions, NormalizeRequest, PipelineRequest,
    ProcessResponse, RetrievalBatchEvaluationRequest,
    RetrievalBatchEvaluationResponse, RetrievalEvaluationRequest,
    RetrievalEvaluationResponse, RagOutputEvaluationResponse,
)
from app.services.chunking import chunk_records, output_count
from app.services.normalization import normalize_records
from app.services.retrieval_evaluation import (
    evaluate_rag_output_records, evaluate_retrieval, evaluate_retrieval_batch,
)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Normalization and chunking service for Persian FAQ/RAG datasets.",
)


def _parse_k_values(raw: str) -> list[int]:
    """Accept a JSON array or Swagger's comma-separated form value."""
    value = raw.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in value.split(",") if part.strip()]

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("k_values must be a non-empty list, for example [1,2] or 1,2.")

    result: list[int] = []
    for item in parsed:
        if isinstance(item, bool):
            raise ValueError("k_values must contain positive integers.")
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("k_values must contain positive integers.") from exc
        if number < 1 or str(item).strip() not in {str(number), f"+{number}"}:
            raise ValueError("k_values must contain positive integers.")
        result.append(number)
    return sorted(set(result))


async def authorize(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.api_key and (not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _response(input_count: int, data, normalization: str | None = None, chunking: str | None = None, download: bool = False, filename: str = "result.json"):
    payload = ProcessResponse(input_records=input_count, output_records=output_count(data) if chunking else len(data), normalization_method=normalization, chunking_method=chunking, data=data).model_dump(mode="json")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if download else None
    return JSONResponse(payload, headers=headers)


def _run_normalize(records, options, source_file=None):
    try:
        return normalize_records(records, options, source_file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _run_chunk(records, options):
    try:
        return chunk_records(records, options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", tags=["system"])
async def root():
    return {"name": settings.app_name, "version": settings.app_version, "docs": "/docs"}


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "version": settings.app_version}


@app.get("/v1/capabilities", dependencies=[Depends(authorize)], tags=["system"])
def capabilities():
    return {"normalization": [item.value for item in NormalizationMethod], "chunking": [item.value for item in ChunkingMethod], "embedding_providers": [item.value for item in EmbeddingProvider], "evaluation": ["recall_at_k", "mrr_at_k", "hit_rate_at_k"], "limits": {"max_upload_mb": settings.max_upload_mb, "max_records": settings.max_records}}


@app.post("/v1/normalize", dependencies=[Depends(authorize)], tags=["json"])
def normalize_json(request: NormalizeRequest, download: bool = Query(False)):
    validate_record_count(request.records)
    data = _run_normalize(request.records, request.options, request.source_file)
    return _response(len(request.records), data, request.options.method.value, download=download, filename="normalized.json")


@app.post("/v1/chunk", dependencies=[Depends(authorize)], tags=["json"])
def chunk_json(request: ChunkRequest, download: bool = Query(False)):
    validate_record_count(request.records)
    data = _run_chunk(request.records, request.options)
    return _response(len(request.records), data, chunking=request.options.method.value, download=download, filename="chunks.json")


@app.post("/v1/pipeline", dependencies=[Depends(authorize)], tags=["json"])
def pipeline_json(request: PipelineRequest, download: bool = Query(False)):
    validate_record_count(request.records)
    normalized = _run_normalize(request.records, request.normalization, request.source_file)
    data = _run_chunk(normalized, request.chunking)
    return _response(len(request.records), data, request.normalization.method.value, request.chunking.method.value, download, "pipeline-result.json")


@app.post(
    "/v1/evaluate/retrieval",
    response_model=RetrievalEvaluationResponse,
    dependencies=[Depends(authorize)],
    tags=["evaluation"],
    summary="Calculate Recall@K and MRR@K for ranked retrieval results",
)
def evaluate_retrieval_endpoint(request: RetrievalEvaluationRequest):
    return evaluate_retrieval(request)


@app.post(
    "/v1/evaluate/retrieval/batch",
    response_model=RetrievalBatchEvaluationResponse,
    dependencies=[Depends(authorize)],
    tags=["evaluation"],
    summary="Calculate aggregate retrieval metrics for multiple queries",
)
def evaluate_retrieval_batch_endpoint(request: RetrievalBatchEvaluationRequest):
    return evaluate_retrieval_batch(request)


def _options(model, raw: str):
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


@app.post("/v1/normalize/file", dependencies=[Depends(authorize)], tags=["file upload"])
async def normalize_file(file: Annotated[UploadFile, File(...)], options: Annotated[str, Form()] = "{}", download: Annotated[bool, Form()] = False):
    records = await parse_upload(file)
    parsed = _options(NormalizationOptions, options)
    data = await run_in_threadpool(_run_normalize, records, parsed, file.filename)
    return _response(len(records), data, parsed.method.value, download=download, filename="normalized.json")


@app.post("/v1/chunk/file", dependencies=[Depends(authorize)], tags=["file upload"])
async def chunk_file(file: Annotated[UploadFile, File(...)], options: Annotated[str, Form()] = "{}", download: Annotated[bool, Form()] = False):
    records = await parse_upload(file)
    parsed = _options(ChunkingOptions, options)
    data = await run_in_threadpool(_run_chunk, records, parsed)
    return _response(len(records), data, chunking=parsed.method.value, download=download, filename="chunks.json")


@app.post("/v1/pipeline/file", dependencies=[Depends(authorize)], tags=["file upload"])
async def pipeline_file(file: Annotated[UploadFile, File(...)], normalization: Annotated[str, Form()] = "{}", chunking: Annotated[str, Form()] = "{}", download: Annotated[bool, Form()] = False):
    records = await parse_upload(file)
    norm_options = _options(NormalizationOptions, normalization)
    chunk_options = _options(ChunkingOptions, chunking)
    normalized = await run_in_threadpool(_run_normalize, records, norm_options, file.filename)
    data = await run_in_threadpool(_run_chunk, normalized, chunk_options)
    return _response(len(records), data, norm_options.method.value, chunk_options.method.value, download, "pipeline-result.json")


@app.post(
    "/v1/evaluate/retrieval/file",
    response_model=RetrievalBatchEvaluationResponse,
    dependencies=[Depends(authorize)],
    tags=["evaluation"],
    summary="Evaluate a JSON/JSONL file containing retrieval cases",
)
async def evaluate_retrieval_file(
    file: Annotated[UploadFile, File(...)],
    k_values: Annotated[str, Form()] = "[1,3,5,10]",
    download: Annotated[bool, Form()] = False,
):
    cases = await parse_upload(file)
    try:
        parsed_k_values = _parse_k_values(k_values)
        request = RetrievalBatchEvaluationRequest(cases=cases, k_values=parsed_k_values)
    except (ValidationError, TypeError, ValueError) as exc:
        detail = json.loads(exc.json()) if isinstance(exc, ValidationError) else str(exc)
        raise HTTPException(status_code=422, detail=detail) from exc
    payload = await run_in_threadpool(evaluate_retrieval_batch, request)
    headers = {"Content-Disposition": 'attachment; filename="retrieval-evaluation.json"'} if download else None
    return JSONResponse(payload, headers=headers)


@app.post(
    "/v1/evaluate/retrieval/rag-output/file",
    response_model=RagOutputEvaluationResponse,
    dependencies=[Depends(authorize)],
    tags=["evaluation"],
    summary="Evaluate grouped RAG output using the documented JSON schema",
)
async def evaluate_rag_output_file(
    file: Annotated[UploadFile, File(...)],
    k_values: Annotated[str, Form()] = "[1,3,5,7,10,12]",
    download: Annotated[bool, Form()] = False,
):
    records = await parse_upload(file)
    try:
        parsed_k_values = _parse_k_values(k_values)
        payload = await run_in_threadpool(evaluate_rag_output_records, records, parsed_k_values)
    except (ValidationError, TypeError, ValueError) as exc:
        detail = json.loads(exc.json()) if isinstance(exc, ValidationError) else str(exc)
        raise HTTPException(status_code=422, detail=detail) from exc
    headers = {"Content-Disposition": 'attachment; filename="rag-output-evaluation.json"'} if download else None
    return JSONResponse(payload, headers=headers)
