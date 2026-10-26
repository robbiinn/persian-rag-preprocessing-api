import json
from typing import Any

from fastapi import HTTPException, UploadFile, status

from app.config import settings


async def parse_upload(file: UploadFile) -> list[dict[str, Any]]:
    filename = file.filename or "upload.json"
    if not filename.lower().endswith((".json", ".jsonl")):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only .json and .jsonl files are supported.")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File exceeds {settings.max_upload_mb} MB.")
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.") from exc
    try:
        if filename.lower().endswith(".jsonl"):
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON at line {exc.lineno}, column {exc.colno}.") from exc
    if isinstance(data, dict):
        if "records" in data:
            data = data["records"]
        else:
            data = [data]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise HTTPException(status_code=400, detail="Input must be a JSON array of objects, an object containing a 'records' array, a single record object, or JSONL objects.")
    validate_record_count(data)
    return data


def validate_record_count(records: list[Any]) -> None:
    if len(records) > settings.max_records:
        raise HTTPException(status_code=413, detail=f"Input exceeds {settings.max_records} records.")
