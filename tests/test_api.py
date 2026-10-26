import io
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from app.main import app
from app.schemas import ChunkingMethod, ChunkingOptions
from app.services.chunking import chunk_records
from app.services.normalization import clean_and_normalize


SAMPLE = [
    {"سوال": "چگونه ثبت نام كنم؟؟", "پاسخ چت بات": "<p>از سايت وارد شويد.</p>", "دسته": "راهنما"},
    {"سوال": "چگونه ثبت نام كنم؟؟", "پاسخ چت بات": "<p>از سايت وارد شويد.</p>"},
]


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    async def test_basic_pipeline_and_exact_dedup(self):
        response = await self.client.post("/v1/pipeline", json={"records": SAMPLE})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["input_records"], 2)
        self.assertEqual(body["output_records"], 1)
        self.assertEqual(body["data"][0]["question_normalized"], "چگونه ثبت‌نام کنم?")
        self.assertIn("persian_normalization", body["data"][0]["metadata"]["processing_steps"])

    async def test_file_upload(self):
        payload = ('[{"سوال":"سوال اول", "پاسخ چت بات":"پاسخ اول"}]').encode()
        response = await self.client.post(
            "/v1/pipeline/file",
            files={"file": ("faq.json", io.BytesIO(payload), "application/json")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["output_records"], 1)

    async def test_wrapped_records_file_upload(self):
        payload = b'{"records":[{"سوال":"سوال اول","پاسخ چت بات":"پاسخ اول"}]}'
        response = await self.client.post(
            "/v1/pipeline/file",
            files={"file": ("pipeline.json", io.BytesIO(payload), "application/json")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["input_records"], 1)
        self.assertEqual(body["output_records"], 1)
        self.assertEqual(body["skipped_count"], 0)

    async def test_skipped_records_are_reported(self):
        response = await self.client.post(
            "/v1/normalize",
            json={"records": [
                {"سوال": "سوال معتبر", "پاسخ چت بات": "پاسخ معتبر"},
                {"سوال": "سوال بدون پاسخ", "پاسخ چت بات": ""},
            ]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["skipped_count"], 1)
        self.assertEqual(body["skipped_records"], [{
            "record_index": 1,
            "reason": "missing_required_fields",
            "missing_fields": ["پاسخ چت بات"],
        }])

    async def test_all_invalid_records_return_422(self):
        response = await self.client.post(
            "/v1/pipeline",
            json={"records": [{"سوال": "", "پاسخ چت بات": "پاسخ"}]},
        )
        self.assertEqual(response.status_code, 422, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["skipped_count"], 1)
        self.assertEqual(detail["skipped_records"][0]["record_index"], 0)

    async def test_custom_question_and_answer_field_names(self):
        response = await self.client.post(
            "/v1/normalize",
            json={
                "records": [{"question": "سوال", "answer": "پاسخ"}],
                "options": {"question_field": "question", "answer_field": "answer"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["output_records"], 1)
        self.assertEqual(response.json()["skipped_count"], 0)

    async def test_bad_extension(self):
        response = await self.client.post("/v1/normalize/file", files={"file": ("faq.txt", b"[]", "text/plain")})
        self.assertEqual(response.status_code, 415)

    def test_question_answer_really_chunks(self):
        records = [{"chunk_id": "1", "question_normalized": "سوال", "answer_normalized": "کلمه " * 80, "metadata": {}}]
        data = chunk_records(records, ChunkingOptions(method=ChunkingMethod.question_answer, max_chars=100, overlap_chars=10))
        self.assertGreater(len(data["answer_chunks"]), 1)
        self.assertTrue(all(len(item["answer_normalized"]) <= 100 for item in data["answer_chunks"]))

    def test_persian_normalization(self):
        self.assertEqual(clean_and_normalize("  يک&nbsp;متن؟؟  "), "یک متن?")

    def test_overlap_must_be_smaller_than_chunk(self):
        with self.assertRaises(ValidationError):
            ChunkingOptions(max_chars=50)

    async def test_download_header_and_custom_fields(self):
        response = await self.client.post(
            "/v1/normalize?download=true",
            json={
                "records": [{"question": "سوال", "answer": "پاسخ"}],
                "options": {"question_field": "question", "answer_field": "answer"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("attachment", response.headers["content-disposition"])

    async def test_retrieval_evaluation(self):
        response = await self.client.post(
            "/v1/evaluate/retrieval",
            json={
                "sample_id": "eval-1",
                "query": "سوال آزمایشی",
                "reference_answer": "پاسخ صحیح",
                "llm_answer": "پاسخ تولیدشده",
                "metadata": {"dataset": "test"},
                "gold_faq_ids": ["faq-2", "faq-3"],
                "retrieved_items": [
                    {"rank": 1, "faq_ids": ["faq-1"], "score": 0.9},
                    {"rank": 2, "faq_ids": ["faq-2"], "score": 0.8},
                    {"rank": 3, "faq_ids": ["faq-3"], "score": 0.7},
                ],
                "k": 2,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["recall_at_k"], 0.5)
        self.assertEqual(body["mrr_at_k"], 0.5)
        self.assertEqual(body["hit_rate_at_k"], 1.0)
        self.assertEqual(body["first_relevant_rank"], 2)
        self.assertEqual(body["matched_faq_ids"], ["faq-2"])
        self.assertEqual(body["missed_faq_ids"], ["faq-3"])
        self.assertEqual(body["reference_answer"], "پاسخ صحیح")
        self.assertEqual(body["llm_answer"], "پاسخ تولیدشده")
        self.assertEqual(body["metadata"], {"dataset": "test"})
        self.assertFalse(body["ranked_items"][0]["is_relevant"])
        self.assertTrue(body["ranked_items"][1]["is_relevant"])
        self.assertTrue(body["ranked_items"][1]["included_at_k"])

    async def test_batch_retrieval_evaluation(self):
        response = await self.client.post(
            "/v1/evaluate/retrieval/batch",
            json={
                "k_values": [3, 1, 3],
                "cases": [
                    {
                        "sample_id": "q1",
                        "gold_faq_ids": ["a"],
                        "retrieved_items": [
                            {"rank": 1, "faq_ids": ["x"]},
                            {"rank": 2, "faq_ids": ["a"]},
                        ],
                    },
                    {
                        "sample_id": "q2",
                        "gold_faq_ids": ["b", "c"],
                        "retrieved_items": [{"rank": 1, "faq_ids": ["b"]}],
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["query_count"], 2)
        self.assertEqual(body["k_values"], [1, 3])
        self.assertEqual(body["aggregate_metrics"][0], {
            "k": 1, "recall_at_k": 0.25, "mrr_at_k": 0.5, "hit_rate_at_k": 0.5,
        })
        self.assertEqual(body["aggregate_metrics"][1], {
            "k": 3, "recall_at_k": 0.75, "mrr_at_k": 0.75, "hit_rate_at_k": 1.0,
        })

    async def test_retrieval_evaluation_file(self):
        payload = b'''[
          {"sample_id":"q1","gold_faq_ids":["a"],"retrieved_items":[{"rank":1,"faq_ids":["a"]}]},
          {"sample_id":"q2","gold_faq_ids":["b"],"retrieved_items":[]}
        ]'''
        response = await self.client.post(
            "/v1/evaluate/retrieval/file",
            files={"file": ("evaluation.json", io.BytesIO(payload), "application/json")},
            data={"k_values": "[1,5]", "download": "false"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["query_count"], 2)
        self.assertEqual(body["aggregate_metrics"][0]["recall_at_k"], 0.5)
        self.assertEqual(body["aggregate_metrics"][0]["mrr_at_k"], 0.5)

    async def test_rag_output_file_matches_llama_script_format(self):
        payload = '''[
          {
            "sample_id": "eval-1__colloquial",
            "question": "سؤال آزمایشی",
            "reference_answer": "پاسخ مرجع",
            "generated_answer": "پاسخ مدل",
            "metrics": {"target_faq_id": "faq-target", "query_type": "colloquial"},
            "retrieval_config": {"initial_retrieval_k": 2, "final_group_k": 1},
            "timestamp": "2026-01-01T00:00:00Z",
            "initial_sources": [
              {"rank": 1, "node_id": "n1", "score": 0.9, "text": "متن اول", "metadata": {"faq_id": "faq-other"}},
              {"rank": 2, "node_id": "n2", "score": 0.8, "text": "متن دوم", "metadata": {"faq_id": "faq-target"}}
            ],
            "final_groups": [
              {"group_rank": 1, "group_id": "g1", "best_score": 0.9, "faq_ids": ["faq-other", "faq-target"], "merged_context": "متن گروه"}
            ]
          }
        ]'''.encode()
        response = await self.client.post(
            "/v1/evaluate/retrieval/rag-output/file",
            files={"file": ("rag-result.json", io.BytesIO(payload), "application/json")},
            # Swagger UI serializes the visible value [1,2] as 1,2.
            data={"k_values": "1,2", "download": "false"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["source_format"], "api_llama_grouped_output")
        self.assertEqual(body["initial_aggregate_metrics"][0]["recall_at_k"], 0.0)
        self.assertEqual(body["initial_aggregate_metrics"][1]["mrr_at_k"], 0.5)
        self.assertEqual(body["grouped_aggregate_metrics"][0]["recall_at_k"], 1.0)
        self.assertEqual(body["grouped_aggregate_metrics"][0]["mrr_at_k"], 1.0)
        self.assertEqual(body["cases"][0]["llm_answer"], "پاسخ مدل")
        self.assertEqual(body["cases"][0]["reference_answer"], "پاسخ مرجع")
        self.assertEqual(body["cases"][0]["gold_faq_ids"], ["faq-target"])

    async def test_rag_output_file_rejects_invalid_k_values(self):
        payload = b'[{"metrics":{"target_faq_id":"a"},"initial_sources":[],"final_groups":[]}]'
        response = await self.client.post(
            "/v1/evaluate/retrieval/rag-output/file",
            files={"file": ("rag-result.json", io.BytesIO(payload), "application/json")},
            data={"k_values": "1,wrong"},
        )
        self.assertEqual(response.status_code, 422)

    async def test_grouped_retrieval_evaluation(self):
        response = await self.client.post(
            "/v1/evaluate/retrieval",
            json={
                "gold_faq_ids": ["faq-target"],
                "retrieved_items": [
                    {"rank": 1, "faq_ids": ["faq-a", "faq-target"]},
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["recall_at_k"], 1.0)
        self.assertEqual(body["mrr_at_k"], 1.0)
        self.assertEqual(body["k"], 1)

    async def test_retrieval_evaluation_rejects_duplicate_ranks(self):
        response = await self.client.post(
            "/v1/evaluate/retrieval",
            json={
                "gold_faq_ids": ["faq-1"],
                "retrieved_items": [
                    {"rank": 1, "faq_ids": ["faq-1"]},
                    {"rank": 1, "faq_ids": ["faq-2"]},
                ],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_semantic_chunking_with_injected_embedder(self):
        records = [{
            "chunk_id": "faq-1",
            "question_normalized": "سوال",
            "answer_normalized": "جمله اول طولانی است. جمله دوم متفاوت است. جمله سوم هم متفاوت است.",
            "metadata": {},
        }]
        options = ChunkingOptions(
            method=ChunkingMethod.semantic,
            answer_length_threshold=1,
            min_chunk_chars=5,
            similarity_threshold=0.5,
        )
        fake_vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        with patch("app.services.chunking.get_embedding_function", return_value=lambda _: fake_vectors):
            data = chunk_records(records, options)
        self.assertEqual(len(data), 3)
        self.assertTrue(all(item["metadata"]["is_split"] for item in data))


if __name__ == "__main__":
    unittest.main()
