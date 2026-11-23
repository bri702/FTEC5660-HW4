"""Reproduce the Task 5 one-component-at-a-time ablation on the public corpus.

Examples:
    python run_ablation.py --offline
    python run_ablation.py
    python run_ablation.py --variants V7 --workers 8

The ground truth is used only to score outputs. Retrieval receives the corpus and
question text, never answers, evidence, probes, or expected query annotations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


VARIANTS = {
    "B0": {"change": "Prescribed naive baseline", "tables": False, "sections": False,
           "lexical": False, "metadata_filter": False, "field_boost": False,
           "company_balance": False, "abstain": False},
    "V1": {"change": "Table linearisation only", "tables": True, "sections": False,
           "lexical": False, "metadata_filter": False, "field_boost": False,
           "company_balance": False, "abstain": False},
    "V2": {"change": "Section-aware chunking only", "tables": True, "sections": True,
           "lexical": False, "metadata_filter": False, "field_boost": False,
           "company_balance": False, "abstain": False},
    "V3": {"change": "Lexical tokenisation only", "tables": True, "sections": True,
           "lexical": True, "metadata_filter": False, "field_boost": False,
           "company_balance": False, "abstain": False},
    "V4": {"change": "Company/document/year metadata routing only", "tables": True,
           "sections": True, "lexical": True, "metadata_filter": True,
           "field_boost": False, "company_balance": False, "abstain": False},
    "V5": {"change": "Intent-based line reranking only", "tables": True,
           "sections": True, "lexical": True, "metadata_filter": True,
           "field_boost": True, "company_balance": False, "abstain": False},
    "V6": {"change": "Evidence allocation only (one passage per comparison subject; three for a single subject)", "tables": True,
           "sections": True, "lexical": True, "metadata_filter": True,
           "field_boost": True, "company_balance": True, "abstain": False},
    "V7": {"change": "Evidence-based abstention only (final system)", "tables": True,
           "sections": True, "lexical": True, "metadata_filter": True,
           "field_boost": True, "company_balance": True, "abstain": True},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AnswerCache:
    """Cache actual fixed-generator outputs, never an index or expected answers."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def key(question: str, chunks: list[Document]) -> str:
        payload = {"model": hw4.LLM_MODEL, "temperature": 0, "max_tokens": 64,
                   "thinking": False, "prompt": hw4.GENERATOR_PROMPT,
                   "question": question, "context": hw4.format_context(chunks)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def answer(self, question: str, chunks: list[Document]) -> dict:
        started = time.perf_counter()
        if not chunks:
            return {"answer": "NOT_FOUND", "cached": False, "error": "", "seconds": 0.0}
        key = self.key(question, chunks)
        with self.lock:
            if key in self.values:
                return {"answer": self.values[key]["answer"], "cached": True,
                        "error": "", "seconds": time.perf_counter() - started}
        try:
            # Every model request goes through the unmodified provided generator.
            answer = hw4.generate(question, chunks)
        except Exception as exc:
            # Failed requests are deliberately not cached; rerunning retries them.
            return {"answer": "", "cached": False, "error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.perf_counter() - started}
        with self.lock:
            self.values[key] = {"answer": answer}
            write_json(self.path, self.values)
        return {"answer": answer, "cached": False, "error": "",
                "seconds": time.perf_counter() - started}


def baseline_chunks(corpus: dict[str, str], linearise_tables: bool) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = []
    for file_name, html in sorted(corpus.items()):
        metadata = hw4._document_metadata(file_name, html)
        soup = BeautifulSoup(html, "html.parser")
        if linearise_tables:
            for table in soup.find_all("table"):
                table.replace_with(NavigableString(hw4.table_to_text(str(table))))
        # B0 deliberately uses get_text() without table linearisation or headers.
        text = soup.get_text()
        chunks.extend(Document(page_content=part, metadata=dict(metadata))
                      for part in splitter.split_text(text))
    return chunks


def structured_chunks(corpus: dict[str, str]) -> list[Document]:
    return [chunk for file_name, html in sorted(corpus.items())
            for chunk in hw4.chunk_document(file_name, html)]


def whitespace_index(chunks: list[Document]) -> dict:
    return {"chunks": chunks, "bm25": BM25Okapi([c.page_content.lower().split() for c in chunks])}


def whitespace_retrieve(index: dict, question: str) -> list[Document]:
    scores = index["bm25"].get_scores(question.lower().split())
    ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    # The prescribed naive baseline always returns top five, even at zero score.
    return [index["chunks"][i] for i in ranking[:hw4.MAX_RETRIEVED]]


def usage_delta(before: dict, after: dict) -> dict:
    return {key: round(after[key] - before.get(key, 0), 6)
            if isinstance(after[key], float) else after[key] - before.get(key, 0)
            for key in after}


def _json_fingerprint(value: Any) -> str:
    """Hash semantic input content, independent of file paths and JSON layout."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def result_fingerprints(corpus: dict[str, str], questions: dict[str, str],
                        gold: dict | None, offline: bool) -> dict[str, dict]:
    """Identify the code, dataset, configuration and generator behind each row.

    The gold file affects scoring only, but changing it must still invalidate a
    previously scored summary. Response-cache keys remain based on exact model
    inputs, so this result provenance check never prevents legitimate reuse of
    an actual generator response to an identical input.
    """
    common = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(Path(hw4.__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": _json_fingerprint({"corpus": corpus, "questions": questions,
                                        "ground_truth": gold}),
        "generator_sha256": _json_fingerprint({
            "model": hw4.LLM_MODEL, "temperature": 0, "max_tokens": 64,
            "thinking": False, "prompt": hw4.GENERATOR_PROMPT}),
        "mode": "offline" if offline else "live",
    }
    return {name: {**common, "config_sha256": _json_fingerprint({
        "variant": name, "configuration": config})} for name, config in VARIANTS.items()}


def _read_matching_summary(path: Path, name: str, expected: dict) -> dict:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot combine existing {name}: cannot read {path}. "
                         "Rerun all variants or choose a fresh --output-dir.") from exc
    actual = summary.get("fingerprint") if isinstance(summary, dict) else None
    if not isinstance(actual, dict) or actual != expected or summary.get("variant") != name:
        changed = (", ".join(key for key in expected if actual.get(key) != expected[key])
                   if isinstance(actual, dict) else "missing result fingerprint")
        if isinstance(summary, dict) and summary.get("variant") != name:
            changed = ", ".join(filter(None, [changed, "variant name"]))
        raise ValueError(f"Cannot combine existing {name} with this run ({changed}). "
                         "Its code, data, configuration or evaluation mode differs. "
                         "Rerun all variants or choose a fresh --output-dir. "
                         "The existing manifest has not been replaced.")
    return summary


def preflight_existing_results(output_dir: Path, selected: list[str],
                               fingerprints: dict[str, dict]) -> None:
    """Reject stale, unselected rows before writing any new run manifest."""
    for name in VARIANTS:
        path = output_dir / name / "summary.json"
        if name not in selected and path.exists():
            _read_matching_summary(path, name, fingerprints[name])


def run_variant(name: str, corpus: dict[str, str], questions: dict[str, str], gold: dict | None,
                output_dir: Path, offline: bool, workers: int, cache: AnswerCache,
                fingerprint: dict) -> dict:
    config = VARIANTS[name]
    started = time.perf_counter()
    before_own, before_generator = hw4.USAGE.summary(), hw4.GENERATOR_USAGE.summary()
    chunks = structured_chunks(corpus) if config["sections"] else baseline_chunks(corpus, config["tables"])
    # V2 and V3 differ only in the index/query tokenizer. Earlier variants use
    # the exact prescribed whitespace BM25, with no query parsing or rejection.
    index = (hw4._build_index(chunks, token_mode="lexical") if config["lexical"]
             else whitespace_index(chunks))
    index_seconds = time.perf_counter() - started
    retrieved, parsed, valid, retrieval_errors = {}, {}, {}, {}
    allowed = {(c.page_content, c.metadata.get("source")) for c in chunks}
    retrieval_started = time.perf_counter()
    for qid, question in questions.items():
        parsed[qid] = hw4.parse_query(question) if config["metadata_filter"] else None
        try:
            result = (hw4._retrieve_config(index, question,
                        metadata_filter=config["metadata_filter"],
                        field_boost=config["field_boost"],
                        company_balance=config["company_balance"],
                        abstain=config["abstain"])
                      if config["lexical"] else whitespace_retrieve(index, question))
            valid[qid] = (isinstance(result, (list, tuple)) and len(result) <= hw4.MAX_RETRIEVED
                          and all(isinstance(c, Document)
                                  and (c.page_content, c.metadata.get("source")) in allowed for c in result))
            retrieved[qid] = list(result)[:hw4.MAX_RETRIEVED] if valid[qid] else []
        except Exception as exc:
            valid[qid], retrieved[qid] = False, []
            retrieval_errors[qid] = f"{type(exc).__name__}: {exc}"
    retrieval_seconds = time.perf_counter() - retrieval_started
    if offline:
        answers = {qid: {"answer": "", "cached": False, "error": "", "seconds": 0.0}
                   for qid in questions}
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = dict(zip(questions, pool.map(
                lambda qid: cache.answer(questions[qid], retrieved[qid]), questions)))
    rows = []
    for qid, question in questions.items():
        result = answers[qid]
        chunks_for_question = retrieved[qid]
        row = {"id": qid, "category": "", "question": question,
               "parsed": json.dumps(parsed[qid], ensure_ascii=False), "parse_score": "",
               "retrieved": ";".join(sorted({c.metadata.get("source", "") for c in chunks_for_question})),
               "valid": valid[qid], "retrieval_hit": "", "answer": result["answer"],
               "correct": "", "expected": "", "retrieval_count": len(chunks_for_question),
               "generator_cached": result["cached"], "generator_error": result["error"],
               "retrieval_error": retrieval_errors.get(qid, ""),
               "generation_seconds": round(result["seconds"], 4)}
        if gold and qid in gold:
            expected = gold[qid]
            row["category"] = expected["category"]
            if parsed[qid] is not None:
                row["parse_score"] = round(hw4.parse_score(parsed[qid], expected), 3)
            row["retrieval_hit"] = hw4.retrieval_hit(chunks_for_question, expected) if expected["evidence"] else ""
            if not offline:
                row["correct"] = hw4.answer_correct(result["answer"], expected)
                if not row["correct"]:
                    row["expected"] = expected["answer"]
        rows.append(row)
    elapsed = time.perf_counter() - started
    summary = {"variant": name, "change": config["change"], "mode": "offline" if offline else "live",
               "fingerprint": fingerprint,
               "documents": len(corpus), "chunks": len(chunks), "questions": len(questions),
               "valid_nonempty_rate": "", "parse_field_accuracy": "", "recall_at_5": "",
               "kept": "", "rejected": "", "answer_accuracy": "", "task3_points": "",
               "task2_3_points": "", "task4_points": "", "index_seconds": round(index_seconds, 4),
               "retrieval_seconds": round(retrieval_seconds, 4), "total_seconds": round(elapsed, 4),
               "within_time_limit": elapsed <= hw4.TIME_LIMIT_SECONDS,
               "cache_hits": sum(r["generator_cached"] for r in rows),
               "generator_errors": sum(bool(r["generator_error"]) for r in rows),
               "retrieval_errors": len(retrieval_errors)}
    if gold:
        scored = [r for r in rows if r["id"] in gold]
        answerable = [r for r in scored if gold[r["id"]]["evidence"]]
        unanswerable = [r for r in scored if not gold[r["id"]]["evidence"]]
        kept = sum(r["retrieval_count"] > 0 for r in answerable) / max(1, len(answerable))
        rejected = sum(r["retrieval_count"] == 0 for r in unanswerable) / max(1, len(unanswerable))
        recall = sum(bool(r["retrieval_hit"]) for r in answerable) / max(1, len(answerable))
        summary.update({"valid_nonempty_rate": sum(r["valid"] and r["retrieval_count"] > 0 for r in answerable) / max(1, len(answerable)),
                        "recall_at_5": recall, "kept": kept, "rejected": rejected,
                        "task3_points": 10 * max(0, kept + rejected - 1),
                        "task2_3_points": 15 * hw4.clip((recall - .5) / .4)})
        if config["metadata_filter"]:
            summary["parse_field_accuracy"] = sum(hw4.parse_score(parsed[r["id"]], gold[r["id"]]) for r in scored) / max(1, len(scored))
        if not offline:
            accuracy = sum(bool(r["correct"]) for r in scored) / max(1, len(scored))
            summary.update({"answer_accuracy": accuracy, "task4_points": 20 * accuracy})
        summary["by_category"] = {}
        for category in sorted({r["category"] for r in scored}):
            subset = [r for r in scored if r["category"] == category]
            hits = [r for r in subset if r["retrieval_hit"] != ""]
            summary["by_category"][category] = {
                "questions": len(subset),
                "recall_at_5": sum(bool(r["retrieval_hit"]) for r in hits) / len(hits) if hits else None,
                "answer_accuracy": None if offline else sum(bool(r["correct"]) for r in subset) / len(subset)}
    summary["own_usage"] = usage_delta(before_own, hw4.USAGE.summary())
    summary["generator_usage"] = usage_delta(before_generator, hw4.GENERATOR_USAGE.summary())
    variant_dir = output_dir / name
    write_csv(variant_dir / "results.csv", rows)
    write_json(variant_dir / "summary.json", summary)
    # Save exact retrieved contexts so an error can be traced to missing evidence,
    # ambiguous evidence, or a generator mistake without rerunning any API calls.
    write_json(variant_dir / "contexts.json", {qid: hw4.format_context(value) for qid, value in retrieved.items()})
    print(json.dumps({key: summary[key] for key in ("variant", "mode", "recall_at_5", "kept", "rejected", "answer_accuracy", "total_seconds", "cache_hits", "generator_errors")}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated names B0,V1,...,V7")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offline", action="store_true", help="Measure retrieval without calling the generator; answer accuracy stays blank")
    args = parser.parse_args()
    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    if not names or any(name not in VARIANTS for name in names):
        parser.error("--variants must contain only B0,V1,V2,V3,V4,V5,V6,V7")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if not args.offline and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("Set DEEPSEEK_API_KEY in .env beside hw4.py, or use --offline")
    data_folder = Path(args.data_folder)
    corpus = hw4.load_corpus(data_folder / "corpus")
    questions = hw4.load_json(data_folder / "questions.json")
    if not corpus or not isinstance(questions, dict) or not questions:
        parser.error("data folder must contain nonempty corpus/*.html and questions.json")
    gold = hw4.load_json(data_folder / "ground_truth.json")
    base_dir = Path(args.output_dir)
    output_dir = base_dir / "offline" if args.offline else base_dir
    fingerprints = result_fingerprints(corpus, questions, gold, args.offline)
    try:
        preflight_existing_results(output_dir, names, fingerprints)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = AnswerCache(base_dir / ".generator_cache.json")
    features = ["tables", "sections", "lexical", "metadata_filter", "field_boost", "company_balance", "abstain"]
    previous = None
    manifest = []
    for name, config in VARIANTS.items():
        changed = [] if previous is None else [key for key in features if config[key] != VARIANTS[previous][key]]
        if previous is not None and len(changed) != 1:
            raise ValueError(f"{name} changes {len(changed)} components relative to {previous}")
        manifest.append({"variant": name, "relative_to": previous, "changed_components": changed, **config})
        previous = name
    write_json(output_dir / "manifest.json", {
        "baseline": "BeautifulSoup.get_text(); RecursiveCharacterTextSplitter(1500,150); required metadata; BM25Okapi lowercase whitespace tokens; global top-5; no query parsing; never abstains",
        "generator": {"model": hw4.LLM_MODEL, "temperature": 0, "max_tokens": 64, "thinking": False},
        "cache": "Actual generator output keyed by SHA256 of model, settings, exact prompt, question, and formatted context; no index is cached",
        "timing": "Measured wall time includes cache lookups; cache_hits and newly incurred token usage are reported explicitly",
        "source_sha256": fingerprints["B0"]["source_sha256"],
        "runner_sha256": fingerprints["B0"]["runner_sha256"],
        "data_sha256": fingerprints["B0"]["data_sha256"],
        "result_fingerprints": fingerprints,
        "variants": manifest})
    for name in names:
        run_variant(name, corpus, questions, gold, output_dir, args.offline, args.workers,
                    cache, fingerprints[name])
    summaries = []
    for name in VARIANTS:
        path = output_dir / name / "summary.json"
        if path.exists():
            summaries.append(_read_matching_summary(path, name, fingerprints[name]))
    write_json(output_dir / "summary.json", summaries)
    flat = [{key: value for key, value in summary.items() if not isinstance(value, dict)}
            | {f"generator_{key}": value for key, value in summary["generator_usage"].items()}
            for summary in summaries]
    write_csv(output_dir / "summary.csv", flat)


if __name__ == "__main__":
    main()
