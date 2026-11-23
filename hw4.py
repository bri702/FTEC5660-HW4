"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: source-only parsing and lexical retrieval.
# =============================================================================

from bs4 import BeautifulSoup


_COMPANY_NAMES: dict[str, str] = {}
_BROKER_NAMES: set[str] = set()


def _html_text(node) -> str:
    """Keep printed cell values; normalise only HTML layout whitespace."""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _table_lines(table_html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return "", []
    caption_tag = table.find("caption", recursive=False)
    caption = _html_text(caption_tag) if caption_tag else ""
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    header_rows = [row for row in rows if row.find_parent("thead") is not None]
    if not header_rows and rows:
        cells = rows[0].find_all(["th", "td"], recursive=False)
        if cells and all(c.name == "th" and c.get("scope") != "row" for c in cells):
            header_rows = [rows[0]]
    # Expand multi-level column headers without changing their printed labels.
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(header_rows):
        c = 0
        for cell in row.find_all(["th", "td"], recursive=False):
            while (r, c) in grid:
                c += 1
            label = _html_text(cell)
            rowspan = max(1, int(cell.get("rowspan", 1)))
            colspan = max(1, int(cell.get("colspan", 1)))
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    grid[rr, cc] = label
            c += colspan
    width = max((c for _, c in grid), default=-1) + 1
    headers = []
    for c in range(width):
        labels = list(dict.fromkeys(grid.get((r, c), "") for r in range(len(header_rows))))
        headers.append(" / ".join(label for label in labels if label))
    lines = []
    for row in rows:
        if any(row is header for header in header_rows):
            continue
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        values = [_html_text(cell) for cell in cells]
        if len(cells) == 1:
            # A spanning section heading is still useful context, but not a value row.
            if values[0]:
                lines.append(values[0])
            continue
        row_label = values[0]
        pieces = []
        column = max(1, int(cells[0].get("colspan", 1)))
        for cell, value in zip(cells[1:], values[1:]):
            label = headers[column] if column < len(headers) else ""
            pieces.append(f"{label}: {value}" if label else value)
            column += max(1, int(cell.get("colspan", 1)))
        line = " | ".join([row_label, *pieces])
        if len(line) <= 400:
            lines.append(line)
        else:
            # Repeat the row label if a wide row would exceed the grading limit.
            lines.extend(f"{row_label} | {piece}" for piece in pieces)
    return caption, lines





def _metadata_from_soup(file_name: str, soup) -> dict:
    title = soup.find("h1")
    if title is None:
        raise ValueError(f"Document {file_name} has no company heading")
    company = re.sub(r"\s*\([^()]*\)\s*$", "", _html_text(title)).strip()
    headings = " ".join(_html_text(tag) for tag in soup.find_all(["h2", "h3"], limit=4))
    subtitle = soup.select_one(".subtitle")
    title_text = _html_text(soup.title) if soup.title else ""
    cover_text = " ".join([_html_text(subtitle) if subtitle else "", headings, title_text])
    if re.search(r"earnings\s+(?:conference\s+)?call", cover_text, re.I):
        doc_type = "earnings_call"
        year_match = re.search(r"(?:full\s+year\s*|FY\s*)(20\d{2})", cover_text, re.I)
    elif re.search(r"results\s+review", cover_text, re.I):
        doc_type = "research_report"
        year_match = re.search(r"FY\s*(20\d{2})", cover_text, re.I)
    else:
        doc_type = "annual_report"
        year_match = re.search(r"annual\s+report\s*(20\d{2})", cover_text, re.I)
    if year_match is None:
        raise ValueError(f"Cannot identify fiscal year in the cover of {file_name}")
    metadata = {"source": file_name, "company": company,
                "fiscal_year": int(year_match.group(1)), "doc_type": doc_type}
    broker_tag = soup.select_one('.broker')
    if doc_type == 'research_report' and broker_tag is not None:
        metadata['broker'] = re.split(r'[|—]', _html_text(broker_tag))[0].strip()
    return metadata


def _document_metadata(file_name: str, html: str) -> dict:
    """Extract the document's own company and reporting year, never an issue date."""
    return _metadata_from_soup(file_name, BeautifulSoup(html, "html.parser"))


def _split_long_text(text: str, budget: int) -> list[str]:
    """Split at whitespace with a small overlap so boundary names survive."""
    if len(text) <= budget:
        return [text] if text else []
    parts = []
    start = 0
    while start < len(text):
        end = min(start + budget, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + budget // 2, end)
            if boundary > start:
                end = boundary
        parts.append(text[start:end].strip())
        if end == len(text):
            break
        overlap = text.rfind(" ", max(start + 1, end - 100), end)
        start = overlap + 1 if overlap > start else end
    return parts

from functools import lru_cache
from rank_bm25 import BM25Okapi


# Generic linguistic/financial vocabulary; no corpus-specific entities or answers.
_QUERY_STOP = set('''a an the what which who how much many was were is are be been
    did does do had has have held hold at on in of for from to by with and or than
    its their it this that as end ended year fiscal fy december usd million millions
    billion billions percent percentage value company companies higher more less
    report reported reports generate generated earn earned spend spent give gives
    set shares share fourth quarter full q4 call earnings
    announce announced programme program approximately about revenuegrowth
    investment review results forecast forecasts forecasted management guide guided
    twelve month 12'''.split())


def _word_root(word: str) -> str:
    aliases = {
        'headquartered': 'headquarter', 'headquarters': 'headquarter',
        'auditor': 'audit', 'auditors': 'audit', 'audited': 'audit',
        'auditing': 'audit', 'borrowing': 'borrow', 'borrowings': 'borrow',
        'repurchases': 'repurchase', 'buyback': 'repurchase', 'buybacks': 'repurchase',
        'headcount': 'employee', 'employed': 'employee', 'staff': 'employee',
        'cfo': 'cfo', 'ceo': 'ceo', 'programme': 'program',
    }
    if word in aliases:
        return aliases[word]
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 4 and word.endswith('s') and not word.endswith(('ss', 'us')):
        return word[:-1]
    return word


def _tokens(text: str, mode: str = 'lexical') -> list[str]:
    if mode == 'whitespace':
        return text.lower().split()
    words = re.findall(r"[a-z]+|\d+(?:[,.]\d+)*", unicodedata.normalize('NFKC', text).lower())
    return [_word_root(w) for w in words if w not in _QUERY_STOP]


def _clean_entity(name: str) -> str:
    # A comparison may put its shared date or unit after the final company.
    name = re.split(
        r'\s+(?:at|on|as\s+of|in|for|during)\s+(?=(?:FY\s*)?(?:19|20)\d{2}\b|'
        r'\d{1,2}\s+[A-Za-z]+\s+(?:19|20)\d{2}\b|(?:the\s+)?(?:end|year[- ]end)\b)',
        name, maxsplit=1, flags=re.I)[0]
    name = re.sub(r',?\s+in\s+(?:USD|US\s+dollars?|percent|millions?|billions?)\b.*$', '', name, flags=re.I)
    name = re.sub(r"^[\s:;,]+|[\s:;,?.]+$", '', name)
    name = re.sub(r'^(?:In|On|For|From|By)\s+', '', name)
    name = re.sub(r"['’]s$|['’]$", '', name)
    return name.strip()


@lru_cache(maxsize=4096)
def _parse_question_cached(question: str, registry: tuple, brokers: tuple) -> tuple:
    year_match = re.search(r'\b(?:FY\s*)?((?:19|20)\d{2})(?:[AE])?\b', question, re.I)
    year = int(year_match.group(1)) if year_match else None
    lower = question.lower()
    if re.search(r'earnings\s+(?:conference\s+)?call|fourth[- ]quarter.*call|\bq4\b.*call', lower):
        doc_type = 'earnings_call'
    elif re.search(r'\brating\b|target\s+price|\b(?:forecast(?:s|ed|ing)?|project(?:s|ed|ing)?|projections?|estimat(?:e|es|ed|ing))\b|results\s+review|research\s+(?:report|note)|broker', lower):
        doc_type = 'research_report'
    else:
        doc_type = 'annual_report'

    entities = []
    # Comparison subjects are explicit. This also retains unseen companies.
    if ':' in question and re.search(r'\bwhich company\b', question, re.I):
        entities = re.split(r'\s+(?:or|versus|vs\.?)\s+', question.split(':', 1)[1].rstrip('?'), flags=re.I)
    if not entities:
        # A possessive is an especially strong boundary for a company name.
        name_pattern = r"[A-Z][A-Za-z0-9&.-]*(?:\s+(?:[A-Z][A-Za-z0-9&.-]*|and|of|the)){0,7}"
        entities = re.findall(rf"({name_pattern})(?:['’]s\b|['’](?=\s))", question)
        entities = [e for e in entities if company_key(_clean_entity(e)) not in brokers]
    if not entities:
        match = re.search(r'\bdid\s+(.+?)\s+(?:hold|report|generate|earn|spend|announce|have|pay)\b', question, re.I)
        if match:
            entities = [match.group(1)]
    if not entities:
        # Generic syntactic fallbacks when the company is absent from the corpus.
        patterns = [
            r'\b(?:audited|audits)\s+(.+?)(?:[\'’]|\s+financial\s+statements)',
            r'\b(?:is|was)\s+(.+?)\s+headquartered\b',
            r'\b(?:forecast(?:s|ed)?|project(?:s|ed)?|estimat(?:e|es|ed))\s+for\s+(.+?)(?:,|\?|\s+in\s+FY)',
            r'\bsets?\s+for\s+(.+?)(?=\s+in\s+its\b|\s+shares\b|[,?]|$)',
            r'\bgive\s+(.+?)\s+in\s+its\b',
            r'\bof\s+(.+?)\s+at\s+the\s+end\b',
            r'\b(?:earned|generated|reported|held|announced|recorded|disclosed)\s+by\s+(.+?)(?=\s+(?:in|for|during|at|as)\b|[,?]|$)',
            r'\b(?:assigned|given|awarded)\s+to\s+(.+?)(?=\s+(?:in|for|during|at)\b|[,?]|$)',
            r'\b(?:of|for)\s+(.+?)\s+(?:in|during|for)\s+(?:FY\s*\d{4}|fiscal)',
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.I)
            if match:
                entities = [match.group(1)]
                break
    if not entities:
        # Reject a known-name prefix inside a longer, unseen company name.
        for key, name in sorted(registry, key=lambda item: -len(item[0])):
            pattern = r'(?<!\w)' + r'\W+'.join(map(re.escape, key.split())) + r'(?!\w)'
            for match in re.finditer(pattern, question, re.I):
                tail = question[match.end():]
                extra = re.match(r'\s+([A-Z][A-Za-z.-]*)', tail)
                if extra and company_key(extra.group(1)):
                    continue
                entities.append(name)
                break
    result = []
    for entity in entities:
        entity = _clean_entity(entity)
        if entity and company_key(entity) not in brokers and company_key(entity) not in {company_key(x) for x in result}:
            result.append(entity)
    return tuple(result), year, doc_type





def _query_profile(question: str, parsed: dict) -> dict:
    q = question.lower()
    # Each alternative describes a financial concept, not a particular question.
    rules = [
        (r'accounts?\s+receivable|receivables?', 'receivables', [('account', 'receivable', 'net')]),
        (r'long[- ]term\s+borrow', 'borrowing', [('long', 'term', 'borrow')]),
        (r'(?:net\s+)?cash\s+from\s+operating|operating\s+cash\s+flow', 'operating_cash', [('cash', 'operating', 'activity'), ('operating', 'cash', 'flow')]),
        (r'cash\s+and\s+cash\s+equivalents?', 'cash', [('cash', 'equivalent')]),
        (r'operating\s+income', 'operating_income', [('operating', 'income')]),
        (r'net\s+income', 'net_income', [('net', 'income')]),
        (r'gross\s+profit', 'gross_profit', [('gross', 'profit')]),
        (r'total\s+assets?', 'assets', [('total', 'asset')]),
        (r'total\s+equity', 'equity', [('total', 'equity')]),
        (r'inventor(?:y|ies)', 'inventory', [('inventory',)]),
        (r'employees?|headcount', 'employees', [('employee',)]),
        (r'chief\s+financial\s+officer|\bcfo\b', 'cfo', [('chief', 'financial', 'officer'), ('cfo',)]),
        (r'chief\s+executive\s+officer|\bceo\b', 'ceo', [('chief', 'executive', 'officer'), ('ceo',)]),
        (r'audit\s+firm|auditor|audited', 'auditor', [('audit',)]),
        (r'headquarter', 'headquarters', [('headquarter',)]),
        (r'target\s+price', 'target_price', [('target', 'price')]),
        (r'\brating\b', 'rating', [('rating',)]),
        (r'share\s+repurchase|buyback', 'repurchase', [('repurchase',)]),
        (r'revenue\s+growth', 'revenue_growth', [('revenue', 'growth')]),
        (r'\brevenue\b', 'revenue', [('revenue',)]),
    ]
    kind, alternatives = 'other', []
    for pattern, label, terms in rules:
        if re.search(pattern, q):
            kind, alternatives = label, [set(x) for x in terms]
            break
    focus = question
    for company in parsed['companies']:
        focus = re.sub(re.escape(company), ' ', focus, flags=re.I)
    if not alternatives:
        alternatives = [{t for t in _tokens(focus)
                         if t not in {'financial', 'statement', 's'} and not re.fullmatch(r'\d+(?:[,.]\d+)*', t)}]

    # Extract a revenue dimension independently of where the date occurs.
    # Remove only question framing; original evidence text is never rewritten.
    dimension_question = re.sub(
        r',?\s*\bin\s+(?:USD\b|US\s+dollars?\b|percent(?:age)?\b|millions?\b|billions?\b).*$',
        '', question, flags=re.I)
    dimension_question = re.sub(
        r'\b(?:at|on|as\s+of)\s+\d{1,2}\s+[A-Za-z]+\s+(?:19|20)\d{2}\b',
        ' ', dimension_question, flags=re.I)
    dimension_question = re.sub(
        r'\b(?:(?:in|for|during)\s+)?FY\s*(?:19|20)\d{2}[AE]?\b|'
        r'\b(?:in|for|during)\s+(?:fiscal(?:\s+year)?\s+)?(?:19|20)\d{2}\b',
        ' ', dimension_question, flags=re.I)
    qualifiers = []
    segment = re.search(r"['’]s\s+(.+?)\s+segment\b", dimension_question, re.I)
    if segment:
        qualifiers = _tokens(segment.group(1))
    elif kind in {'revenue', 'revenue_growth'}:
        region = re.search(
            r'\b(?:in|from|attributable\s+to)\s+(?:(?:its|the)\s+)?(.+?)'
            r'(?=\s+(?:according\s+to|as\s+reported)\b|[?]|$)',
            dimension_question, re.I)
        if region:
            candidate = set(_tokens(region.group(1).strip(' ,.;')))
            framing = {'annual', 'report', 'financial', 'statement', 'result', 'review',
                       'research', 'earning', 'call', 'conference', 'transcript', 'note'}
            if not candidate <= framing:
                qualifiers = list(candidate)
    return {'kind': kind, 'alternatives': alternatives, 'qualifiers': set(qualifiers)}


def _build_index(chunks: list[Document], token_mode: str = 'lexical') -> dict:
    """The same index builder is used by the submitted pipeline and its ablations."""
    texts = [c.page_content for c in chunks]
    tokenized = [_tokens(text, token_mode) or ['_empty_'] for text in texts]
    doc_years = {}
    for c in chunks:
        source = c.metadata.get('source')
        doc_years.setdefault(source, set()).update(map(int, re.findall(r'\b((?:19|20)\d{2})\b', c.page_content)))
        if isinstance(c.metadata.get('fiscal_year'), int):
            doc_years[source].add(c.metadata['fiscal_year'])
    return {'chunks': chunks, 'bm25': BM25Okapi(tokenized, k1=1.5, b=0.75) if chunks else None,
            'token_mode': token_mode, 'doc_years': doc_years,
            'line_tokens': [[set(_tokens(line)) for line in text.splitlines()] for text in texts],
            'companies': {company_key(c.metadata.get('company', '')) for c in chunks}}





def _profile_match(index: dict, i: int, profile: dict) -> tuple[float, bool]:
    chunk = index['chunks'][i]
    lines = chunk.page_content.splitlines()
    token_lines = index['line_tokens'][i]
    alternatives, qualifiers = profile['alternatives'], profile['qualifiers']
    best = 0.0
    supported = False
    # Table captions provide context for rows such as a segment name with revenue values.
    context = set(_tokens(chunk.metadata.get('section', '')))
    if chunk.metadata.get('kind') == 'table':
        context |= set(_tokens(' '.join(lines[:3])))
    for line, tokens in zip(lines, token_lines):
        if not tokens:
            continue
        coverage = max((len(terms & tokens) / len(terms) for terms in alternatives if terms), default=0.0)
        if qualifiers:
            qualifier_coverage = len(qualifiers & tokens) / len(qualifiers)
            context_coverage = max((len(terms & (tokens | context)) / len(terms) for terms in alternatives if terms), default=0.0)
            coverage = min(context_coverage, qualifier_coverage)
        negated = bool(re.search(r'not\s+(?:separately\s+)?(?:disclosed|reported|provided|available)|does\s+not\s+disclose', line, re.I))
        best = max(best, coverage)
        numeric_kind = profile['kind'] not in {'cfo', 'ceo', 'auditor', 'headquarters', 'rating'}
        without_years = re.sub(r'\b(?:FY\s*)?(?:19|20)\d{2}[AE]?\b', ' ', line, flags=re.I)
        has_value = bool(re.search(r'\d', without_years) or re.search(r':\s*\(?[-−]?\d', line))
        if coverage >= 0.99 and not negated and (not numeric_kind or has_value):
            supported = True
    return best, supported


def _metadata_allows(chunk: Document, parsed: dict, profile: dict, index: dict) -> bool:
    meta = chunk.metadata
    companies = {company_key(c) for c in parsed['companies']}
    if companies and company_key(meta.get('company', '')) not in companies:
        return False
    if meta.get('doc_type') != parsed['doc_type']:
        return False
    year, doc_year = parsed['fiscal_year'], meta.get('fiscal_year')
    if year is None:
        return True
    if parsed['doc_type'] == 'annual_report':
        if profile['kind'] in {'cfo', 'ceo', 'auditor'}:
            return doc_year == year
        # Annual reports present the current and previous year; dates in risk
        # paragraphs must not make unrelated reports eligible.
        return doc_year in (year, year + 1)
    if profile['kind'] in {'rating', 'target_price', 'repurchase'}:
        return doc_year == year
    return year in index['doc_years'].get(meta.get('source'), set())


def _retrieve_config(index: Any, question: str, *, metadata_filter: bool = True,
                     company_balance: bool = True, abstain: bool = True,
                     field_boost: bool = True) -> list[Document]:
    if not index or not index['chunks']:
        return []
    parsed = parse_query(question)
    profile = _query_profile(question, parsed)
    chunks = index['chunks']
    raw = index['bm25'].get_scores(_tokens(question, index['token_mode']))
    scale = max(float(max(raw)), 1.0)
    ranked = []
    matches = {}
    for i, chunk in enumerate(chunks):
        if metadata_filter and not _metadata_allows(chunk, parsed, profile, index):
            continue
        match, supported = _profile_match(index, i, profile)
        matches[i] = supported
        score = float(raw[i]) / scale
        if field_boost:
            score += 12.0 * match
            if supported:
                score += 4.0
            meta = chunk.metadata
            if meta.get('fiscal_year') == parsed['fiscal_year']:
                score += 0.4
            if profile['kind'] in {'cfo', 'ceo'}:
                if 'executive' in meta.get('table_class', ''):
                    score += 2.0
            elif profile['kind'] not in {'auditor', 'headquarters', 'rating', 'target_price', 'repurchase', 'revenue_growth', 'other'}:
                if meta.get('kind') == 'table':
                    score += 1.5
            if profile['kind'] == 'rating' and re.search(r'\bRatings:', chunk.page_content):
                score -= 2.0
        ranked.append((score, i))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if not ranked:
        return []
    if abstain:
        # Every requested company needs actual evidence, rather than merely a
        # high BM25 score. In particular, a missing subject is not an alias.
        subjects = {company_key(c) for c in parsed['companies']}
        if not subjects:
            return []
        if subjects and not subjects <= index['companies']:
            return []
        supported_subjects = {company_key(chunks[i].metadata.get('company', ''))
                              for _, i in ranked if matches[i]}
        if (subjects and not subjects <= supported_subjects) or not any(matches[i] for _, i in ranked):
            return []
    if company_balance:
        subjects = [company_key(c) for c in parsed['companies']]
        if len(subjects) > 1:
            selected = []
            for subject in subjects:
                candidates = [i for _, i in ranked if company_key(chunks[i].metadata.get('company', '')) == subject]
                if candidates:
                    selected.append(candidates[0])
            return [chunks[i] for i in selected[:MAX_RETRIEVED]]
        # Three focused passages give factual questions enough context without
        # filling the prompt with additional, conflicting documents.
        return [chunks[i] for _, i in ranked[:min(3, MAX_RETRIEVED)]]
    return [chunks[i] for _, i in ranked[:MAX_RETRIEVED]]


def table_to_text(table_html: str) -> str:
    """Represent table rows with their original column labels, values and caption."""
    caption, lines = _table_lines(table_html)
    return "\n".join(([caption] if caption else []) + lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Preserve section context and complete table rows in small source-only chunks."""
    soup = BeautifulSoup(html, "html.parser")
    metadata = _metadata_from_soup(file_name, soup)
    _COMPANY_NAMES[company_key(metadata["company"])] = metadata["company"]
    if metadata.get('broker'):
        _BROKER_NAMES.add(company_key(metadata['broker']))
    for tag in soup.find_all(["nav", "script", "style", "head"]):
        tag.decompose()
    body = soup.body or soup
    chunks = []
    section = ""
    subsection = ""
    pending: list[str] = []

    def header() -> str:
        context = " | ".join(part for part in [section, subsection] if part)
        broker = f" | broker: {metadata['broker']}" if metadata.get('broker') else ''
        return (f"{metadata['company']} | FY{metadata['fiscal_year']} | {metadata['doc_type']}{broker}\n"
                + (context + "\n" if context else ""))

    def emit(lines: list[str], kind: str, table_class: str = "", caption: str = "") -> None:
        if not lines:
            return
        prefix = header() + (caption + "\n" if caption else "")
        # This budget includes both the metadata/section header and repeated caption.
        if len(prefix) > 700:
            prefix = prefix[:700].rstrip() + "\n"
        budget = min(1150, MAX_CHUNK_CHARS - len(prefix))
        packed = ""
        for line in lines:
            for piece in _split_long_text(line, budget):
                candidate = (packed + "\n" + piece).strip()
                if packed and len(candidate) > budget:
                    chunks.append(Document(page_content=prefix + packed, metadata={
                        **metadata, "section": " | ".join(x for x in [section, subsection] if x),
                        "kind": kind, "table_class": table_class}))
                    packed = piece
                else:
                    packed = candidate
        if packed:
            chunks.append(Document(page_content=prefix + packed, metadata={
                **metadata, "section": " | ".join(x for x in [section, subsection] if x),
                "kind": kind, "table_class": table_class}))

    def flush() -> None:
        nonlocal pending
        emit(pending, "text")
        pending = []

    for node in body.find_all(["h1", "h2", "h3", "p", "li", "table"]):
        if node.find_parent("table") is not None:
            continue
        # Avoid reading a nested paragraph/list item twice.
        if node.name in {"p", "li"} and node.find_parent(["p", "li"]) is not None:
            continue
        if node.name in {"h1", "h2", "h3"}:
            flush()
            if node.name == "h2":
                section, subsection = _html_text(node), ""
            elif node.name == "h3":
                subsection = _html_text(node)
            # H1 is preserved as the company in every short chunk header.
        elif node.name == "table":
            flush()
            lines = table_to_text(str(node)).splitlines()
            caption_tag = node.find('caption', recursive=False)
            caption = _html_text(caption_tag) if caption_tag else ''
            if caption and lines and lines[0] == caption:
                lines = lines[1:]
            emit(lines, "table", " ".join(node.get("class", [])), caption)
        else:
            text = _html_text(node)
            if text:
                pending.append(text)
    flush()
    return chunks


def parse_query(question: str) -> dict:
    """Parse company subjects, the question's year, and the requested source type."""
    companies, year, doc_type = _parse_question_cached(question, tuple(sorted(_COMPANY_NAMES.items())), tuple(sorted(_BROKER_NAMES)))
    return {'companies': list(companies), 'fiscal_year': year, 'doc_type': doc_type}


def build_index(chunks: list[Document]) -> Any:
    return _build_index(chunks, token_mode='lexical')


def retrieve(index: Any, question: str) -> list[Document]:
    return _retrieve_config(index, question)


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
