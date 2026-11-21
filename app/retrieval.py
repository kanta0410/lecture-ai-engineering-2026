"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from collections import defaultdict, deque
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
    HYBRID_CANDIDATES, HYBRID_TOP_K, HYBRID_RRF_K, HYBRID_RERANK, MODEL,
    PIPELINE_TOP_K, PIPELINE_NEIGHBOR_TOP,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def vector_search(query: str, limit: int) -> list[dict]:
    """質問を埋め込み、近いページを最大limit件返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), limit))
        return [dict(row) for row in rows]


def search(query: str) -> list[dict]:
    """質問を埋め込み、近いページを最大RAG_TOP_K件返す。"""
    return vector_search(query, RAG_TOP_K)


# ---- 発展: BM25 + ベクトル検索のハイブリッド + リランク ----

def tokenize(text: str) -> list[str]:
    """BM25用の分かち書き。英数字は単語、日本語などは文字2-gramにする（辞書不要）。"""
    text = text.lower()
    words = re.findall(r"[a-z0-9][a-z0-9.\-_]*", text)
    tokens = list(words)
    for run in re.findall(r"[^\x00-\x7f\s]+", text):
        run = re.sub(r"[、。，．・！？「」（）()【】『』：；\u3000]", "", run)
        if len(run) == 1:
            tokens.append(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


@lru_cache(maxsize=1)
def bm25_index() -> dict:
    """対象ページ全体のBM25用の統計（出現回数・文書頻度・長さ）を作る。"""
    pages = load_pages(RAG_SOURCE)
    docs = []
    df = defaultdict(int)
    for page in pages:
        counts = defaultdict(int)
        for token in tokenize(page["title"] + "\n" + page["content"]):
            counts[token] += 1
        docs.append(counts)
        for token in counts:
            df[token] += 1
    lengths = [sum(c.values()) for c in docs]
    return {"pages": pages, "docs": docs, "df": df, "lengths": lengths,
            "avg": sum(lengths) / max(len(lengths), 1)}


def bm25_search(query: str, limit: int, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    """BM25（TF-IDFを改良したキーワード検索）で上位limit件を返す。"""
    index = bm25_index()
    n = len(index["pages"])
    terms = set(tokenize(query))
    scored = []
    for page, counts, length in zip(index["pages"], index["docs"], index["lengths"]):
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = index["df"][term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / index["avg"]))
        if score > 0:
            scored.append((score, page["page_id"], page))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{**page, "bm25_score": round(score, 3)} for score, _, page in scored[:limit]]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """複数の順位表を、順位だけで合成する（スコアの尺度が違っても使える）。"""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, page_id in enumerate(ranking, 1):
            fused[page_id] += 1 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


# リランクで使ったトークン数。[usage]には含まれないため、比較時はここを足す。
RERANK_USAGE = {"input": 0, "output": 0, "calls": 0}


def rerank(query: str, candidates: list[dict]) -> list[str]:
    """LLMに質問との関連が強い順にページIDを並べ替えさせる。失敗時は元の順序を返す。"""
    lines = "\n".join(
        f"{c['page_id']} | {c['title']} | {c['content'].replace(chr(10), ' ')[:300]}" for c in candidates)
    prompt = (
        "質問に答える根拠として役立つ順に、候補ページのIDを並べ替えてください。"
        "用語が似ているだけで答えの根拠にならないページは後ろにしてください。\n"
        f"質問: {query}\n候補（ID | タイトル | 冒頭）:\n{lines}"
    )
    try:
        with genai.Client(api_key=get_api_key()) as client:
            response = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={"type": "object", "properties": {
                        "ranked_ids": {"type": "array", "items": {"type": "string"}}},
                        "required": ["ranked_ids"]},
                    temperature=0,
                ),
            )
        usage = response.usage_metadata
        RERANK_USAGE["calls"] += 1
        RERANK_USAGE["input"] += usage.prompt_token_count or 0
        RERANK_USAGE["output"] += usage.candidates_token_count or 0
        ranked = json.loads(response.text)["ranked_ids"]
    except Exception:
        return [c["page_id"] for c in candidates]
    valid = {c["page_id"] for c in candidates}
    ordered = [pid for pid in dict.fromkeys(ranked) if pid in valid]
    return ordered + [c["page_id"] for c in candidates if c["page_id"] not in ordered]


def search_hybrid(query: str, use_rerank: bool | None = None) -> list[dict]:
    """ベクトル検索とBM25の結果をRRFで合成し、（任意で）LLMで並べ替えて上位HYBRID_TOP_K件を返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    use_rerank = HYBRID_RERANK if use_rerank is None else use_rerank
    vec = vector_search(query, HYBRID_CANDIDATES)
    kw = bm25_search(query, HYBRID_CANDIDATES)
    vec_rank = {h["page_id"]: i for i, h in enumerate(vec, 1)}
    kw_rank = {h["page_id"]: i for i, h in enumerate(kw, 1)}
    pages = {h["page_id"]: h for h in kw}
    pages.update({h["page_id"]: h for h in vec})
    fused = reciprocal_rank_fusion([[h["page_id"] for h in vec], [h["page_id"] for h in kw]], HYBRID_RRF_K)
    order = [pid for pid, _ in fused]
    if use_rerank:
        order = rerank(query, [pages[pid] for pid in order])
    rrf = dict(fused)
    results = []
    for pid in order[:HYBRID_TOP_K]:
        page = {key: value for key, value in pages[pid].items() if key not in ("distance", "bm25_score")}
        page.update({"vector_rank": vec_rank.get(pid), "bm25_rank": kw_rank.get(pid), "rrf": round(rrf[pid], 4)})
        results.append(page)
    return results


def search_graph(query: str, seeds_override: list[dict] | None = None) -> dict:
    """ベクトル検索（またはseeds_overrideで渡した検索結果）を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    found = seeds_override if seeds_override is not None else search(query)
    seeds = [hit for hit in found if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed.get("distance"), "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}


def search_pipeline(query: str) -> list[dict]:
    """最終構成: ハイブリッドの上位を入口にGraphで広げ、LLMで並べ替え、上位の前後ページも付ける。

    1. ベクトル検索+BM25を合成した上位GRAPH_START_K件を入口にする
    2. Graphで前後ページ・共通用語をたどり、候補を広げる
    3. 候補をLLMで並べ替え、上位PIPELINE_TOP_K件を残す
    4. 上位PIPELINE_NEIGHBOR_TOP件について、同じ資料内の前後1ページを追加する（重複なし）
    """
    seeds = search_hybrid(query, use_rerank=False)[:GRAPH_START_K]
    graph = search_graph(query, seeds_override=seeds)
    candidates = graph["seeds"] + graph["expanded"]
    by_id = {c["page_id"]: c for c in candidates}
    order = rerank(query, candidates)[:PIPELINE_TOP_K]
    pages = load_pages(RAG_SOURCE)
    position = {p["page_id"]: i for i, p in enumerate(pages)}
    results = [{**{k: v for k, v in by_id[pid].items() if k not in ("distance", "path", "hops")},
                "role": "hit"} for pid in order]
    seen = set(order)
    for pid in order[:PIPELINE_NEIGHBOR_TOP]:
        i = position[pid]
        for j in (i - 1, i + 1):
            if 0 <= j < len(pages) and pages[j]["document_id"] == pages[i]["document_id"] \
                    and pages[j]["page_id"] not in seen:
                seen.add(pages[j]["page_id"])
                results.append({**pages[j], "role": "neighbor"})
    return results



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    parser.add_argument("--method", choices=["vector", "bm25", "hybrid", "rerank"], default="vector",
                        help="vector=ベクトル検索 / bm25=キーワード検索 / hybrid=両方をRRFで合成 / rerank=hybrid+LLM並べ替え")
    args = parser.parse_args()
    found = {"vector": lambda: search(args.query),
             "bm25": lambda: bm25_search(args.query, RAG_TOP_K),
             "hybrid": lambda: search_hybrid(args.query, use_rerank=False),
             "rerank": lambda: search_hybrid(args.query, use_rerank=True)}[args.method]()
    for i, result in enumerate(found, 1):
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"\n{result['content']}")
