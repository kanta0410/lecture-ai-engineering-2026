"""第5回 宿題(発展): BM25・RRF・ハイブリッド検索のテスト（APIは呼ばない）。"""

import app.retrieval as R


def test_tokenize_mixes_ascii_words_and_japanese_bigrams():
    assert R.tokenize("LoRAと QLoRA：元の重み") == ["lora", "qlora", "と", "元の", "の重", "重み"]


def test_bm25_finds_the_lora_page_for_a_proper_noun_query():
    pages = [h["page_id"] for h in R.bm25_search("LoRA QLoRA 違い", 5)]
    assert pages[0] == "lec02-main-p052"


def test_rrf_prefers_pages_ranked_high_by_both_retrievers():
    fused = R.reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [pid for pid, _ in fused][:2] == ["b", "a"]


def test_search_hybrid_merges_and_reranks(monkeypatch):
    def page(pid):
        return {"page_id": pid, "document_id": "d", "lecture": 2, "document_title": "t", "section": "s",
                "title": pid, "page": 1, "content": "x", "distance": 0.1}

    monkeypatch.setattr(R, "vector_search", lambda q, n: [page("a"), page("b")])
    monkeypatch.setattr(R, "bm25_search", lambda q, n: [page("b"), page("c")])
    monkeypatch.setattr(R, "rerank", lambda q, cands: ["c", "a", "b"])
    plain = R.search_hybrid("q", use_rerank=False)
    assert plain[0]["page_id"] == "b" and plain[0]["vector_rank"] == 2 and plain[0]["bm25_rank"] == 1
    reranked = R.search_hybrid("q", use_rerank=True)
    assert [p["page_id"] for p in reranked] == ["c", "a", "b"]
    assert "distance" not in reranked[0]


def test_pipeline_reranks_graph_candidates_and_adds_neighbors(monkeypatch):
    pages = R.load_pages()
    target = next(p for p in pages if p["page_id"] == "lec02-main-p052")
    hit = dict(target)  # ハイブリッド検索の結果と同じく distance を持たない
    monkeypatch.setattr(R, "search_hybrid", lambda q, use_rerank=False: [hit])
    monkeypatch.setattr(R, "rerank", lambda q, cands: [c["page_id"] for c in cands])
    results = R.search_pipeline("LoRA")
    ids = [(r["page_id"], r["role"]) for r in results]
    assert ids[0] == ("lec02-main-p052", "hit")
    neighbors = {pid for pid, role in ids if role == "neighbor"}
    assert neighbors <= {"lec02-main-p051", "lec02-main-p053"}
    assert len({pid for pid, _ in ids}) == len(ids)  # 重複なし
