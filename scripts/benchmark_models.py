"""2モデル・3温度・各2回の比較測定を行う。"""

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import errors
from google.genai import types

from app.config import get_api_key

QUESTION = "LoRAとは何ですか？"
MODELS = ("gemini-3.6-flash", "gemini-3.5-flash-lite")
TEMPERATURES = (0.1, 0.5, 1.0)
REPETITIONS = 2
PRICES_PER_MILLION = {
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}
OUTPUT_DIR = Path("data")
CSV_PATH = OUTPUT_DIR / "model_benchmark.csv"
MARKDOWN_PATH = OUTPUT_DIR / "model_benchmark.md"


def run_once(client: genai.Client, model: str, temperature: float) -> dict[str, object]:
    started_at = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=QUESTION,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    elapsed_seconds = time.perf_counter() - started_at
    text = response.text or ""
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    input_price, output_price = PRICES_PER_MILLION[model]
    estimated_cost_usd = (
        input_tokens * input_price + output_tokens * output_price
    ) / 1_000_000
    return {
        "status": "ok",
        "error": "",
        "model": model,
        "temperature": temperature,
        "question": QUESTION,
        "answer": text,
        "answer_characters": len(text),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "estimated_cost_usd": round(estimated_cost_usd, 8),
        "free_tier_rate_fraction": "算出不可（プロジェクトごとに上限が異なる）",
    }


def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]]) -> None:
    lines = [
        "# モデル比較結果",
        "",
        f"- 質問: {QUESTION}",
        f"- 実行日時: {datetime.now(timezone.utc).isoformat()}",
        "- レート制限の消費割合: Google AI Studio のプロジェクト別上限値が必要なため未算出",
        "",
        "| モデル | 温度 | 回 | 回答文字数 | 入力トークン | 出力トークン | 合計トークン | 応答時間(秒) | 有料換算(USD) | 無料枠消費割合 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    counts: dict[tuple[str, float], int] = {}
    for row in rows:
        key = (str(row["model"]), float(row["temperature"]))
        counts[key] = counts.get(key, 0) + 1
        lines.append(
            f"| {row['model']} | {row['temperature']} | {counts[key]} | "
            f"{row['answer_characters']} | {row['input_tokens']} | {row['output_tokens']} | "
            f"{row['total_tokens']} | {row['elapsed_seconds']:.3f} | "
            f"{row['estimated_cost_usd']:.8f} | {row['free_tier_rate_fraction']} |"
        )
    lines.extend(["", "## 回答本文", ""])
    answer_counts: dict[tuple[str, float], int] = {}
    for index, row in enumerate(rows, start=1):
        key = (str(row["model"]), float(row["temperature"]))
        answer_counts[key] = answer_counts.get(key, 0) + 1
        lines.extend(
            [
                f"### {index}. {row['model']} / temperature={row['temperature']} / run={answer_counts[key]}",
                "",
                str(row["answer"]),
                "",
            ]
        )
    OUTPUT_DIR.mkdir(exist_ok=True)
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    client = genai.Client(api_key=get_api_key())
    rows = []
    for model in MODELS:
        for temperature in TEMPERATURES:
            for repetition in range(REPETITIONS):
                print(f"[{len(rows) + 1}/12] {model}, temperature={temperature}, run={repetition + 1}")
                try:
                    rows.append(run_once(client, model, temperature))
                except errors.APIError as error:
                    rows.append(
                        {
                            "status": "error",
                            "error": f"{error.code}: {error.message}",
                            "model": model,
                            "temperature": temperature,
                            "question": QUESTION,
                            "answer": "",
                            "answer_characters": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "elapsed_seconds": 0,
                            "estimated_cost_usd": 0,
                            "free_tier_rate_fraction": "算出不可",
                        }
                    )
                    print(f"  ERROR: {error.code} {error.message}")
                write_csv(rows)
                write_markdown(rows)
    print(f"CSV: {CSV_PATH}")
    print(f"Markdown: {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()