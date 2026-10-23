"""第1回: 会話履歴を保ちながらLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 既定の質問後、続けて質問できる

ツールは持たないが、同じチャット内では会話履歴を保持する。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import os
import time

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
TEMPERATURE = float(os.getenv("AIE_TEMPERATURE", "0.5"))


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    elapsed_seconds = time.perf_counter() - started_at

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）
    usage = response.usage_metadata
    print(f"[usage] input_tokens={usage.prompt_token_count} output_tokens={usage.candidates_token_count}")
    print(f"[timing] elapsed_seconds={elapsed_seconds:.3f}")

    return response.text


def ask_in_chat(chat, question: str) -> str:
    """チャットに質問し、履歴込みの利用量と応答時間を表示する。"""
    started_at = time.perf_counter()
    response = chat.send_message(question)
    elapsed_seconds = time.perf_counter() - started_at
    usage = response.usage_metadata
    print(f"[usage] input_tokens={usage.prompt_token_count} output_tokens={usage.candidates_token_count}")
    print(f"[timing] elapsed_seconds={elapsed_seconds:.3f}")
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(temperature=TEMPERATURE),
        )
        print(ask_in_chat(chat, args.question))
        print("\n続けて質問できます。終了するには exit または Ctrl-D を入力してください。")
        while True:
            try:
                question = input("\nあなた: ")
            except EOFError:
                print()
                break
            if question.strip().lower() in {"exit", "quit"}:
                break
            if not question.strip():
                continue
            print(f"\n[質問] {question}\n")
            print(ask_in_chat(chat, question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
