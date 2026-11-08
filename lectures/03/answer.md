# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
  - 素のLLMはモデルの知識だけで回答する。
  - エージェントは、必要に応じて用語集や講義資料をツールで検索し、その結果を根拠として回答する。
- `--trace` で見えたツール呼び出し（ツール名と引数）：
  - `search_glossary(query="LoRA")`
  - 質問によっては `search_glossary(query="QLoRA")` も呼ばれる。
- 1つの質問での LLM 呼び出し回数（`requests`）：
  - ツールを選ぶ呼び出しと、ツールの結果から最終回答を作る呼び出しのため、基本的に2回。
  - 実際の値は実行時の `--trace` に表示される `[usage]` を確認する。
- 追加したツールと、docstring に書いた説明：
  - `make_quiz`
  - 指定した回の用語から1つ選び、復習用の確認問題と答えを返すツール。
  - `app/tools.py` に実装し、`app/agent.py` に登録した。

## 宿題

- 取り組んだレベル：基礎、実装
- やったこと：
  - Instructions やツールのdocstringを変更し、ツール選択と回答への影響を比較する。
  - `save_learning_log` ツールを追加し、学習内容を `learning_log/` に保存する。
  - `app/agent.py` で `requires_approval=True` を付けて登録する。
- 結果：
  - `search_glossary` のdocstringだけを変更した実験では、ツール呼び出しは変わらなかった。
  - `INSTRUCTIONS` で用語集を使わないよう明示した実験では、ツールが呼ばれず `requests=1` になった。
  - Instructionsはエージェント全体への指示としてモデルに渡り、docstringはツールの説明として渡るが、docstringの変更が必ず挙動を変えるわけではない。
- 気づき・疑問：
  - ツールの説明は、ツールの機能だけでなく「いつ使うか」を具体的に書くことが重要。
  - 承認が必要なツールは一度 `DeferredToolRequests` として停止し、利用者の判断を渡してから処理を再開する。

### 発展A：Google ADK

- Google ADK（Agent Development Kit）は、モデル・Instructions・ツールを登録してエージェントを構築するフレームワーク。
- `app/adk_agent/agent.py` に、既存のInstructionsと3つのツールをADKの `root_agent` として登録した。
- Pydantic AI版は `Agent(instructions=...)` と `tool_plain(...)`、ADK版は `Agent(instruction=..., tools=[...])` を使う。
- 実行・回答比較は、依存関係を同期した後に `adk web app` または `adk run app.adk_agent` で行う。

### 発展B：Piとの比較

- Piは、コード編集用のツールと実行ループを持つコーディングエージェント。
- 公開されている `system-prompt.ts` では、基本指示、ツール一覧、ルール、ドキュメント、スキル、作業ディレクトリなどを実行時に組み立てている。
- このエージェントの `INSTRUCTIONS` は固定のシステム指示だが、Piは環境や利用可能なツールに応じてシステムプロンプトを構成する。
- 共通点は、モデルに指示とツールを渡し、モデルのツール呼び出しと結果をループで処理する点である。
