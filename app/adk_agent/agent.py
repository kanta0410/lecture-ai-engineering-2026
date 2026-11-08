"""Google ADK版の学習支援エージェント。

Pydantic AI版と同じInstructionsとツールを、ADKのAgentに登録する。
"""

from google.adk.agents import Agent

from app.agent import INSTRUCTIONS
from app.tools import get_lecture_summary, make_quiz, search_glossary

root_agent = Agent(
    name="aie_learning_agent",
    model="gemini-2.0-flash",
    instruction=INSTRUCTIONS,
    tools=[search_glossary, get_lecture_summary, make_quiz],
)
