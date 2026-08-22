from pathlib import Path

from src.ai.coding_agent import CodingAgent
from src.ai.user_profile_database import (
    UserProfileDatabase,
)
from src.weather.clothing_advisor import (
    ClothingAdvisor,
)


class FakeLLM:
    def generate(
        self,
        prompt: str,
        model: str,
        timeout: float | None = None,
    ) -> str:
        return "The module defines a stored value."


def test_profile_database_stores_personalization_data(
    tmp_path: Path,
):
    database = UserProfileDatabase(
        tmp_path / "profile.db"
    )
    database.save_fact(
        "Josh prefers a hoodie when the weather is cold.",
        category="clothing",
    )
    database.reinforce_interest(
        "computer engineering",
        strength=0.8,
        evidence="Asked a circuit question",
    )
    database.save_style_snapshot(
        {
            "samples": 4,
            "signals": {
                "casual": 0.8,
                "humor": 0.6,
            },
            "slang": {
                "bro": 3,
            },
            "phrases": {
                "step by step": 2,
            },
            "interests": {},
        }
    )
    database.store_code_document(
        "src/example.py",
        "VALUE = 1\n",
        request="Explain the module",
    )

    counts = database.counts()
    assert counts["profile_facts"] == 1
    assert counts["interests"] == 1
    assert counts["style_metrics"] == 2
    assert counts["style_terms"] == 2
    assert counts["code_documents"] == 1

    context = database.get_prompt_context(
        "Help with my engineering code project"
    )
    assert "computer engineering" in context
    assert "src/example.py" in context
    assert "hoodie" in context


def test_memory_actions_update_profile_database(
    tmp_path: Path,
):
    database = UserProfileDatabase(
        tmp_path / "profile.db"
    )
    database.record_memory_action(
        {
            "action": "save_confirmed",
            "note": "Josh prefers detailed engineering explanations.",
            "category": "communication",
            "importance": 0.9,
            "confidence": 1.0,
        }
    )
    database.record_memory_action(
        {
            "action": "reinforce_interest",
            "topic": "robotics",
            "category": "interest",
            "evidence_strength": 0.7,
        },
        source_text="How can I improve my rover?",
    )

    context = database.get_prompt_context(
        "Explain my robotics project"
    )
    assert "detailed engineering explanations" in context
    assert "robotics" in context


def test_clothing_advisor_uses_saved_preferences(
    tmp_path: Path,
):
    database = UserProfileDatabase(
        tmp_path / "profile.db"
    )
    database.save_fact(
        "Josh gets cold easily and prefers a hoodie.",
        category="clothing",
    )
    advisor = ClothingAdvisor(
        profile_database=database
    )

    response = advisor.recommend(
        {
            "name": "Philadelphia",
            "main": {
                "temp": 60,
                "feels_like": 58,
                "humidity": 55,
            },
            "wind": {
                "speed": 5,
            },
            "clouds": {
                "all": 30,
            },
            "weather": [
                {
                    "main": "Clear",
                    "description": "clear sky",
                }
            ],
        }
    )

    assert "hoodie" in response
    assert "saved clothing preferences" in response


def test_coding_agent_saves_code_context(
    tmp_path: Path,
):
    database = UserProfileDatabase(
        tmp_path / "profile.db"
    )
    source = tmp_path / "app.py"
    source.write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    agent = CodingAgent(
        workspace=tmp_path,
        proposal_path=tmp_path / "proposals.json",
        profile_database=database,
    )

    response = agent.analyze_from_prompt(
        "Explain app.py",
        FakeLLM(),
    )

    assert "stored value" in response
    counts = database.counts()
    assert counts["code_documents"] == 1
    assert counts["code_requests"] == 1
