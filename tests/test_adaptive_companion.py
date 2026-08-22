from pathlib import Path

import pytest

from src.ai.coding_agent import ApprovalRequiredError, CodingAgent
from src.ai.user_style import UserStyleManager


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, model: str) -> str:
        return self.response


def test_style_manager_learns_repeated_slang(tmp_path: Path):
    manager = UserStyleManager(tmp_path / "style.json")
    manager.observe("Yo bro that was funny lol")
    manager.observe("bro give me the full file lol")
    context = manager.get_prompt_context()
    assert "bro" in context
    assert "Humor preference" in context


def test_coding_agent_requires_exact_approval(tmp_path: Path):
    source = tmp_path / "app.py"
    source.write_text("print('old')\n", encoding="utf-8")
    agent = CodingAgent(tmp_path, tmp_path / "proposals.json")
    proposal = agent.propose("Update greeting", {"app.py": "print('new')\n"})

    with pytest.raises(ApprovalRequiredError):
        agent.apply(proposal.proposal_id)

    assert source.read_text(encoding="utf-8") == "print('old')\n"
    assert agent.approve_from_text(f"approve coding proposal {proposal.proposal_id}") == proposal.proposal_id
    assert source.read_text(encoding="utf-8") == "print('new')\n"


def test_coding_agent_blocks_path_escape(tmp_path: Path):
    agent = CodingAgent(tmp_path, tmp_path / "proposals.json")
    with pytest.raises(ValueError):
        agent.propose("unsafe", {"../outside.py": "bad"})


def test_coding_agent_stages_a_new_file_from_prompt(tmp_path: Path):
    agent = CodingAgent(tmp_path, tmp_path / "proposals.json")
    llm = FakeLLM(
        '{"summary":"Add helper","changes":[{"path":"src/helper.py","content":"VALUE = 1\\n"}]}'
    )
    proposal = agent.propose_from_prompt(
        "Create src/helper.py for a small helper module.", llm
    )
    assert proposal.changes[0].path == "src/helper.py"
    assert not (tmp_path / "src/helper.py").exists()
