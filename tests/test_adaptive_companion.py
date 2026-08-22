from pathlib import Path

import pytest

from src.ai.coding_agent import ApprovalRequiredError, CodingAgent
from src.ai.user_style import UserStyleManager
from src.tools.coding_agent_tool import CodingAgentTool


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_timeout = None

    def generate(
        self,
        prompt: str,
        model: str,
        timeout: float | None = None,
    ) -> str:
        self.last_timeout = timeout
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


def test_coding_agent_tool_implements_abstract_execute(tmp_path: Path):
    agent = CodingAgent(tmp_path, tmp_path / "proposals.json")
    tool = CodingAgentTool(agent, FakeLLM("{}"))

    assert tool.name == "coding_agent"
    assert "Coding commands" in tool.execute("coding help")
    assert tool.run("coding help") == tool.execute("coding help")


def test_code_analysis_uses_extended_timeout(tmp_path: Path):
    source = tmp_path / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    llm = FakeLLM("The file defines one constant.")
    agent = CodingAgent(
        tmp_path,
        tmp_path / "proposals.json",
        request_timeout_seconds=600,
    )

    response = agent.analyze_from_prompt(
        "Explain app.py",
        llm,
    )

    assert "constant" in response
    assert llm.last_timeout == 600


def test_code_analysis_rejects_oversized_combined_context(tmp_path: Path):
    (tmp_path / "one.py").write_text("a" * 3000, encoding="utf-8")
    (tmp_path / "two.py").write_text("b" * 3000, encoding="utf-8")
    agent = CodingAgent(
        tmp_path,
        tmp_path / "proposals.json",
        max_total_source_chars=4000,
    )

    with pytest.raises(ValueError, match="Select fewer or smaller files"):
        agent.analyze_from_prompt(
            "Explain one.py and two.py",
            FakeLLM("unused"),
        )
