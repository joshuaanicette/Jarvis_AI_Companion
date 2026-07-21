from __future__ import annotations

from src.web.chat_database import ChatDatabase


def test_conversation_crud(tmp_path) -> None:
    database = ChatDatabase(tmp_path / "jarvis_chat.db")
    conversation = database.create_conversation()

    database.add_message(
        conversation_id=str(conversation["id"]),
        role="user",
        content="Explain Kirchhoff's voltage law.",
    )
    database.add_message(
        conversation_id=str(conversation["id"]),
        role="assistant",
        content="The signed voltage changes around a loop sum to zero.",
        category="engineering",
        model="qwen",
    )

    saved = database.get_conversation(str(conversation["id"]))
    messages = database.get_messages(str(conversation["id"]))

    assert saved is not None
    assert saved["title"] == "Explain Kirchhoff's voltage law."
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
    ]

    renamed = database.rename_conversation(
        str(conversation["id"]),
        "KVL notes",
    )
    assert renamed is not None
    assert renamed["title"] == "KVL notes"

    assert database.delete_conversation(str(conversation["id"]))
    assert database.list_conversations() == []
