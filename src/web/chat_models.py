from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
    )
    voice_enabled: bool = True


class ChatAction(BaseModel):
    type: str
    label: str
    url: str
    target: str = "_blank"


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    title: str
    category: str
    model: str
    actions: list[ChatAction] = Field(default_factory=list)


class CreateConversationRequest(BaseModel):
    title: str = Field(default="New chat", max_length=120)


class UpdateConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class MessageRecord(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    category: str | None = None
    model: str | None = None
    created_at: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationDetails(ConversationSummary):
    messages: list[MessageRecord] = Field(default_factory=list)