from fastapi import FastAPI


def create_api(app_context):
    api = FastAPI(title="Jay AI Companion API")

    @api.get("/health")
    def health():
        return {"status": "ok", "online": app_context.state.online}

    @api.post("/chat")
    def chat(payload: dict):
        text = str(payload.get("text", ""))
        return {"response": app_context.conversation.process(text)}

    return api
