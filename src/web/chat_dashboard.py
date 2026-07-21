from __future__ import annotations

import inspect
import logging
import subprocess
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import (
    HTMLResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles

from src.web.chat_database import ChatDatabase
from src.web.chat_models import (
    ChatAction,
    ChatRequest,
    ChatResponse,
    ConversationDetails,
    ConversationSummary,
    CreateConversationRequest,
    UpdateConversationRequest,
)


logger = logging.getLogger(
    "jarvis.chat"
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TEMPLATE_PATH = (
    PROJECT_ROOT
    / "templates"
    / "jarvis_chat.html"
)

STATIC_PATH = (
    PROJECT_ROOT
    / "static"
    / "chat"
)

CHAT_DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "chat"
    / "jarvis_chat.db"
)

MAX_AUDIO_BYTES = (
    20
    * 1024
    * 1024
)


class JayChatDashboard:
    """
    Jarvis browser chat dashboard.

    The dashboard shares the Application instance passed into it,
    including ConversationManager, Whisper, vision, reminders,
    weather, navigation, memory, Piper, and Ollama.
    """

    def __init__(
        self,
        application: Any,
        host: str = "127.0.0.1",
        port: int = 8780,
    ) -> None:
        self.application = application
        self.host = str(host)
        self.port = int(port)

        self.running = False

        self._thread: (
            threading.Thread
            | None
        ) = None

        self._server: (
            uvicorn.Server
            | None
        ) = None

        self._transcription_lock = (
            threading.Lock()
        )

        self._camera_lock = (
            threading.Lock()
        )

        self._conversation_lock = (
            threading.Lock()
        )

        self.chat_database = ChatDatabase(
            CHAT_DATABASE_PATH
        )

        self.app = FastAPI(
            title=(
                "Jarvis AI Companion"
            ),
            version="2.0.0",
        )

        STATIC_PATH.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.app.mount(
            "/static/chat",
            StaticFiles(
                directory=str(
                    STATIC_PATH
                )
            ),
            name="chat_static",
        )

        self._configure_routes()

    @property
    def url(
        self,
    ) -> str:
        return (
            f"http://{self.host}:"
            f"{self.port}/"
        )

    def _configure_routes(
        self,
    ) -> None:
        @self.app.get(
            "/",
            response_class=HTMLResponse,
        )
        def index() -> HTMLResponse:
            if not TEMPLATE_PATH.exists():
                return HTMLResponse(
                    content=(
                        "<h1>Jarvis chat "
                        "template is missing.</h1>"
                    ),
                    status_code=500,
                )

            return HTMLResponse(
                content=(
                    TEMPLATE_PATH.read_text(
                        encoding="utf-8"
                    )
                )
            )

        @self.app.get(
            "/api/status"
        )
        def status() -> dict[str, Any]:
            return {
                "online": True,
                "name": "Jarvis",
                "fast_model": (
                    self._fast_model()
                ),
                "reasoning_model": (
                    self._reasoning_model()
                ),
                "microphone_available": (
                    self._stt_available()
                ),
                "camera_available": (
                    self._camera_available()
                ),
                "weather_url": (
                    self._dashboard_url(
                        attribute_name=(
                            "weather_dashboard"
                        ),
                        fallback_port=8765,
                    )
                ),
                "navigation_url": (
                    self._dashboard_url(
                        attribute_name=(
                            "navigation_dashboard"
                        ),
                        fallback_port=8770,
                    )
                ),
            }

        @self.app.get(
            "/api/conversations",
            response_model=list[ConversationSummary],
        )
        def list_conversations() -> list[dict[str, object]]:
            return self.chat_database.list_conversations()

        @self.app.post(
            "/api/conversations",
            response_model=ConversationSummary,
            status_code=201,
        )
        def create_conversation(
            request: CreateConversationRequest,
        ) -> dict[str, object]:
            return self.chat_database.create_conversation(
                title=request.title
            )

        @self.app.get(
            "/api/conversations/{conversation_id}",
            response_model=ConversationDetails,
        )
        def get_conversation(
            conversation_id: str,
        ) -> dict[str, object]:
            conversation = self.chat_database.get_conversation(
                conversation_id
            )

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

            return {
                **conversation,
                "messages": self.chat_database.get_messages(
                    conversation_id
                ),
            }

        @self.app.patch(
            "/api/conversations/{conversation_id}",
            response_model=ConversationSummary,
        )
        def rename_conversation(
            conversation_id: str,
            request: UpdateConversationRequest,
        ) -> dict[str, object]:
            conversation = self.chat_database.rename_conversation(
                conversation_id=conversation_id,
                title=request.title,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

            return conversation

        @self.app.delete(
            "/api/conversations/{conversation_id}",
            status_code=204,
        )
        def delete_conversation(
            conversation_id: str,
        ) -> Response:
            deleted = self.chat_database.delete_conversation(
                conversation_id
            )

            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

            return Response(status_code=204)

        @self.app.post(
            "/api/chat",
            response_model=ChatResponse,
        )
        def chat(
            request: ChatRequest,
        ) -> ChatResponse:
            message = (
                request.message.strip()
            )

            if not message:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "A message is required."
                    ),
                )

            try:
                print(
                    (
                        "\nYou [browser]: "
                        f"{message}"
                    ),
                    flush=True,
                )

                category = self._classify(
                    message
                )

                model = self._select_model(
                    message=message,
                    category=category,
                )

                with self._conversation_lock:
                    self.chat_database.ensure_conversation(
                        request.conversation_id
                    )

                    self._activate_conversation_history(
                        request.conversation_id
                    )

                    response_text = (
                        self._process_message(
                            message=message,
                            voice_enabled=(
                                request.voice_enabled
                            ),
                        )
                    )

                    response_text = str(
                        response_text
                    ).strip()

                    self.chat_database.add_message(
                        conversation_id=request.conversation_id,
                        role="user",
                        content=message,
                        category=category,
                        model=model,
                    )

                    self.chat_database.add_message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=response_text,
                        category=category,
                        model=model,
                    )

                    conversation = self.chat_database.get_conversation(
                        request.conversation_id
                    )

                print(
                    (
                        "Jarvis: "
                        f"{response_text}"
                    ),
                    flush=True,
                )

                actions = self._detect_actions(
                    message=message,
                    response=response_text,
                )

                return ChatResponse(
                    response=response_text,
                    conversation_id=(
                        request.conversation_id
                    ),
                    title=str(
                        (conversation or {}).get(
                            "title",
                            "New chat",
                        )
                    ),
                    category=category,
                    model=model,
                    actions=actions,
                )

            except HTTPException:
                raise

            except Exception as error:
                logger.exception(
                    "Chat request failed"
                )

                raise HTTPException(
                    status_code=500,
                    detail=str(error),
                ) from error

        @self.app.post(
            "/api/transcribe"
        )
        async def transcribe_audio(
            audio: UploadFile = File(...),
        ) -> dict[str, str]:
            if not self._stt_available():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Whisper speech "
                        "recognition is not "
                        "available."
                    ),
                )

            raw_content_type = (
                audio.content_type
                or (
                    "application/"
                    "octet-stream"
                )
            )

            content_type = (
                raw_content_type
                .split(";", 1)[0]
                .strip()
                .lower()
            )

            allowed_types = {
                "audio/webm",
                "video/webm",
                "audio/ogg",
                "audio/wav",
                "audio/x-wav",
                "audio/mpeg",
                "audio/mp4",
                "application/octet-stream",
            }

            if (
                content_type
                not in allowed_types
            ):
                raise HTTPException(
                    status_code=415,
                    detail=(
                        "Unsupported audio "
                        "format: "
                        f"{raw_content_type}"
                    ),
                )

            audio_bytes = (
                await audio.read()
            )

            if not audio_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The recording was "
                        "empty."
                    ),
                )

            if (
                len(audio_bytes)
                > MAX_AUDIO_BYTES
            ):
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "The recording is "
                        "too large."
                    ),
                )

            source_path: (
                Path
                | None
            ) = None

            wav_path: (
                Path
                | None
            ) = None

            try:
                source_path = (
                    self._write_temp_audio(
                        audio_bytes=(
                            audio_bytes
                        ),
                        suffix=(
                            self._audio_suffix(
                                filename=(
                                    audio.filename
                                ),
                                content_type=(
                                    content_type
                                ),
                            )
                        ),
                    )
                )

                wav_path = (
                    self
                    ._convert_to_whisper_wav(
                        source_path
                    )
                )

                with (
                    self
                    ._transcription_lock
                ):
                    text = (
                        self._transcribe_file(
                            wav_path
                        )
                    )

                cleaned_text = str(
                    text
                ).strip()

                if not cleaned_text:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Whisper did not "
                            "detect any speech."
                        ),
                    )

                return {
                    "text": cleaned_text
                }

            except HTTPException:
                raise

            except Exception as error:
                logger.exception(
                    (
                        "Audio transcription "
                        "failed"
                    )
                )

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Audio transcription "
                        f"failed: {error}"
                    ),
                ) from error

            finally:
                for path in (
                    source_path,
                    wav_path,
                ):
                    if path is None:
                        continue

                    try:
                        path.unlink(
                            missing_ok=True
                        )

                    except OSError:
                        pass

        @self.app.get(
            "/api/camera/snapshot"
        )
        def camera_snapshot() -> Response:
            vision = getattr(
                self.application,
                "vision",
                None,
            )

            if vision is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Jarvis vision is "
                        "not available."
                    ),
                )

            try:
                with self._camera_lock:
                    scene = (
                        vision.process_once(
                            show_camera=False
                        )
                    )

                    frame = getattr(
                        vision,
                        "latest_frame",
                        None,
                    )

                    if frame is None:
                        raise RuntimeError(
                            (
                                "The camera did "
                                "not return a frame."
                            )
                        )

                    encoded, buffer = (
                        cv2.imencode(
                            ".jpg",
                            frame,
                            [
                                int(
                                    cv2
                                    .IMWRITE_JPEG_QUALITY
                                ),
                                85,
                            ],
                        )
                    )

                if not encoded:
                    raise RuntimeError(
                        (
                            "The camera frame "
                            "could not be encoded."
                        )
                    )

                safe_scene = (
                    str(scene)
                    .replace("\n", " ")
                    .replace("\r", " ")
                )[:500]

                return Response(
                    content=(
                        buffer.tobytes()
                    ),
                    media_type="image/jpeg",
                    headers={
                        "X-Jarvis-Scene": (
                            safe_scene
                        ),
                        "Cache-Control": (
                            "no-store, "
                            "no-cache, "
                            "must-revalidate"
                        ),
                    },
                )

            except HTTPException:
                raise

            except Exception as error:
                logger.exception(
                    (
                        "Camera snapshot "
                        "failed"
                    )
                )

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Camera snapshot "
                        f"failed: {error}"
                    ),
                ) from error

        @self.app.post(
            "/api/open/weather"
        )
        def open_weather() -> dict[str, str]:
            self._start_dashboard(
                "weather_dashboard"
            )

            return {
                "url": (
                    self._dashboard_url(
                        attribute_name=(
                            "weather_dashboard"
                        ),
                        fallback_port=8765,
                    )
                )
            }

        @self.app.post(
            "/api/open/navigation"
        )
        def open_navigation() -> dict[str, str]:
            self._start_dashboard(
                "navigation_dashboard"
            )

            return {
                "url": (
                    self._dashboard_url(
                        attribute_name=(
                            "navigation_dashboard"
                        ),
                        fallback_port=8770,
                    )
                )
            }

    def start(
        self,
    ) -> bool:
        if self.running:
            return False

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )

        self._server = uvicorn.Server(
            config
        )

        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
            name=(
                "JarvisChatDashboard"
            ),
        )

        self.running = True
        self._thread.start()

        time.sleep(
            0.8
        )

        if (
            self._thread is None
            or not self._thread.is_alive()
        ):
            self.running = False

            raise RuntimeError(
                (
                    "Jarvis chat dashboard "
                    "failed to start."
                )
            )

        logger.info(
            (
                "Jarvis chat dashboard "
                "started at %s"
            ),
            self.url,
        )

        return True

    def open(
        self,
    ) -> str:
        """
        Start the server when necessary and open its URL
        in the system's default browser.
        """

        if not self.running:
            self.start()

        opened = webbrowser.open_new_tab(
            self.url
        )

        if opened:
            logger.info(
                (
                    "Opened Jarvis chat "
                    "dashboard at %s"
                ),
                self.url,
            )

            return (
                "The Jarvis chat dashboard "
                "was opened in the browser."
            )

        logger.warning(
            (
                "Browser could not be opened "
                "automatically. Dashboard URL: %s"
            ),
            self.url,
        )

        return (
            "The Jarvis chat dashboard is "
            f"running at {self.url}"
        )

    def stop(
        self,
    ) -> None:
        if self._server is not None:
            self._server.should_exit = True

        if (
            self._thread is not None
            and self._thread.is_alive()
            and (
                self._thread
                is not threading.current_thread()
            )
        ):
            self._thread.join(
                timeout=3.0
            )

        self._thread = None
        self._server = None
        self.running = False

        logger.info(
            (
                "Jarvis chat dashboard "
                "stopped"
            )
        )

    def _process_message(
        self,
        message: str,
        voice_enabled: bool,
    ) -> str:
        conversation = getattr(
            self.application,
            "conversation",
            None,
        )

        process = getattr(
            conversation,
            "process",
            None,
        )

        if not callable(process):
            raise RuntimeError(
                (
                    "ConversationManager "
                    "does not provide "
                    "process()."
                )
            )

        try:
            parameters = (
                inspect.signature(
                    process
                ).parameters
            )

        except (
            TypeError,
            ValueError,
        ):
            parameters = {}

        if (
            "speak_response"
            in parameters
        ):
            return str(
                process(
                    message,
                    speak_response=(
                        voice_enabled
                    ),
                )
            )

        return str(
            process(
                message
            )
        )

    def _activate_conversation_history(
        self,
        conversation_id: str,
    ) -> None:
        conversation = getattr(
            self.application,
            "conversation",
            None,
        )

        replace_history = getattr(
            conversation,
            "replace_history",
            None,
        )

        if not callable(replace_history):
            return

        stored_messages = self.chat_database.get_messages(
            conversation_id=conversation_id,
            limit=12,
        )

        replace_history(
            [
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
                for message in stored_messages
                if message["role"] in {"user", "assistant"}
            ]
        )

    def _stt_available(
        self,
    ) -> bool:
        return (
            getattr(
                self.application,
                "stt",
                None,
            )
            is not None
        )

    def _camera_available(
        self,
    ) -> bool:
        return (
            getattr(
                self.application,
                "vision",
                None,
            )
            is not None
        )

    @staticmethod
    def _audio_suffix(
        filename: str | None,
        content_type: str,
    ) -> str:
        if filename:
            suffix = (
                Path(filename)
                .suffix
                .lower()
            )

            if suffix in {
                ".webm",
                ".ogg",
                ".wav",
                ".mp3",
                ".mp4",
                ".m4a",
            }:
                return suffix

        suffix_map = {
            "audio/webm": ".webm",
            "video/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".mp4",
        }

        return suffix_map.get(
            content_type,
            ".webm",
        )

    @staticmethod
    def _write_temp_audio(
        audio_bytes: bytes,
        suffix: str,
    ) -> Path:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(
                audio_bytes
            )

            return Path(
                temporary_file.name
            )

    @staticmethod
    def _convert_to_whisper_wav(
        source_path: Path,
    ) -> Path:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            wav_path = Path(
                temporary_file.name
            )

        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

        except FileNotFoundError as error:
            raise RuntimeError(
                (
                    "FFmpeg is not installed. "
                    "Run: sudo apt install ffmpeg"
                )
            ) from error

        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                (
                    "Audio conversion "
                    "timed out."
                )
            ) from error

        if result.returncode != 0:
            message = (
                result.stderr.strip()
                or "Unknown FFmpeg error."
            )

            raise RuntimeError(
                (
                    "FFmpeg could not "
                    "convert the recording: "
                    f"{message}"
                )
            )

        if (
            not wav_path.exists()
            or wav_path.stat().st_size == 0
        ):
            raise RuntimeError(
                (
                    "FFmpeg did not create "
                    "a valid WAV file."
                )
            )

        return wav_path

    def _transcribe_file(
        self,
        wav_path: Path,
    ) -> str:
        stt = getattr(
            self.application,
            "stt",
            None,
        )

        if stt is None:
            raise RuntimeError(
                (
                    "Whisper speech "
                    "recognition is unavailable."
                )
            )

        method_names = (
            "transcribe_file",
            "transcribe_path",
            "transcribe",
        )

        for method_name in method_names:
            method = getattr(
                stt,
                method_name,
                None,
            )

            if not callable(method):
                continue

            attempts = (
                lambda: method(
                    str(wav_path)
                ),
                lambda: method(
                    wav_path
                ),
                lambda: method(
                    audio_path=(
                        str(wav_path)
                    )
                ),
                lambda: method(
                    file_path=(
                        str(wav_path)
                    )
                ),
                lambda: method(
                    path=str(wav_path)
                ),
            )

            for attempt in attempts:
                try:
                    result = attempt()

                    return (
                        self
                        ._extract_transcription(
                            result
                        )
                    )

                except TypeError:
                    continue

        model = getattr(
            stt,
            "model",
            None,
        )

        model_transcribe = getattr(
            model,
            "transcribe",
            None,
        )

        if callable(model_transcribe):
            result = model_transcribe(
                str(wav_path)
            )

            return (
                self
                ._extract_transcription(
                    result
                )
            )

        raise RuntimeError(
            (
                "WhisperSpeechToText "
                "does not provide a "
                "supported transcription "
                "method."
            )
        )

    @staticmethod
    def _extract_transcription(
        result: Any,
    ) -> str:
        if result is None:
            return ""

        if isinstance(
            result,
            str,
        ):
            return result.strip()

        if isinstance(
            result,
            dict,
        ):
            for key in (
                "text",
                "transcription",
                "response",
            ):
                value = result.get(
                    key
                )

                if isinstance(
                    value,
                    str,
                ):
                    return value.strip()

        text_attribute = getattr(
            result,
            "text",
            None,
        )

        if isinstance(
            text_attribute,
            str,
        ):
            return (
                text_attribute.strip()
            )

        segments = (
            result[0]
            if (
                isinstance(
                    result,
                    tuple,
                )
                and result
            )
            else result
        )

        try:
            segment_texts = []

            for segment in segments:
                segment_text = getattr(
                    segment,
                    "text",
                    None,
                )

                if segment_text:
                    segment_texts.append(
                        str(
                            segment_text
                        ).strip()
                    )

            if segment_texts:
                return " ".join(
                    segment_texts
                ).strip()

        except TypeError:
            pass

        return str(
            result
        ).strip()

    def _classify(
        self,
        message: str,
    ) -> str:
        router = getattr(
            self.application,
            "subject_router",
            None,
        )

        if router is None:
            return "general"

        for method_name in (
            "classify",
            "route",
            "classify_question",
        ):
            method = getattr(
                router,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:
                result = method(
                    message
                )

            except Exception as error:
                logger.warning(
                    (
                        "Subject classification "
                        "failed: %s"
                    ),
                    error,
                )

                return "general"

            if isinstance(
                result,
                str,
            ):
                return result

            if isinstance(
                result,
                dict,
            ):
                return str(
                    result.get(
                        "category",
                        "general",
                    )
                )

            category = getattr(
                result,
                "category",
                None,
            )

            if category:
                return str(
                    category
                )

        return "general"

    def _select_model(
        self,
        message: str,
        category: str,
    ) -> str:
        router = getattr(
            self.application,
            "model_router",
            None,
        )

        if router is None:
            return (
                self._fast_model()
            )

        for method_name in (
            "select_model",
            "route",
            "choose_model",
        ):
            method = getattr(
                router,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:
                result = method(
                    text=message,
                    category=category,
                )

            except TypeError:
                try:
                    result = method(
                        message,
                        category,
                    )

                except TypeError:
                    result = method(
                        message
                    )

            if isinstance(
                result,
                str,
            ):
                return result

            if isinstance(
                result,
                dict,
            ):
                return str(
                    result.get(
                        "model",
                        self._fast_model(),
                    )
                )

            model = getattr(
                result,
                "model",
                None,
            )

            if model:
                return str(
                    model
                )

        return self._fast_model()

    def _detect_actions(
        self,
        message: str,
        response: str,
    ) -> list[ChatAction]:
        normalized = (
            f"{message} {response}"
            .casefold()
        )

        actions: list[
            ChatAction
        ] = []

        navigation_phrases = (
            "open navigation",
            "navigation dashboard",
            "show navigation",
            "open the map",
            "show the map",
        )

        if any(
            phrase in normalized
            for phrase
            in navigation_phrases
        ):
            self._start_dashboard(
                "navigation_dashboard"
            )

            actions.append(
                ChatAction(
                    type="open_url",
                    label=(
                        "Open Navigation"
                    ),
                    url=(
                        self._dashboard_url(
                            attribute_name=(
                                "navigation_dashboard"
                            ),
                            fallback_port=8770,
                        )
                    ),
                    target="_blank",
                )
            )

        weather_phrases = (
            "open weather",
            "weather dashboard",
            "show weather",
            "show the forecast",
        )

        if any(
            phrase in normalized
            for phrase
            in weather_phrases
        ):
            self._start_dashboard(
                "weather_dashboard"
            )

            actions.append(
                ChatAction(
                    type="open_url",
                    label="Open Weather",
                    url=(
                        self._dashboard_url(
                            attribute_name=(
                                "weather_dashboard"
                            ),
                            fallback_port=8765,
                        )
                    ),
                    target="_blank",
                )
            )

        return actions

    def _start_dashboard(
        self,
        attribute_name: str,
    ) -> None:
        dashboard = getattr(
            self.application,
            attribute_name,
            None,
        )

        if dashboard is None:
            return

        start = getattr(
            dashboard,
            "start",
            None,
        )

        if not callable(start):
            return

        try:
            parameters = (
                inspect.signature(
                    start
                ).parameters
            )

        except (
            TypeError,
            ValueError,
        ):
            parameters = {}

        try:
            if (
                "open_browser"
                in parameters
            ):
                start(
                    open_browser=False
                )
            else:
                start()

        except Exception as error:
            logger.warning(
                (
                    "Could not start "
                    "%s: %s"
                ),
                attribute_name,
                error,
            )

    def _dashboard_url(
        self,
        attribute_name: str,
        fallback_port: int,
    ) -> str:
        dashboard = getattr(
            self.application,
            attribute_name,
            None,
        )

        url = getattr(
            dashboard,
            "url",
            None,
        )

        if url:
            return str(
                url
            )

        return (
            "http://127.0.0.1:"
            f"{fallback_port}/"
        )

    def _fast_model(
        self,
    ) -> str:
        router = getattr(
            self.application,
            "model_router",
            None,
        )

        return str(
            getattr(
                router,
                "fast_model",
                "gemma3:1b",
            )
        )

    def _reasoning_model(
        self,
    ) -> str:
        router = getattr(
            self.application,
            "model_router",
            None,
        )

        return str(
            getattr(
                router,
                "reasoning_model",
                "qwen2.5:3b",
            )
        )