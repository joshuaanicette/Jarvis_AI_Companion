"use strict";

const $ = id => document.getElementById(id);

const elements = {
    conversation: $("conversation"),
    welcomePanel: $("welcomePanel"),
    messageList: $("messageList"),
    thinkingCard: $("thinkingCard"),
    thinkingStage: $("thinkingStage"),
    thinkingModel: $("thinkingModel"),
    chatForm: $("chatForm"),
    messageInput: $("messageInput"),
    sendButton: $("sendButton"),
    voiceToggle: $("voiceToggle"),
    microphoneButton: $("microphoneButton"),
    cameraComposerButton: $("cameraComposerButton"),
    characterCount: $("characterCount"),
    connectionText: $("connectionText"),
    modelStatusText: $("modelStatusText"),
    sidebar: $("sidebar"),
    sidebarBackdrop: $("sidebarBackdrop"),
    closeSidebarButton: $("closeSidebarButton"),
    menuButton: $("menuButton"),
    cameraPanel: $("cameraPanel"),
    cameraPreview: $("cameraPreview"),
    cameraStatus: $("cameraStatus"),
    refreshCameraButton: $("refreshCameraButton"),
    closeCameraButton: $("closeCameraButton"),
    newChatButton: $("newChatButton"),
    conversationList: $("conversationList"),
    historySearch: $("historySearch"),
    historyEmpty: $("historyEmpty"),
    historyCount: $("historyCount"),
    conversationTitle: $("conversationTitle"),
    themeButton: $("themeButton"),
    historyContextMenu: $("historyContextMenu"),
    renameChatAction: $("renameChatAction"),
    deleteChatAction: $("deleteChatAction"),
    toast: $("toast"),
};

let voiceEnabled = readPreference("jarvis-voice", "true") !== "false";
let sending = false;
let openingTool = false;
let recording = false;
let mediaRecorder = null;
let microphoneStream = null;
let recordedChunks = [];
let stageTimer = null;
let stageIndex = 0;
let toastTimer = null;
let conversationId = null;
let conversations = [];
let microphoneAvailable = false;
let cameraAvailable = false;
let selectedHistoryItem = null;

const processingStages = [
    "Understanding your request…",
    "Selecting the best local model…",
    "Checking tools and memory…",
    "Generating and verifying the response…",
];

function makeId() {
    if (
        typeof crypto !== "undefined"
        && typeof crypto.randomUUID === "function"
    ) {
        return crypto.randomUUID();
    }

    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function readPreference(key, fallback) {
    try {
        return localStorage.getItem(key) ?? fallback;
    } catch {
        return fallback;
    }
}

function savePreference(key, value) {
    try {
        localStorage.setItem(key, String(value));
    } catch {
        // The app still works if browser storage is unavailable.
    }
}

function escapeHtml(value) {
    const node = document.createElement("div");
    node.textContent = String(value ?? "");
    return node.innerHTML;
}

function inlineMarkdown(value) {
    return value
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/__([^_]+)__/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function formatResponse(value) {
    const codeBlocks = [];
    let source = escapeHtml(value).replace(
        /```(?:[\w.+-]+)?\n?([\s\S]*?)```/g,
        (_, code) => {
            const token = `@@JARVIS_CODE_${codeBlocks.length}@@`;
            codeBlocks.push(`<pre><code>${code.replace(/^\n|\n$/g, "")}</code></pre>`);
            return `\n${token}\n`;
        }
    );

    const lines = source.split("\n");
    const output = [];
    let listType = null;

    const closeList = () => {
        if (listType) {
            output.push(`</${listType}>`);
            listType = null;
        }
    };

    for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        const codeMatch = line.trim().match(/^@@JARVIS_CODE_(\d+)@@$/);

        if (codeMatch) {
            closeList();
            output.push(codeBlocks[Number(codeMatch[1])]);
            continue;
        }

        if (!line.trim()) {
            closeList();
            continue;
        }

        const heading = line.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
            closeList();
            const level = heading[1].length + 1;
            output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
            continue;
        }

        const unordered = line.match(/^\s*[-*]\s+(.+)$/);
        if (unordered) {
            if (listType !== "ul") {
                closeList();
                listType = "ul";
                output.push("<ul>");
            }
            output.push(`<li>${inlineMarkdown(unordered[1])}</li>`);
            continue;
        }

        const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
        if (ordered) {
            if (listType !== "ol") {
                closeList();
                listType = "ol";
                output.push("<ol>");
            }
            output.push(`<li>${inlineMarkdown(ordered[1])}</li>`);
            continue;
        }

        closeList();
        output.push(`<p>${inlineMarkdown(line)}</p>`);
    }

    closeList();
    return output.join("");
}

function shortModel(model) {
    const name = String(model || "").split(":")[0];
    return name
        .replace(/qwen[\w.-]*/i, "Qwen")
        .replace(/gemma[\w.-]*/i, "Gemma")
        || "Local model";
}

function showToast(message) {
    clearTimeout(toastTimer);
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    toastTimer = window.setTimeout(
        () => elements.toast.classList.remove("show"),
        2200
    );
}

async function api(url, options = {}) {
    const response = await fetch(url, options);
    let payload = null;

    if (response.status !== 204) {
        const contentType = response.headers.get("content-type") || "";
        payload = contentType.includes("application/json")
            ? await response.json()
            : await response.text();
    }

    if (!response.ok) {
        const detail = payload && typeof payload === "object"
            ? payload.detail
            : payload;
        throw new Error(detail || `Request failed (${response.status}).`);
    }

    return payload;
}

async function loadStatus() {
    try {
        const status = await api("/api/status");
        microphoneAvailable = Boolean(status.microphone_available);
        cameraAvailable = Boolean(status.camera_available);

        elements.connectionText.textContent = "Local AI online";
        elements.modelStatusText.textContent =
            `${shortModel(status.fast_model)} + ${shortModel(status.reasoning_model)}`;

        elements.microphoneButton.disabled = !microphoneAvailable;
        elements.microphoneButton.title = microphoneAvailable
            ? "Start voice recording"
            : "Whisper is not available";
        elements.cameraComposerButton.disabled = !cameraAvailable;

        document.querySelectorAll('[data-tool="camera"]').forEach(button => {
            button.disabled = !cameraAvailable;
        });
    } catch {
        elements.connectionText.textContent = "Local AI unavailable";
        elements.modelStatusText.textContent = "Connection unavailable";
    }
}

function setWelcomeVisible(visible) {
    elements.welcomePanel.classList.toggle("hidden", !visible);
}

function addUserMessage(text) {
    setWelcomeVisible(false);
    const row = document.createElement("article");
    row.className = "message-row user-row";
    row.innerHTML = `
        <div class="message-column">
            <div class="message-bubble">${escapeHtml(text)}</div>
        </div>
    `;
    elements.messageList.appendChild(row);
    scrollToBottom();
}

function addAssistantMessage(payload, options = {}) {
    setWelcomeVisible(false);
    const responseText = String(payload.response ?? payload.content ?? "");
    const row = document.createElement("article");
    row.className = "message-row assistant-row";

    const actions = (payload.actions || [])
        .filter(action => action && action.url)
        .map(action => `
            <button class="action-button" type="button" data-url="${escapeHtml(action.url)}">
                ${escapeHtml(action.label || "Open")}
            </button>
        `)
        .join("");

    const badge = payload.model
        ? `<span class="model-badge">${escapeHtml(shortModel(payload.model))}</span>`
        : "";

    row.innerHTML = `
        <div class="assistant-avatar">J</div>
        <div class="message-column">
            <div class="message-author"><strong>Jarvis</strong>${badge}</div>
            <div class="message-bubble">${formatResponse(responseText)}</div>
            ${actions ? `<div class="message-actions">${actions}</div>` : ""}
            <div class="message-utility">
                <button class="copy-button" type="button" aria-label="Copy response" title="Copy response">
                    <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>
                </button>
            </div>
        </div>
    `;

    row.querySelectorAll(".action-button").forEach(button => {
        button.addEventListener("click", () => openExternal(button.dataset.url));
    });

    row.querySelector(".copy-button").addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(responseText);
            showToast("Response copied");
        } catch {
            showToast("Could not copy the response");
        }
    });

    elements.messageList.appendChild(row);
    if (!options.noScroll) {
        scrollToBottom();
    }
}

function addErrorMessage(text) {
    addAssistantMessage({
        response: `I could not complete that request: ${text}`,
        model: "system",
        category: "error",
        actions: [],
    });
}

function renderStoredMessage(message) {
    if (message.role === "user") {
        addUserMessage(message.content);
    } else if (message.role === "assistant") {
        addAssistantMessage({
            response: message.content,
            model: message.model,
            category: message.category,
            actions: [],
        }, { noScroll: true });
    }
}

function showThinking(message) {
    stageIndex = 0;
    elements.thinkingStage.textContent = processingStages[0];
    elements.thinkingModel.textContent =
        /integral|circuit|engineering|math|solve|physics|program|code/i.test(message)
            ? "Qwen likely"
            : "Selecting model";
    elements.thinkingCard.classList.remove("hidden");
    scrollToBottom();

    clearInterval(stageTimer);
    stageTimer = window.setInterval(() => {
        stageIndex = Math.min(stageIndex + 1, processingStages.length - 1);
        elements.thinkingStage.textContent = processingStages[stageIndex];
        scrollToBottom();
    }, 1500);
}

function hideThinking() {
    clearInterval(stageTimer);
    stageTimer = null;
    elements.thinkingCard.classList.add("hidden");
}

function setSending(value) {
    sending = Boolean(value);
    elements.messageInput.disabled = sending;
    updateSendState();
}

function updateSendState() {
    elements.sendButton.disabled = sending || !elements.messageInput.value.trim();
}

async function sendMessage(text) {
    const message = String(text || "").trim();
    if (!message || sending) {
        return;
    }

    if (!conversationId) {
        conversationId = makeId();
    }

    addUserMessage(message);
    elements.messageInput.value = "";
    resizeTextarea();
    updateCharacterCount();
    showThinking(message);
    setSending(true);

    try {
        const payload = await api("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message,
                conversation_id: conversationId,
                voice_enabled: voiceEnabled,
            }),
        });

        hideThinking();
        addAssistantMessage(payload);
        elements.conversationTitle.textContent = payload.title || "Jarvis";
        setHash(conversationId);
        await refreshConversationHistory();
    } catch (error) {
        hideThinking();
        addErrorMessage(error.message);
    } finally {
        setSending(false);
        elements.messageInput.focus();
    }
}

function renderConversationHistory() {
    const query = elements.historySearch.value.trim().toLocaleLowerCase();
    const visible = conversations.filter(item =>
        !query || item.title.toLocaleLowerCase().includes(query)
    );

    elements.conversationList.innerHTML = "";
    elements.historyCount.textContent = String(conversations.length);
    elements.historyEmpty.classList.toggle("hidden", visible.length > 0);
    elements.historyEmpty.textContent = conversations.length
        ? "No chats match your search."
        : "Your saved chats will appear here.";

    for (const item of visible) {
        const wrapper = document.createElement("div");
        wrapper.className = `history-item${item.id === conversationId ? " active" : ""}`;
        wrapper.setAttribute("role", "listitem");

        const main = document.createElement("button");
        main.className = "history-main";
        main.type = "button";
        main.textContent = item.title;
        main.title = item.title;
        main.addEventListener("click", () => loadConversation(item.id));

        const menu = document.createElement("button");
        menu.className = "history-menu";
        menu.type = "button";
        menu.textContent = "•••";
        menu.setAttribute("aria-label", `Chat options for ${item.title}`);
        menu.addEventListener("click", event => {
            event.stopPropagation();
            openHistoryMenu(item, menu);
        });

        wrapper.append(main, menu);
        elements.conversationList.appendChild(wrapper);
    }
}

async function refreshConversationHistory() {
    try {
        conversations = await api("/api/conversations");
        renderConversationHistory();
    } catch (error) {
        showToast(error.message);
    }
}

async function loadConversation(identifier, options = {}) {
    if (!identifier || sending) {
        return;
    }

    try {
        const data = await api(`/api/conversations/${encodeURIComponent(identifier)}`);
        conversationId = data.id;
        elements.messageList.innerHTML = "";
        hideThinking();
        closeCameraPanel();
        data.messages.forEach(renderStoredMessage);
        setWelcomeVisible(data.messages.length === 0);
        elements.conversationTitle.textContent = data.title || "Jarvis";
        setHash(conversationId);
        renderConversationHistory();
        closeSidebar();

        if (!options.noFocus) {
            elements.messageInput.focus();
        }

        window.requestAnimationFrame(() => {
            elements.conversation.scrollTop = elements.conversation.scrollHeight;
        });
    } catch (error) {
        showToast(error.message);
    }
}

function startNewChat() {
    if (sending) {
        return;
    }

    conversationId = makeId();
    elements.messageList.innerHTML = "";
    elements.conversationTitle.textContent = "Jarvis";
    elements.messageInput.value = "";
    setWelcomeVisible(true);
    hideThinking();
    closeCameraPanel();
    setHash("");
    renderConversationHistory();
    resizeTextarea();
    updateCharacterCount();
    closeSidebar();
    elements.messageInput.focus();
}

function openHistoryMenu(item, anchor) {
    selectedHistoryItem = item;
    const bounds = anchor.getBoundingClientRect();
    const width = 150;
    const height = 82;
    const left = Math.max(8, Math.min(bounds.right - width, window.innerWidth - width - 8));
    const top = Math.max(8, Math.min(bounds.bottom + 4, window.innerHeight - height - 8));

    elements.historyContextMenu.style.left = `${left}px`;
    elements.historyContextMenu.style.top = `${top}px`;
    elements.historyContextMenu.classList.remove("hidden");
}

function closeHistoryMenu() {
    selectedHistoryItem = null;
    elements.historyContextMenu.classList.add("hidden");
}

async function renameSelectedChat() {
    const item = selectedHistoryItem;
    closeHistoryMenu();
    if (!item) {
        return;
    }

    const title = window.prompt("Rename this chat:", item.title);
    if (!title || !title.trim() || title.trim() === item.title) {
        return;
    }

    try {
        const updated = await api(
            `/api/conversations/${encodeURIComponent(item.id)}`,
            {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title: title.trim() }),
            }
        );
        if (item.id === conversationId) {
            elements.conversationTitle.textContent = updated.title;
        }
        await refreshConversationHistory();
        showToast("Chat renamed");
    } catch (error) {
        showToast(error.message);
    }
}

async function deleteSelectedChat() {
    const item = selectedHistoryItem;
    closeHistoryMenu();
    if (!item) {
        return;
    }

    const confirmed = window.confirm(`Delete “${item.title}”? This cannot be undone.`);
    if (!confirmed) {
        return;
    }

    try {
        await api(`/api/conversations/${encodeURIComponent(item.id)}`, {
            method: "DELETE",
        });
        if (item.id === conversationId) {
            startNewChat();
        }
        await refreshConversationHistory();
        showToast("Chat deleted");
    } catch (error) {
        showToast(error.message);
    }
}

function setHash(identifier) {
    const next = identifier ? `#chat=${encodeURIComponent(identifier)}` : location.pathname;
    history.replaceState(null, "", next);
}

function getHashConversation() {
    const match = location.hash.match(/^#chat=(.+)$/);
    return match ? decodeURIComponent(match[1]) : null;
}

function openExternal(url) {
    const cleanUrl = String(url || "").trim();
    if (!cleanUrl) {
        return;
    }

    try {
        const parsed = new URL(cleanUrl, location.origin);
        if (!["http:", "https:"].includes(parsed.protocol)) {
            throw new Error("Unsupported link protocol.");
        }
        window.open(parsed.href, "_blank", "noopener,noreferrer");
    } catch {
        showToast("Jarvis blocked an invalid link");
    }
}

async function openTool(toolName) {
    if (openingTool) {
        return;
    }

    openingTool = true;
    try {
        const payload = await api(`/api/open/${encodeURIComponent(toolName)}`, {
            method: "POST",
        });
        addAssistantMessage({
            response: `The ${toolName} dashboard is ready.`,
            model: "tool",
            category: toolName,
            actions: [{ label: `Open ${toolName}`, url: payload.url }],
        });
    } catch (error) {
        addErrorMessage(error.message);
    } finally {
        openingTool = false;
    }
}

async function loadCameraSnapshot() {
    if (!cameraAvailable) {
        showToast("Jarvis Vision is not available");
        return;
    }

    elements.cameraPanel.classList.remove("hidden");
    elements.cameraStatus.textContent = "Scanning with YOLO…";
    elements.refreshCameraButton.disabled = true;

    try {
        const response = await fetch(`/api/camera/snapshot?t=${Date.now()}`, {
            method: "GET",
            cache: "no-store",
        });

        if (!response.ok) {
            let detail = "Camera request failed.";
            try {
                const payload = await response.json();
                detail = payload.detail || detail;
            } catch {
                // The response was not JSON.
            }
            throw new Error(detail);
        }

        const imageBlob = await response.blob();
        const oldUrl = elements.cameraPreview.dataset.objectUrl;
        if (oldUrl) {
            URL.revokeObjectURL(oldUrl);
        }

        const objectUrl = URL.createObjectURL(imageBlob);
        elements.cameraPreview.src = objectUrl;
        elements.cameraPreview.dataset.objectUrl = objectUrl;
        elements.cameraStatus.textContent =
            response.headers.get("X-Jarvis-Scene") || "Camera scan complete";
        elements.cameraPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (error) {
        elements.cameraStatus.textContent = error.message;
        addErrorMessage(error.message);
    } finally {
        elements.refreshCameraButton.disabled = false;
    }
}

function closeCameraPanel() {
    elements.cameraPanel.classList.add("hidden");
    const objectUrl = elements.cameraPreview.dataset.objectUrl;
    if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
    }
    elements.cameraPreview.removeAttribute("src");
    delete elements.cameraPreview.dataset.objectUrl;
    elements.cameraStatus.textContent = "Camera ready";
}

async function toggleRecording() {
    if (recording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    if (!microphoneAvailable) {
        showToast("Whisper is not available");
        return;
    }

    try {
        microphoneStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        const mimeType = chooseMimeType();
        mediaRecorder = new MediaRecorder(
            microphoneStream,
            mimeType ? { mimeType } : undefined
        );
        recordedChunks = [];

        mediaRecorder.addEventListener("dataavailable", event => {
            if (event.data && event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        });
        mediaRecorder.addEventListener("stop", uploadRecording);
        mediaRecorder.start(250);

        recording = true;
        elements.microphoneButton.classList.add("recording");
        elements.connectionText.textContent = "Listening…";
    } catch (error) {
        stopMicrophoneStream();
        addErrorMessage(error.message);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    recording = false;
    elements.microphoneButton.classList.remove("recording");
    elements.connectionText.textContent = "Transcribing…";
}

async function uploadRecording() {
    try {
        if (!recordedChunks.length) {
            throw new Error("The recording was empty.");
        }

        const mimeType = mediaRecorder?.mimeType || "audio/webm";
        const extension = mimeType.includes("ogg") ? "ogg" : "webm";
        const blob = new Blob(recordedChunks, { type: mimeType });
        const formData = new FormData();
        formData.append("audio", blob, `jarvis-recording.${extension}`);

        elements.microphoneButton.disabled = true;
        const payload = await api("/api/transcribe", {
            method: "POST",
            body: formData,
        });

        elements.messageInput.value = String(payload.text || "").trim();
        resizeTextarea();
        updateCharacterCount();
        elements.connectionText.textContent = "Speech ready";
        elements.messageInput.focus();
    } catch (error) {
        addErrorMessage(error.message);
    } finally {
        elements.microphoneButton.disabled = !microphoneAvailable;
        elements.microphoneButton.classList.remove("recording");
        recordedChunks = [];
        stopMicrophoneStream();
        window.setTimeout(() => {
            elements.connectionText.textContent = "Local AI online";
        }, 1600);
    }
}

function stopMicrophoneStream() {
    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
    }
    microphoneStream = null;
    mediaRecorder = null;
    recording = false;
}

function chooseMimeType() {
    const choices = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/ogg",
    ];
    return choices.find(choice => MediaRecorder.isTypeSupported(choice)) || "";
}

function scrollToBottom() {
    window.requestAnimationFrame(() => {
        elements.conversation.scrollTop = elements.conversation.scrollHeight;
    });
}

function resizeTextarea() {
    elements.messageInput.style.height = "auto";
    elements.messageInput.style.height =
        `${Math.min(elements.messageInput.scrollHeight, 190)}px`;
}

function updateCharacterCount() {
    elements.characterCount.textContent =
        `${elements.messageInput.value.length} / 8000`;
    updateSendState();
}

function openSidebar() {
    elements.sidebar.classList.add("open");
    elements.sidebarBackdrop.classList.add("show");
}

function closeSidebar() {
    elements.sidebar.classList.remove("open");
    elements.sidebarBackdrop.classList.remove("show");
}

function applyTheme() {
    const saved = readPreference("jarvis-theme", "system");
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const dark = saved === "dark" || (saved === "system" && systemDark);
    document.body.classList.toggle("dark", dark);
}

function toggleTheme() {
    const dark = !document.body.classList.contains("dark");
    document.body.classList.toggle("dark", dark);
    savePreference("jarvis-theme", dark ? "dark" : "light");
}

function updateVoiceToggle() {
    elements.voiceToggle.classList.toggle("active", voiceEnabled);
    elements.voiceToggle.setAttribute("aria-pressed", String(voiceEnabled));
    elements.voiceToggle.querySelector("span").textContent =
        voiceEnabled ? "Voice on" : "Voice off";
}

function bindEvents() {
    elements.chatForm.addEventListener("submit", event => {
        event.preventDefault();
        sendMessage(elements.messageInput.value);
    });

    elements.messageInput.addEventListener("input", () => {
        resizeTextarea();
        updateCharacterCount();
    });

    elements.messageInput.addEventListener("keydown", event => {
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            sendMessage(elements.messageInput.value);
        }
    });

    document.querySelectorAll(".suggestion").forEach(button => {
        button.addEventListener("click", () => sendMessage(button.dataset.prompt));
    });

    document.querySelectorAll(".tool-button").forEach(button => {
        button.addEventListener("click", async () => {
            if (button.disabled) {
                return;
            }
            const tool = button.dataset.tool;
            const command = button.dataset.command;
            if (tool === "camera") {
                await loadCameraSnapshot();
            } else if (tool) {
                await openTool(tool);
            } else if (command) {
                await sendMessage(command);
            }
            closeSidebar();
        });
    });

    elements.voiceToggle.addEventListener("click", () => {
        voiceEnabled = !voiceEnabled;
        savePreference("jarvis-voice", voiceEnabled);
        updateVoiceToggle();
    });

    elements.microphoneButton.addEventListener("click", toggleRecording);
    elements.cameraComposerButton.addEventListener("click", loadCameraSnapshot);
    elements.refreshCameraButton.addEventListener("click", loadCameraSnapshot);
    elements.closeCameraButton.addEventListener("click", closeCameraPanel);
    elements.newChatButton.addEventListener("click", startNewChat);
    elements.historySearch.addEventListener("input", renderConversationHistory);
    elements.themeButton.addEventListener("click", toggleTheme);
    elements.renameChatAction.addEventListener("click", renameSelectedChat);
    elements.deleteChatAction.addEventListener("click", deleteSelectedChat);
    elements.menuButton.addEventListener("click", openSidebar);
    elements.closeSidebarButton.addEventListener("click", closeSidebar);
    elements.sidebarBackdrop.addEventListener("click", closeSidebar);

    document.addEventListener("click", event => {
        if (!elements.historyContextMenu.contains(event.target)) {
            closeHistoryMenu();
        }
    });

    window.addEventListener("resize", closeHistoryMenu);

    window.addEventListener("beforeunload", () => {
        stopMicrophoneStream();
        closeCameraPanel();
    });
}

async function initialize() {
    applyTheme();
    updateVoiceToggle();
    bindEvents();
    updateCharacterCount();

    await Promise.all([
        loadStatus(),
        refreshConversationHistory(),
    ]);

    const requested = getHashConversation();
    const initial = requested && conversations.some(item => item.id === requested)
        ? requested
        : conversations[0]?.id;

    if (initial) {
        await loadConversation(initial, { noFocus: true });
    } else {
        startNewChat();
    }

    elements.messageInput.focus();
}

initialize();
