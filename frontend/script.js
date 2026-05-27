/**
 * Tech News RAG Assistant — frontend script
 *
 * Wires the input box to the FastAPI /ask endpoint and renders the response
 * (answer + source cards) into the conversation thread. The first submission
 * also flips the layout from the home screen to the chat view via the
 * `has-content` class on the app container, with a brief slide animation
 * on the input box (FLIP technique).
 */

document.addEventListener("DOMContentLoaded", () => {

    // ---------- Constants ----------

    const ASK_ENDPOINT = "/ask";
    const SCROLL_SETTLE_MS = 50;
    const HOME_TO_CHAT_TRANSITION_MS = 300;
    const GENERIC_ERROR_MESSAGE =
        "The assistant is temporarily unavailable. Please try again later.";

    // ---------- DOM references ----------

    const appContainer       = document.getElementById("appContainer");
    const chatArea           = document.getElementById("chatArea");
    const userInput          = document.getElementById("userInput");
    const sendBtn            = document.getElementById("sendBtn");
    const userQuestionText   = document.getElementById("userQuestionText");
    const assistantAnswerText = document.getElementById("assistantAnswerText");
    const sourcesContainer   = document.getElementById("sourcesContainer");
    const sourcesList        = document.getElementById("sourcesList");
    const inputContainerBox  = document.querySelector(".input-container-box");
    const inputPositioner    = document.querySelector(".input-box-positioner");

    // ---------- Event bindings ----------

    // Clicking anywhere on the input shell focuses the textarea.
    inputContainerBox.addEventListener("click", () => userInput.focus());

    // Auto-resize the textarea and toggle the send button's active state.
    userInput.addEventListener("input", () => {
        toggleSendButton(userInput.value.trim().length > 0);
        autoResizeTextarea();
    });

    // Enter submits, Shift+Enter inserts a newline.
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (userInput.value.trim().length > 0) {
                submitQuestion();
            }
        }
    });

    sendBtn.addEventListener("click", () => {
        if (userInput.value.trim().length > 0) {
            submitQuestion();
        }
    });

    // ---------- UI helpers ----------

    function toggleSendButton(isActive) {
        sendBtn.classList.toggle("is-active", isActive);
    }

    function autoResizeTextarea() {
        userInput.style.height = "auto";
        userInput.style.height = `${userInput.scrollHeight}px`;
    }

    function resetTextarea() {
        userInput.value = "";
        userInput.style.height = "auto";
        toggleSendButton(false);
    }

    function scrollChatToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function wait(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function nextFrame() {
        return new Promise((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(resolve))
        );
    }

    function renderLoadingState() {
        assistantAnswerText.innerHTML = `
            <div class="status-loading-wrapper">
                <div class="status-message">Seeking sources...</div>
                <div class="loading-skeleton">
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                </div>
            </div>
        `;
    }

    function renderError(message) {
        assistantAnswerText.classList.add("error-text");
        assistantAnswerText.textContent = message;
    }

    function renderAnswer(answer) {
        assistantAnswerText.classList.remove("error-text");
        assistantAnswerText.textContent = answer;
    }

    function renderSources(sources) {
        sourcesList.innerHTML = "";

        if (!sources || sources.length === 0) {
            sourcesContainer.classList.add("is-hidden");
            return;
        }

        sources.forEach((source) => {
            sourcesList.appendChild(buildSourceCard(source));
        });
        sourcesContainer.classList.remove("is-hidden");
    }

    function buildSourceCard(source) {
        const card = document.createElement("a");
        card.className = "source-card";
        card.href = source.url;
        card.target = "_blank";
        card.rel = "noopener noreferrer";
        card.innerHTML = `
            <div class="source-content">
                <div class="source-title">${source.title}</div>
                <div class="source-meta">${source.publication}</div>
            </div>
            <div class="source-icon">
                <span class="material-symbols-rounded">open_in_new</span>
            </div>
        `;
        return card;
    }

    /**
     * Animates the home → chat transition with a FLIP technique.
     *
     *   1. First  — measure the input's current (home) position.
     *   2. Last   — apply `has-content`: the layout snaps to chat mode and
     *               the input is now at the bottom of the screen.
     *   3. Invert — apply an inverse translateY (synchronously) so the
     *               input visually stays at its home position. Force a
     *               reflow so the browser commits this as the starting
     *               point of the upcoming transition.
     *   4. Play   — enable the CSS transition and clear the transform on
     *               the next frame. The input glides down to its docked
     *               position.
     */
    async function playHomeToChatTransition() {
        // First.
        const firstTop = inputPositioner.getBoundingClientRect().top;

        // Last.
        appContainer.classList.add("has-content");
        const lastTop = inputPositioner.getBoundingClientRect().top;
        const deltaY = firstTop - lastTop;

        // Invert (synchronous, no transition yet).
        inputPositioner.style.transform = `translateY(${deltaY}px)`;

        // Force a reflow so the browser commits the inverted position
        // before we enable the transition. Without this, the transform
        // and its removal can be coalesced into a single paint and the
        // animation is skipped entirely.
        void inputPositioner.offsetHeight;

        // Play.
        appContainer.classList.add("is-transitioning");
        await nextFrame();
        inputPositioner.style.transform = "";

        await wait(HOME_TO_CHAT_TRANSITION_MS);

        appContainer.classList.remove("is-transitioning");
    }

    // ---------- Request flow ----------

    async function submitQuestion() {
        const question = userInput.value.trim();
        if (!question) return;

        const isFirstSubmission = !appContainer.classList.contains("has-content");

        // Prepare the conversation content first. While we are still on the
        // home screen, the conversation thread is hidden, so these updates
        // are invisible to the user — they only become visible once the
        // transition swaps the layout.
        sourcesContainer.classList.add("is-hidden");
        sourcesList.innerHTML = "";
        assistantAnswerText.classList.remove("error-text");
        userQuestionText.textContent = question;
        renderLoadingState();
        resetTextarea();

        // First submission: animate the home → chat transition. The input
        // box slides from its centered home position down to its docked
        // position.
        if (isFirstSubmission) {
            await playHomeToChatTransition();
        }

        scrollChatToBottom();

        try {
            const response = await fetch(ASK_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });

            if (!response.ok) {
                throw new Error("Network server error");
            }

            const data = await response.json();

            if (data.error) {
                renderError(data.answer);
            } else {
                renderAnswer(data.answer);
                renderSources(data.sources);
            }
        } catch (error) {
            renderError(GENERIC_ERROR_MESSAGE);
            console.error(error);
        }

        // Settle the scroll position once the answer has rendered.
        setTimeout(scrollChatToBottom, SCROLL_SETTLE_MS);
    }
});
