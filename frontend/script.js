/**
 * Tech News RAG Assistant — frontend script
 *
 * Drives a multi-turn conversation against the FastAPI /ask endpoint.
 *
 * Home screen: the composer sits centred between the title and the
 * suggestions. On the first submission it slides down to the dock (FLIP
 * technique) and the thread takes over. Subsequent answers append to the
 * thread, which scrolls behind the docked composer; its dissolve mask and
 * bottom padding are sized from --composer-h, kept in sync with the dock's
 * real height so the effect holds as the textarea wraps.
 *
 * Clicking the brand resets everything back to the home screen.
 */

document.addEventListener("DOMContentLoaded", () => {

    // ---------- Configuration ----------

    const ASK_ENDPOINT = "/ask";
    const SCROLL_SETTLE_MS = 60;
    const SLIDE_DURATION_MS = 300;
    const GENERIC_ERROR =
        "The assistant is temporarily unavailable. Please try again later.";

    // ---------- DOM references ----------

    const app                = document.getElementById("app");
    const brand              = document.getElementById("brand");
    const thread             = document.getElementById("thread");
    const threadInner        = document.getElementById("threadInner");
    const dock               = document.getElementById("dock");
    const composerPositioner = document.getElementById("composerPositioner");
    const composer           = document.getElementById("composer");
    const suggestions        = document.getElementById("suggestions");
    const input              = document.getElementById("input");
    const sendButton         = document.getElementById("send");
    const scrollbar          = document.getElementById("scrollbar");
    const scrollbarThumb     = document.getElementById("scrollbarThumb");

    let isAwaitingResponse = false;

    // ---------- Composer height tracking ----------

    const syncComposerHeight = () => {
        if (app.dataset.state !== "chat") return;
        const height = Math.round(dock.getBoundingClientRect().height);
        app.style.setProperty("--composer-h", `${height}px`);
        updateScrollbar();
    };

    new ResizeObserver(syncComposerHeight).observe(dock);

    // ---------- Overlay scrollbar ----------

    const MIN_THUMB_HEIGHT = 32;

    // Geometry is measured only when it can actually change (resize / content
    // growth). During scroll and drag we read just scrollTop — a cheap read
    // that triggers no layout — so the thumb never lags behind.
    let geom = { travel: 0, scrollable: 0 };
    let thumbFrame = null;

    /** Recompute thumb size and visibility; call on resize and after renders. */
    function updateScrollbar() {
        const { scrollHeight, clientHeight } = thread;
        const scrollable = scrollHeight - clientHeight;
        const track = scrollbar.clientHeight;

        if (scrollable <= 1 || track <= 0) {
            scrollbar.classList.remove("is-scrollable");
            geom = { travel: 0, scrollable: 0 };
            return;
        }

        scrollbar.classList.add("is-scrollable");
        const thumbHeight = Math.max(
            MIN_THUMB_HEIGHT,
            (clientHeight / scrollHeight) * track
        );
        geom = { travel: track - thumbHeight, scrollable };
        scrollbarThumb.style.height = `${thumbHeight}px`;
        renderThumb();
    }

    /** Position the thumb from the current scrollTop (no layout reads). */
    function renderThumb() {
        if (geom.scrollable <= 0) return;
        const top = (thread.scrollTop / geom.scrollable) * geom.travel;
        scrollbarThumb.style.transform = `translateY(${top}px)`;
    }

    // Coalesce scroll updates into a single frame.
    thread.addEventListener(
        "scroll",
        () => {
            if (thumbFrame) return;
            thumbFrame = requestAnimationFrame(() => {
                thumbFrame = null;
                renderThumb();
            });
        },
        { passive: true }
    );

    new ResizeObserver(updateScrollbar).observe(thread);
    new ResizeObserver(updateScrollbar).observe(threadInner);

    // Drag the thumb to scroll. Smooth scrolling is disabled for the duration
    // so scrollTop tracks the pointer instantly, then restored on release.
    let isDragging = false;
    let dragStartY = 0;
    let dragStartScroll = 0;

    scrollbarThumb.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        isDragging = true;
        dragStartY = event.clientY;
        dragStartScroll = thread.scrollTop;
        scrollbarThumb.classList.add("is-dragging");
        scrollbarThumb.setPointerCapture(event.pointerId);
        thread.style.scrollBehavior = "auto";
    });

    scrollbarThumb.addEventListener("pointermove", (event) => {
        if (!isDragging) return;
        const ratio = geom.travel > 0 ? (event.clientY - dragStartY) / geom.travel : 0;
        thread.scrollTop = dragStartScroll + ratio * geom.scrollable;
        renderThumb();
    });

    scrollbarThumb.addEventListener("pointerup", (event) => {
        isDragging = false;
        scrollbarThumb.classList.remove("is-dragging");
        scrollbarThumb.releasePointerCapture(event.pointerId);
        thread.style.scrollBehavior = "";
    });

    // ---------- Event bindings ----------

    composer.addEventListener("click", () => input.focus());

    input.addEventListener("input", () => {
        autoResize();
        refreshSendButton();
    });

    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
        }
    });

    sendButton.addEventListener("click", submit);

    suggestions.addEventListener("click", (event) => {
        const chip = event.target.closest(".chip");
        if (!chip) return;
        input.value = chip.textContent;
        autoResize();
        refreshSendButton();
        submit();
    });

    brand.addEventListener("click", resetConversation);

    // ---------- Composer helpers ----------

    function hasText() {
        return input.value.trim().length > 0;
    }

    /** Toggle the send button's active state — the original behaviour. */
    function refreshSendButton() {
        sendButton.classList.toggle("is-active", hasText());
    }

    function autoResize() {
        input.style.height = "auto";
        input.style.height = `${input.scrollHeight}px`;
    }

    function clearComposer() {
        input.value = "";
        input.style.height = "auto";
        refreshSendButton();
    }

    // ---------- Scrolling ----------

    // Distance from the thread's top at which a freshly sent question should
    // settle — just below the (overlay) top bar, matching the thread's own
    // top padding (header height + breathing room).
    const QUESTION_TOP_OFFSET = 72;

    /**
     * Bring a message to the top of the viewport so the answer reads from
     * just beneath it. If there isn't enough content below to push it all the
     * way up, the browser clamps the scroll — exactly like the big chatbots.
     */
    function scrollQuestionToTop(messageEl) {
        const delta =
            messageEl.getBoundingClientRect().top -
            thread.getBoundingClientRect().top -
            QUESTION_TOP_OFFSET;
        thread.scrollTo({ top: thread.scrollTop + delta, behavior: "smooth" });
    }

    function nextFrame() {
        return new Promise((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(resolve))
        );
    }

    function wait(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    /**
     * Slide the composer from its centred home position down to the dock,
     * using FLIP: measure first, switch layout, invert with a transform, then
     * play by clearing it on the next frame.
     */
    async function playHomeToChatTransition() {
        const firstTop = composerPositioner.getBoundingClientRect().top;

        app.dataset.state = "chat";
        const lastTop = composerPositioner.getBoundingClientRect().top;

        composerPositioner.style.transform = `translateY(${firstTop - lastTop}px)`;
        void composerPositioner.offsetHeight; // commit the inverted position

        app.classList.add("is-transitioning");
        await nextFrame();
        composerPositioner.style.transform = "";

        await wait(SLIDE_DURATION_MS);
        app.classList.remove("is-transitioning");
    }

    // ---------- Rendering ----------

    /**
     * Escape HTML so untrusted text is safe to inject via innerHTML. Must run
     * before formatInline, otherwise the markdown step would be defeated.
     */
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    /** Convert the supported markdown subset (**bold** only) to inline HTML. */
    function formatInline(text) {
        return text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    }

    function appendUserMessage(question) {
        const wrapper = document.createElement("div");
        wrapper.className = "message message-user";

        const bubble = document.createElement("div");
        bubble.className = "user-bubble";
        bubble.textContent = question;

        wrapper.appendChild(bubble);
        threadInner.appendChild(wrapper);
        return wrapper;
    }

    /**
     * Append an assistant turn in its loading state and return the answer
     * element, so the caller can swap in the response once it arrives.
     */
    function appendAssistantMessage() {
        const wrapper = document.createElement("div");
        wrapper.className = "message message-assistant";

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerHTML =
            '<span class="material-symbols-rounded">subtitles</span>';

        const answer = document.createElement("div");
        answer.className = "answer";
        answer.innerHTML = `
            <div class="thinking">
                <div class="thinking-label">Seeking sources&hellip;</div>
                <div class="skeleton">
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                </div>
            </div>
        `;

        wrapper.append(avatar, answer);
        threadInner.appendChild(wrapper);
        return answer;
    }

    function renderAnswer(answerEl, text) {
        answerEl.classList.remove("is-error");

        // Defensive: an empty (but non-error) payload would otherwise wipe the
        // skeleton and leave a blank turn. Fall back to a clear message.
        if (!text || !text.trim()) {
            renderError(answerEl, GENERIC_ERROR);
            return;
        }

        answerEl.innerHTML = "";

        text.split(/\n\n+/).forEach((block) => {
            const trimmed = block.trim();
            if (!trimmed) return;
            const p = document.createElement("p");
            p.innerHTML = formatInline(escapeHtml(trimmed));
            answerEl.appendChild(p);
        });
    }

    function renderError(answerEl, message) {
        answerEl.classList.add("is-error");
        answerEl.textContent = message;
    }

    function renderSources(answerEl, sources) {
        if (!Array.isArray(sources) || sources.length === 0) return;

        const container = document.createElement("div");
        container.className = "sources";
        container.innerHTML = '<div class="sources-title">SOURCES</div>';

        const list = document.createElement("div");
        list.className = "sources-list";
        sources.forEach((source) => list.appendChild(buildSourceCard(source)));

        container.appendChild(list);
        answerEl.appendChild(container);
    }

    function buildSourceCard(source) {
        const card = document.createElement("a");
        card.className = "source-card";
        card.href = source.url;
        card.target = "_blank";
        card.rel = "noopener noreferrer";

        const text = document.createElement("div");
        text.className = "source-text";

        const title = document.createElement("div");
        title.className = "source-title";
        title.textContent = source.title;

        const meta = document.createElement("div");
        meta.className = "source-meta";
        meta.textContent = source.publication;

        text.append(title, meta);

        const icon = document.createElement("div");
        icon.className = "source-icon";
        icon.innerHTML =
            '<span class="material-symbols-rounded">open_in_new</span>';

        card.append(text, icon);
        return card;
    }

    // ---------- Conversation flow ----------

    async function submit() {
        if (isAwaitingResponse || !hasText()) return;

        const question = input.value.trim();
        const isFirstSubmission = app.dataset.state === "empty";
        isAwaitingResponse = true;

        // Build the turn while still on the home screen (the thread is hidden,
        // so these updates only become visible after the slide).
        const userMessage = appendUserMessage(question);
        const answerEl = appendAssistantMessage();
        clearComposer();
        input.focus();

        if (isFirstSubmission) {
            await playHomeToChatTransition();
        }

        syncComposerHeight();
        scrollQuestionToTop(userMessage);

        try {
            const response = await fetch(ASK_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });

            if (!response.ok) throw new Error("Server error");

            const data = await response.json();

            if (data.error) {
                renderError(answerEl, data.answer || GENERIC_ERROR);
            } else {
                renderAnswer(answerEl, data.answer);
                renderSources(answerEl, data.sources);
            }
        } catch (error) {
            renderError(answerEl, GENERIC_ERROR);
            console.error(error);
        } finally {
            isAwaitingResponse = false;
            setTimeout(() => {
                scrollQuestionToTop(userMessage);
                updateScrollbar();
            }, SCROLL_SETTLE_MS);
        }
    }

    /** Reset to the home screen — the brand acts as a fresh start, but only
     *  while a conversation is open. On the home screen it does nothing. */
    function resetConversation() {
        if (app.dataset.state !== "chat") return;

        threadInner
            .querySelectorAll(".message")
            .forEach((node) => node.remove());

        app.dataset.state = "empty";
        app.style.removeProperty("--composer-h");
        composerPositioner.style.transform = "";
        thread.scrollTop = 0;
        updateScrollbar();

        clearComposer();
        input.focus();
    }
});
