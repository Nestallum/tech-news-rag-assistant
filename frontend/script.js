document.addEventListener("DOMContentLoaded", () => {
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const conversationContainer = document.getElementById("conversationContainer");
    const userQuestionText = document.getElementById("userQuestionText");
    const assistantAnswerText = document.getElementById("assistantAnswerText");
    const sourcesContainer = document.getElementById("sourcesContainer");
    const sourcesList = document.getElementById("sourcesList");
    const chatArea = document.getElementById("chatArea");
    const appContainer = document.getElementById("appContainer");

    const inputContainer = document.querySelector(".input-container-box");
    inputContainer.addEventListener("click", () => {
        userInput.focus();
    });

    // --- Dynamic Textarea Sizing & Send Button States ---
    userInput.addEventListener("input", () => {
        const hasText = userInput.value.trim().length > 0;

        if (hasText) {
            sendBtn.classList.add("is-active");
        } else {
            sendBtn.classList.remove("is-active");
        }

        userInput.style.height = "auto";
        userInput.style.height = `${userInput.scrollHeight}px`;
    });

    // --- Keyboard Event Interceptor (Enter without Shift) ---
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (userInput.value.trim().length > 0) {
                submitQuestion();
            }
        }
    });

    // --- Persistent Button Interaction Event ---
    sendBtn.addEventListener("click", () => {
        if (userInput.value.trim().length > 0) {
            submitQuestion();
        }
    });

    // --- RAG Request Engine Execution ---
    async function submitQuestion() {
        const question = userInput.value.trim();
        if (!question) return;

        // Switches layout from centered home to conversation flow
        appContainer.classList.add("has-content");

        // Reset previous answer artefacts
        sourcesContainer.style.display = "none";
        sourcesList.innerHTML = "";
        assistantAnswerText.classList.remove("error-text");

        // Push question content into the right-hand bubble context
        userQuestionText.textContent = question;

        // Render response status and animated loading lines
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

        // Recycle textarea components
        userInput.value = "";
        userInput.style.height = "auto";
        sendBtn.classList.remove("is-active");
        chatArea.scrollTop = chatArea.scrollHeight;

        try {
            const response = await fetch("/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                throw new Error("Network server error");
            }

            const data = await response.json();

            if (data.error) {
                assistantAnswerText.classList.add("error-text");
                assistantAnswerText.textContent = data.answer;
            } else {
                assistantAnswerText.textContent = data.answer;

                // Create full width cards block
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach(source => {
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
                        sourcesList.appendChild(card);
                    });
                    sourcesContainer.style.display = "block";
                }
            }

        } catch (error) {
            assistantAnswerText.classList.add("error-text");
            assistantAnswerText.textContent = "The assistant is temporarily unavailable. Please try again later.";
            console.error(error);
        }

        // Align page anchor to capture full contents
        setTimeout(() => {
            chatArea.scrollTop = chatArea.scrollHeight;
        }, 50);
    }
});


