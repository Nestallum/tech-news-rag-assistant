"use strict";

const app = document.querySelector(".app");
const thread = document.getElementById("thread");
const textarea = document.getElementById("question");
const sendBtn = document.getElementById("send");
const examples = document.getElementById("examples");

const LOADING_STEPS = [
  "Searching the archive",
  "Reading the sources",
  "Writing the answer",
];

/** Escape text before inserting it as HTML. */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/** Auto-grow the textarea with its content. */
function autoGrow() {
  textarea.style.height = "auto";
  textarea.style.height = textarea.scrollHeight + "px";
}

/** Append the user's question to the thread. */
function addUserMessage(question) {
  const el = document.createElement("div");
  el.className = "user-msg";
  el.textContent = question;
  thread.appendChild(el);
}

/** Append a loading placeholder and animate the step text. Returns a stop fn. */
function addLoading() {
  const el = document.createElement("div");
  el.className = "loading";
  el.textContent = LOADING_STEPS[0];
  thread.appendChild(el);

  let step = 0;
  const timer = setInterval(() => {
    step = Math.min(step + 1, LOADING_STEPS.length - 1);
    el.textContent = LOADING_STEPS[step];
  }, 6000);

  return { el, stop: () => clearInterval(timer) };
}

/** Render the answer (and sources) into the given element. */
function renderAnswer(el, data) {
  el.className = data.error ? "answer error" : "answer";
  el.textContent = data.answer;

  if (data.sources && data.sources.length > 0) {
    const box = document.createElement("div");
    box.className = "sources";

    const label = document.createElement("div");
    label.className = "sources-label";
    label.textContent = "Sources";
    box.appendChild(label);

    data.sources.forEach((s) => {
      const a = document.createElement("a");
      a.className = "source";
      a.href = s.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.innerHTML =
        '<div class="source-title">' +
        escapeHtml(s.title) +
        '</div><div class="source-pub">' +
        escapeHtml(s.publication) +
        "</div>";
      box.appendChild(a);
    });

    el.appendChild(box);
  }
}

/** Send a question to the API and display the result. */
async function ask(question) {
  question = question.trim();
  if (!question) return;

  // Option B: each question replaces the previous one.
  thread.innerHTML = "";
  app.classList.add("answered");
  textarea.value = "";
  autoGrow();
  sendBtn.disabled = true;

  addUserMessage(question);
  const loading = addLoading();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });
    const data = await res.json();
    loading.stop();
    renderAnswer(loading.el, data);
  } catch (err) {
    loading.stop();
    renderAnswer(loading.el, {
      answer: "Could not reach the assistant. Please try again.",
      sources: [],
      error: true,
    });
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", () => ask(textarea.value));

textarea.addEventListener("input", autoGrow);

textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask(textarea.value);
  }
});

examples.addEventListener("click", (e) => {
  if (e.target.classList.contains("example")) {
    ask(e.target.textContent);
  }
});
