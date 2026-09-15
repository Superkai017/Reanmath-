const chatEl = document.getElementById("chat");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");

let history = [];

function addMessage(role, text, { pending = false, error = false } = {}) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg--${role}${pending ? " msg--pending" : ""}${error ? " msg--error" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.textContent = text;

  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return wrap;
}

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}
inputEl.addEventListener("input", autoGrow);

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  addMessage("user", message);
  inputEl.value = "";
  autoGrow();
  sendEl.disabled = true;

  const pendingNode = addMessage("bot", "កំពុងគិត...", { pending: true });

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `សំណើបរាជ័យ (${res.status})`);
    }

    const data = await res.json();
    history = data.history;
    pendingNode.remove();
    addMessage("bot", data.reply);
  } catch (err) {
    pendingNode.remove();
    addMessage("bot", `មានបញ្ហា៖ ${err.message}`, { error: true });
  } finally {
    sendEl.disabled = false;
    inputEl.focus();
  }
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});
