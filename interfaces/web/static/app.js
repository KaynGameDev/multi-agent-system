const state = {
  activeConversationId: null,
  activeConversationTitle: "New chat",
  contextMenuConversationId: null,
  contextMenuConversationTitle: "New chat",
  conversations: [],
  sending: false,
};

const elements = {
  conversationList: document.getElementById("conversationList"),
  chatStage: document.getElementById("chatStage"),
  emptyState: document.getElementById("emptyState"),
  messages: document.getElementById("messages"),
  composerForm: document.getElementById("composerForm"),
  messageInput: document.getElementById("messageInput"),
  sendButton: document.getElementById("sendButton"),
  newChatButton: document.getElementById("newChatButton"),
  displayNameInput: document.getElementById("displayNameInput"),
  emailInput: document.getElementById("emailInput"),
  mobileMenuButton: document.getElementById("mobileMenuButton"),
  sidebar: document.getElementById("sidebar"),
  sidebarOverlay: document.getElementById("sidebarOverlay"),
  activeConversationTitle: document.getElementById("activeConversationTitle"),
  renameConversationButton: document.getElementById("renameConversationButton"),
  conversationContextMenu: document.getElementById("conversationContextMenu"),
  contextMenuRenameButton: document.getElementById("contextMenuRenameButton"),
};

const PROFILE_STORAGE_KEY = "jade-web-profile";
const ACTIVE_CONVERSATION_STORAGE_KEY = "jade-active-conversation";

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed.");
  }

  return response.json();
}

function restoreProfile() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(PROFILE_STORAGE_KEY) || "{}");
    elements.displayNameInput.value = saved.display_name || "";
    elements.emailInput.value = saved.email || "";
  } catch (_error) {
    elements.displayNameInput.value = "";
    elements.emailInput.value = "";
  }
}

function saveProfile() {
  const payload = {
    display_name: elements.displayNameInput.value.trim(),
    email: elements.emailInput.value.trim(),
  };
  window.localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(payload));
}

function autoResizeTextarea() {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 224)}px`;
}

function formatRelativeTime(isoString) {
  const timestamp = new Date(isoString);
  if (Number.isNaN(timestamp.getTime())) {
    return "";
  }

  const diff = Date.now() - timestamp.getTime();
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diff < hour) {
    return `${Math.max(1, Math.round(diff / minute))}m`;
  }
  if (diff < day) {
    return `${Math.round(diff / hour)}h`;
  }
  return timestamp.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function startOfLocalDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function getConversationGroupLabel(isoString) {
  const timestamp = new Date(isoString);
  if (Number.isNaN(timestamp.getTime())) {
    return "Older";
  }

  const today = startOfLocalDay(new Date());
  const targetDay = startOfLocalDay(timestamp);
  const dayDiff = Math.round((today.getTime() - targetDay.getTime()) / (24 * 60 * 60 * 1000));

  if (dayDiff <= 0) {
    return "Today";
  }
  if (dayDiff === 1) {
    return "Yesterday";
  }
  if (dayDiff < 7) {
    return "Previous 7 Days";
  }
  if (dayDiff < 30) {
    return "Previous 30 Days";
  }
  return "Older";
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function closeSidebar() {
  elements.sidebar.classList.remove("is-open");
  elements.sidebarOverlay.classList.remove("is-open");
}

function openSidebar() {
  elements.sidebar.classList.add("is-open");
  elements.sidebarOverlay.classList.add("is-open");
}

function closeConversationContextMenu() {
  state.contextMenuConversationId = null;
  state.contextMenuConversationTitle = "New chat";
  elements.conversationContextMenu.hidden = true;
}

function openConversationContextMenu(conversationId, conversationTitle, clientX, clientY) {
  state.contextMenuConversationId = conversationId;
  state.contextMenuConversationTitle = conversationTitle || "New chat";

  const menu = elements.conversationContextMenu;
  menu.hidden = false;

  const menuWidth = menu.offsetWidth || 180;
  const menuHeight = menu.offsetHeight || 48;
  const maxLeft = Math.max(12, window.innerWidth - menuWidth - 12);
  const maxTop = Math.max(12, window.innerHeight - menuHeight - 12);
  const left = Math.min(clientX, maxLeft);
  const top = Math.min(clientY, maxTop);

  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
}

function updateConversationHeader(title) {
  const normalizedTitle = (title || "New chat").trim() || "New chat";
  state.activeConversationTitle = normalizedTitle;
  elements.activeConversationTitle.textContent = normalizedTitle;
  elements.renameConversationButton.disabled = !state.activeConversationId;
}

async function renameConversation(conversationId, currentTitle) {
  const nextTitle = window.prompt("Rename chat", currentTitle || "New chat");
  if (nextTitle === null) {
    return;
  }

  const normalizedTitle = nextTitle.trim() || "New chat";
  if (normalizedTitle === (currentTitle || "New chat")) {
    return;
  }

  await api(`/api/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify({ title: normalizedTitle }),
  });
  if (conversationId === state.activeConversationId) {
    updateConversationHeader(normalizedTitle);
  }
  await refreshConversationList();
}

function renderConversationList() {
  elements.conversationList.innerHTML = "";
  closeConversationContextMenu();

  const orderedGroups = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Older"];
  const groups = new Map(orderedGroups.map((label) => [label, []]));

  for (const conversation of state.conversations) {
    const label = getConversationGroupLabel(conversation.updated_at);
    groups.get(label).push(conversation);
  }

  let renderedCount = 0;
  for (const label of orderedGroups) {
    const conversations = groups.get(label) || [];
    if (!conversations.length) {
      continue;
    }

    const group = document.createElement("section");
    group.className = "conversation-group";

    const heading = document.createElement("div");
    heading.className = "conversation-group__label";
    heading.textContent = label;
    group.appendChild(heading);

    for (const conversation of conversations) {
      const row = document.createElement("div");
      row.className = "conversation-list__row";

      const button = document.createElement("button");
      button.type = "button";
      button.className = "conversation-list__item";
      if (conversation.conversation_id === state.activeConversationId) {
        button.classList.add("is-active");
      }
      button.innerHTML = `
        <span class="conversation-list__title">${escapeHtml(conversation.title || "New chat")}</span>
        <span class="conversation-list__time">${escapeHtml(formatRelativeTime(conversation.updated_at))}</span>
      `;
      button.title = "Right-click for chat actions";
      button.addEventListener("click", async () => {
        await loadConversation(conversation.conversation_id);
        closeSidebar();
      });
      row.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        openConversationContextMenu(
          conversation.conversation_id,
          conversation.title || "New chat",
          event.clientX,
          event.clientY,
        );
      });

      row.append(button);
      group.appendChild(row);
      renderedCount += 1;
    }

    elements.conversationList.appendChild(group);
  }

  if (!renderedCount) {
    const empty = document.createElement("div");
    empty.className = "conversation-list__empty";
    empty.textContent = "No saved chats yet.";
    elements.conversationList.appendChild(empty);
  }
}

function renderMessages(conversation, routeMeta = null) {
  const messages = conversation.messages || [];
  elements.messages.innerHTML = "";

  if (!messages.length) {
    elements.emptyState.classList.remove("is-hidden");
    return;
  }

  elements.emptyState.classList.add("is-hidden");
  messages.forEach((message, index) => {
    const wrapper = document.createElement("article");
    wrapper.className = `message message--${message.role}`;

    const meta = document.createElement("div");
    meta.className = "message__meta";
    meta.textContent = message.role === "user" ? "You" : "Jade Agent";

    const card = document.createElement("div");
    card.className = "message__card";
    card.innerHTML = message.html;
    wrapper.append(meta, card);

    if (
      routeMeta &&
      index === messages.length - 1 &&
      message.role === "assistant" &&
      routeMeta.route_reason
    ) {
      const route = document.createElement("div");
      route.className = "message__route";
      route.innerHTML = `<strong>${escapeHtml(routeMeta.route || "Route")}:</strong> ${escapeHtml(routeMeta.route_reason)}`;
      wrapper.appendChild(route);
    }

    elements.messages.appendChild(wrapper);
  });

  requestAnimationFrame(() => {
    elements.chatStage.scrollTop = elements.chatStage.scrollHeight;
  });
}

async function refreshConversationList() {
  const payload = await api("/api/conversations");
  state.conversations = payload.conversations || [];
  renderConversationList();
}

async function createConversation() {
  const conversation = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ title: "New chat" }),
  });
  window.localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, conversation.conversation_id);
  await refreshConversationList();
  await loadConversation(conversation.conversation_id, { skipListRefresh: true });
}

async function loadConversation(conversationId, options = {}) {
  const conversation = await api(`/api/conversations/${conversationId}`);
  state.activeConversationId = conversation.conversation_id;
  window.localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, conversation.conversation_id);
  updateConversationHeader(conversation.title || "New chat");
  renderMessages(conversation);

  if (!options.skipListRefresh) {
    await refreshConversationList();
  } else {
    renderConversationList();
  }
}

function appendPendingAssistant() {
  const wrapper = document.createElement("article");
  wrapper.className = "message message--assistant";
  wrapper.id = "pendingAssistant";
  wrapper.innerHTML = `
    <div class="message__meta">Jade Agent</div>
    <div class="message__card"><p>Thinking…</p></div>
  `;
  elements.messages.appendChild(wrapper);
  elements.emptyState.classList.add("is-hidden");
  requestAnimationFrame(() => {
    elements.chatStage.scrollTop = elements.chatStage.scrollHeight;
  });
}

async function handleSubmit(event) {
  event.preventDefault();
  if (state.sending) {
    return;
  }

  const message = elements.messageInput.value.trim();
  if (!message) {
    return;
  }

  if (!state.activeConversationId) {
    await createConversation();
  }

  state.sending = true;
  elements.sendButton.disabled = true;
  saveProfile();

  const optimisticConversation = await api(`/api/conversations/${state.activeConversationId}`);
  optimisticConversation.messages.push({
    id: crypto.randomUUID(),
    role: "user",
    html: `<p>${escapeHtml(message)}</p>`,
    markdown: message,
    created_at: new Date().toISOString(),
  });
  renderMessages(optimisticConversation);
  appendPendingAssistant();
  elements.messageInput.value = "";
  autoResizeTextarea();

  try {
    const payload = await api(`/api/conversations/${state.activeConversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        message,
        display_name: elements.displayNameInput.value.trim(),
        email: elements.emailInput.value.trim(),
      }),
    });
    renderMessages(payload, {
      route: payload.route,
      route_reason: payload.route_reason,
    });
    await refreshConversationList();
  } catch (error) {
    const pending = document.getElementById("pendingAssistant");
    if (pending) {
      pending.remove();
    }
    const failureConversation = await api(`/api/conversations/${state.activeConversationId}`);
    failureConversation.messages.push({
      id: crypto.randomUUID(),
      role: "assistant",
      html: `<p>${escapeHtml(error.message)}</p>`,
      markdown: error.message,
      created_at: new Date().toISOString(),
    });
    renderMessages(failureConversation);
  } finally {
    state.sending = false;
    elements.sendButton.disabled = false;
    elements.messageInput.focus();
  }
}

async function bootstrap() {
  restoreProfile();
  autoResizeTextarea();
  updateConversationHeader("New chat");
  await refreshConversationList();

  const savedConversationId = window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
  const hasSavedConversation = state.conversations.some(
    (conversation) => conversation.conversation_id === savedConversationId,
  );

  if (hasSavedConversation) {
    await loadConversation(savedConversationId, { skipListRefresh: true });
  } else if (state.conversations.length > 0) {
    await loadConversation(state.conversations[0].conversation_id, { skipListRefresh: true });
  } else {
    await createConversation();
  }
}

elements.newChatButton.addEventListener("click", async () => {
  await createConversation();
  closeSidebar();
  elements.messageInput.focus();
});
elements.composerForm.addEventListener("submit", handleSubmit);
elements.messageInput.addEventListener("input", autoResizeTextarea);
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composerForm.requestSubmit();
  }
});
elements.displayNameInput.addEventListener("change", saveProfile);
elements.emailInput.addEventListener("change", saveProfile);
elements.mobileMenuButton.addEventListener("click", openSidebar);
elements.sidebarOverlay.addEventListener("click", closeSidebar);
document.addEventListener("click", (event) => {
  if (elements.conversationContextMenu.hidden) {
    return;
  }
  if (!elements.conversationContextMenu.contains(event.target)) {
    closeConversationContextMenu();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeConversationContextMenu();
  }
});
window.addEventListener("blur", closeConversationContextMenu);
window.addEventListener("resize", closeConversationContextMenu);
elements.renameConversationButton.addEventListener("click", async () => {
  if (!state.activeConversationId) {
    return;
  }

  try {
    await renameConversation(state.activeConversationId, state.activeConversationTitle);
  } catch (error) {
    window.alert(error.message || "Could not rename that chat.");
  }
});
elements.contextMenuRenameButton.addEventListener("click", async () => {
  if (!state.contextMenuConversationId) {
    return;
  }

  const conversationId = state.contextMenuConversationId;
  const conversationTitle = state.contextMenuConversationTitle;
  closeConversationContextMenu();

  try {
    await renameConversation(conversationId, conversationTitle);
  } catch (error) {
    window.alert(error.message || "Could not rename that chat.");
  }
});

document.querySelectorAll(".starter-card").forEach((button) => {
  button.addEventListener("click", () => {
    elements.messageInput.value = button.dataset.prompt || "";
    autoResizeTextarea();
    elements.messageInput.focus();
  });
});

bootstrap().catch((error) => {
  elements.emptyState.classList.remove("is-hidden");
  elements.messages.innerHTML = `
    <article class="message message--assistant">
      <div class="message__meta">Jade Agent</div>
      <div class="message__card"><p>${escapeHtml(error.message)}</p></div>
    </article>
  `;
});
