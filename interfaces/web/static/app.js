const state = {
  activeConversationId: null,
  activeConversationTitle: "New chat",
  contextMenuConversationId: null,
  contextMenuConversationTitle: "New chat",
  conversations: [],
  sending: false,
  authEnabled: false,
  authenticatedUsername: "",
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
  conversationContextMenu: document.getElementById("conversationContextMenu"),
  contextMenuRenameButton: document.getElementById("contextMenuRenameButton"),
  contextMenuDeleteButton: document.getElementById("contextMenuDeleteButton"),
  sidebarAuth: document.getElementById("sidebarAuth"),
  authUsername: document.getElementById("authUsername"),
  logoutButton: document.getElementById("logoutButton"),
};

const PROFILE_STORAGE_KEY = "jade-web-profile";
const ACTIVE_CONVERSATION_STORAGE_KEY = "jade-active-conversation";
const CONTEXT_MENU_DEBUG_VERSION = "2026-04-09-ctx-debug-1";
const CONTEXT_MENU_DEBUG_ENABLED = new URLSearchParams(window.location.search).has("debugContextMenu");

let contextMenuDebugPanel = null;

function describeDebugTarget(target) {
  if (!(target instanceof Element)) {
    return String(target);
  }

  const tagName = target.tagName.toLowerCase();
  const idPart = target.id ? `#${target.id}` : "";
  const classPart = target.classList.length ? `.${Array.from(target.classList).join(".")}` : "";
  return `${tagName}${idPart}${classPart}`;
}

function ensureContextMenuDebugPanel() {
  if (!CONTEXT_MENU_DEBUG_ENABLED || contextMenuDebugPanel) {
    return contextMenuDebugPanel;
  }

  const panel = document.createElement("aside");
  panel.className = "debug-panel";
  panel.innerHTML = `
    <div class="debug-panel__header">
      <strong>Context Menu Debug</strong>
      <span>${escapeHtml(CONTEXT_MENU_DEBUG_VERSION)}</span>
    </div>
    <div class="debug-panel__body" id="contextMenuDebugBody"></div>
  `;
  document.body.appendChild(panel);
  contextMenuDebugPanel = panel.querySelector("#contextMenuDebugBody");
  return contextMenuDebugPanel;
}

function logContextMenuDebug(label, event = null, details = {}) {
  if (!CONTEXT_MENU_DEBUG_ENABLED) {
    return;
  }

  const payload = {
    label,
    ...(event
      ? {
          type: event.type,
          button: "button" in event ? event.button : undefined,
          buttons: "buttons" in event ? event.buttons : undefined,
          clientX: "clientX" in event ? event.clientX : undefined,
          clientY: "clientY" in event ? event.clientY : undefined,
          defaultPrevented: event.defaultPrevented,
          target: describeDebugTarget(event.target),
        }
      : {}),
    ...details,
  };

  const timestamp = new Date().toLocaleTimeString();
  console.info("[ctx-debug]", CONTEXT_MENU_DEBUG_VERSION, payload);

  const body = ensureContextMenuDebugPanel();
  if (!body) {
    return;
  }

  const row = document.createElement("div");
  row.className = "debug-panel__entry";
  row.textContent = `${timestamp} ${JSON.stringify(payload)}`;
  body.prepend(row);

  while (body.childElementCount > 30) {
    body.lastElementChild?.remove();
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (response.status === 401) {
    window.location.href = "/login";
    throw new Error("Your session has expired. Please sign in again.");
  }

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed.");
  }

  return response.json();
}

function syncAuthUi(session) {
  state.authEnabled = Boolean(session && session.enabled);
  state.authenticatedUsername = session && session.authenticated ? (session.username || "") : "";

  elements.sidebarAuth.hidden = !state.authEnabled;
  if (state.authEnabled) {
    elements.authUsername.textContent = state.authenticatedUsername
      ? `Signed in as ${state.authenticatedUsername}`
      : "Protected access";
  }
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

function isConversationContextMenuOpen() {
  return !elements.conversationContextMenu.hidden;
}

function closeConversationContextMenu() {
  logContextMenuDebug("menu-close", null, {
    conversationId: state.contextMenuConversationId,
    open: isConversationContextMenuOpen(),
  });
  state.contextMenuConversationId = null;
  state.contextMenuConversationTitle = "New chat";
  elements.conversationContextMenu.hidden = true;
  elements.conversationContextMenu.style.left = "";
  elements.conversationContextMenu.style.top = "";
}

function openConversationContextMenu(conversationId, conversationTitle, clientX, clientY) {
  state.contextMenuConversationId = conversationId;
  state.contextMenuConversationTitle = conversationTitle || "New chat";

  const menu = elements.conversationContextMenu;
  menu.hidden = false;
  void menu.offsetHeight;

  const menuWidth = menu.offsetWidth;
  const menuHeight = menu.offsetHeight;
  const maxLeft = Math.max(12, window.innerWidth - menuWidth - 12);
  const maxTop = Math.max(12, window.innerHeight - menuHeight - 12);
  const left = Math.min(clientX, maxLeft);
  const top = Math.min(clientY, maxTop);

  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
  logContextMenuDebug("menu-open", null, {
    conversationId,
    conversationTitle: conversationTitle || "New chat",
    requestedX: clientX,
    requestedY: clientY,
    left,
    top,
  });
}

function updateConversationHeader(title) {
  const normalizedTitle = (title || "New chat").trim() || "New chat";
  state.activeConversationTitle = normalizedTitle;
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

async function deleteConversation(conversationId, currentTitle) {
  const normalizedTitle = (currentTitle || "New chat").trim() || "New chat";
  const confirmed = window.confirm(`Delete "${normalizedTitle}"? This cannot be undone.`);
  if (!confirmed) {
    return;
  }

  const wasActiveConversation = conversationId === state.activeConversationId;
  await api(`/api/conversations/${conversationId}`, { method: "DELETE" });

  if (wasActiveConversation) {
    state.activeConversationId = null;
    window.localStorage.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
    updateConversationHeader("New chat");
    renderMessages({ messages: [] });
  }

  await refreshConversationList();

  if (!wasActiveConversation) {
    return;
  }

  if (state.conversations.length > 0) {
    await loadConversation(state.conversations[0].conversation_id, { skipListRefresh: true });
    return;
  }

  await createConversation();
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
      row.dataset.conversationId = conversation.conversation_id;
      row.dataset.conversationTitle = conversation.title || "New chat";
      const openActionsMenuFromPointer = (event) => {
        logContextMenuDebug("row-open-request", event, {
          conversationId: conversation.conversation_id,
          source: describeDebugTarget(event.currentTarget),
        });
        event.preventDefault();
        event.stopPropagation();
        openConversationContextMenu(
          conversation.conversation_id,
          conversation.title || "New chat",
          event.clientX,
          event.clientY,
        );
      };

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
      button.addEventListener("keydown", (event) => {
        if (event.key !== "ContextMenu" && !(event.shiftKey && event.key === "F10")) {
          return;
        }

        logContextMenuDebug("button-keydown-open-request", event, {
          conversationId: conversation.conversation_id,
          key: event.key,
          shiftKey: event.shiftKey,
        });
        event.preventDefault();
        const rect = button.getBoundingClientRect();
        openConversationContextMenu(
          conversation.conversation_id,
          conversation.title || "New chat",
          rect.right - 8,
          rect.bottom + 6,
        );
      });
      button.addEventListener("contextmenu", openActionsMenuFromPointer);

      row.addEventListener("contextmenu", (event) => {
        logContextMenuDebug("row-contextmenu", event, {
          conversationId: conversation.conversation_id,
          hitRowDirectly: event.target === row,
        });
        if (event.target !== row) {
          return;
        }
        openActionsMenuFromPointer(event);
      });

      const actionsButton = document.createElement("button");
      actionsButton.type = "button";
      actionsButton.className = "conversation-list__actions";
      actionsButton.setAttribute("aria-label", `Open actions for ${conversation.title || "chat"}`);
      actionsButton.setAttribute("aria-haspopup", "menu");
      actionsButton.setAttribute("aria-controls", "conversationContextMenu");
      actionsButton.textContent = "...";
      actionsButton.title = "Open chat actions";
      actionsButton.addEventListener("click", (event) => {
        logContextMenuDebug("actions-button-click", event, {
          conversationId: conversation.conversation_id,
        });
        event.preventDefault();
        event.stopPropagation();
        const rect = actionsButton.getBoundingClientRect();
        openConversationContextMenu(
          conversation.conversation_id,
          conversation.title || "New chat",
          rect.right - 8,
          rect.bottom + 6,
        );
      });
      actionsButton.addEventListener("contextmenu", openActionsMenuFromPointer);

      row.append(button, actionsButton);
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
    const roleName = message.role === "user" ? "You" : "Jade Agent";
    const timeStr = message.created_at ? formatRelativeTime(message.created_at) : "";
    meta.textContent = timeStr ? `${roleName} \u00b7 ${timeStr}` : roleName;

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

  injectCodeCopyButtons();

  requestAnimationFrame(() => {
    elements.chatStage.scrollTo({ top: elements.chatStage.scrollHeight, behavior: "smooth" });
  });
}

function injectCodeCopyButtons() {
  elements.messages.querySelectorAll("pre").forEach((pre) => {
    if (pre.querySelector(".code-copy-button")) {
      return;
    }
    const wrapper = document.createElement("div");
    wrapper.className = "code-block";
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "code-copy-button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.addEventListener("click", async () => {
      const code = pre.querySelector("code");
      const text = (code || pre).textContent;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied!";
        button.classList.add("is-copied");
        setTimeout(() => {
          button.textContent = "Copy";
          button.classList.remove("is-copied");
        }, 2000);
      } catch (_err) {
        button.textContent = "Failed";
        setTimeout(() => { button.textContent = "Copy"; }, 2000);
      }
    });
    wrapper.appendChild(button);
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
    <div class="message__card">
      <div class="thinking-dots" aria-label="Thinking">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  elements.messages.appendChild(wrapper);
  elements.emptyState.classList.add("is-hidden");
  requestAnimationFrame(() => {
    elements.chatStage.scrollTo({ top: elements.chatStage.scrollHeight, behavior: "smooth" });
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
      const card = pending.querySelector(".message__card");
      if (card) {
        card.innerHTML = `<p class="message__error">${escapeHtml(error.message || "Something went wrong. Please try again.")}</p>`;
      }
      pending.removeAttribute("id");
    }
  } finally {
    state.sending = false;
    elements.sendButton.disabled = false;
    elements.messageInput.focus();
  }
}

async function bootstrap() {
  const session = await api("/api/auth/session");
  syncAuthUi(session);
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
elements.conversationContextMenu.addEventListener("contextmenu", (event) => {
  logContextMenuDebug("menu-contextmenu", event);
  event.preventDefault();
});
document.addEventListener("pointerdown", (event) => {
  logContextMenuDebug("document-pointerdown", event, {
    menuOpen: isConversationContextMenuOpen(),
  });
  if (!isConversationContextMenuOpen()) {
    return;
  }
  if (event.button !== 0 && event.button !== -1) {
    logContextMenuDebug("document-pointerdown-ignored", event, {
      reason: "non-primary-button",
    });
    return;
  }
  if (!elements.conversationContextMenu.contains(event.target)) {
    logContextMenuDebug("document-pointerdown-close", event);
    closeConversationContextMenu();
  }
}, true);
document.addEventListener("contextmenu", (event) => {
  logContextMenuDebug("document-contextmenu", event, {
    menuOpen: isConversationContextMenuOpen(),
    insideMenu: elements.conversationContextMenu.contains(event.target),
  });
  if (elements.conversationContextMenu.contains(event.target)) {
    event.preventDefault();
  } else if (isConversationContextMenuOpen()) {
    logContextMenuDebug("document-contextmenu-close", event);
    closeConversationContextMenu();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    logContextMenuDebug("document-keydown-escape", event);
    closeConversationContextMenu();
  }
});
document.addEventListener("scroll", closeConversationContextMenu, true);
window.addEventListener("blur", closeConversationContextMenu);
window.addEventListener("resize", closeConversationContextMenu);
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
elements.contextMenuDeleteButton.addEventListener("click", async () => {
  if (!state.contextMenuConversationId) {
    return;
  }

  const conversationId = state.contextMenuConversationId;
  const conversationTitle = state.contextMenuConversationTitle;
  closeConversationContextMenu();

  try {
    await deleteConversation(conversationId, conversationTitle);
  } catch (error) {
    window.alert(error.message || "Could not delete that chat.");
  }
});
elements.logoutButton.addEventListener("click", async () => {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } finally {
    window.location.href = "/login";
  }
});

document.querySelectorAll(".starter-card").forEach((button) => {
  button.addEventListener("click", () => {
    elements.messageInput.value = button.dataset.prompt || "";
    autoResizeTextarea();
    closeSidebar();
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

if (CONTEXT_MENU_DEBUG_ENABLED) {
  ensureContextMenuDebugPanel();
  logContextMenuDebug("debug-enabled", null, {
    version: CONTEXT_MENU_DEBUG_VERSION,
    userAgent: navigator.userAgent,
  });
}
