// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - slash commands. */
"use strict";

// --- ES module imports ---
import { addMessageRow, chat, currentConv, newConversation, renderChat, renderConvList, saveConversations } from "./chat.js";
import { exportCoderSession, openFilesModal } from "./coder.js";
import { $, authHeaders, autoGrow, el, jobStatusWord, nearBottom, openModal, streamJob, toast } from "./helpers.js";
import { t } from "./i18n.js";
import { applyPersona, exportConversation, openMemoryModal, personaCache, pluginSuggestion, rememberFact, requestWebTool, runCompletion } from "./settings-perf.js";

/* ================================================================ */
/*  Slash commands                                                   */
/* ================================================================ */

// hint/args hold catalog keys, not literal text; callers resolve them with t().
export const CHAT_COMMANDS = [
  { cmd: "generate-image", hint: "slash.chatCmd.generateImage.hint", args: "slash.chatCmd.generateImage.args" },
  { cmd: "generate-music", hint: "slash.chatCmd.generateMusic.hint", args: "slash.chatCmd.generateMusic.args" },
  { cmd: "generate-video", hint: "slash.chatCmd.generateVideo.hint", args: "slash.chatCmd.generateVideo.args" },
  { cmd: "web", hint: "slash.chatCmd.web.hint", args: "slash.chatCmd.web.args" },
  { cmd: "clear", hint: "slash.chatCmd.clear.hint" },
  { cmd: "compact", hint: "slash.chatCmd.compact.hint" },
  { cmd: "export", hint: "slash.chatCmd.export.hint" },
  { cmd: "rename", hint: "slash.chatCmd.rename.hint", args: "slash.chatCmd.rename.args" },
  { cmd: "persona", hint: "slash.chatCmd.persona.hint", args: "slash.chatCmd.persona.args" },
  { cmd: "remember", hint: "slash.chatCmd.remember.hint", args: "slash.chatCmd.remember.args" },
  { cmd: "memory", hint: "slash.chatCmd.memory.hint" },
  { cmd: "pin", hint: "slash.chatCmd.pin.hint" },
  { cmd: "folder", hint: "slash.chatCmd.folder.hint", args: "slash.chatCmd.folder.args" },
  { cmd: "system", hint: "slash.chatCmd.system.hint" },
  { cmd: "new", hint: "slash.chatCmd.new.hint" },
];

export const CODER_COMMANDS = [
  { cmd: "undo", hint: "slash.coderCmd.undo.hint" },
  { cmd: "files", hint: "slash.coderCmd.files.hint" },
  { cmd: "compact", hint: "slash.coderCmd.compact.hint" },
  { cmd: "export", hint: "slash.coderCmd.export.hint" },
  { cmd: "log", hint: "slash.coderCmd.log.hint" },
  { cmd: "stop", hint: "slash.coderCmd.stop.hint" },
  { cmd: "end", hint: "slash.coderCmd.end.hint" },
  { cmd: "help", hint: "slash.coderCmd.help.hint" },
];

export async function runImagineInChat(promptText) {
  if (!promptText) { toast(t("slash.usage.image"), true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/generate-image " + promptText });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = t("slash.generatingImage");
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ prompt: promptText }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: [
          { type: "text", text: t("slash.resultImageLabel") },
          { type: "image_url",
            image_url: { url: "/api/imagine/file/" + encodeURIComponent(end.result) } },
        ],
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = t("slash.imageStatus", { status: jobStatusWord(end.status) });
    }
  } catch (e) {
    body.textContent = t("slash.imageFailed", { message: e.message });
    toast(e.message, true);
  }
}

/** /web <query> - search, inject the results into the conversation, and let the
 *  model answer from them. */
export async function runWebInChat(query) {
  if (!query) { toast(t("slash.usage.web"), true); return; }
  if (chat.abort) { toast(t("chat.waitForReply"), true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/web " + query });
  if (conv.messages.length === 1) {
    conv.title = query.slice(0, 42) + (query.length > 42 ? "…" : "");
    renderConvList();
  }
  saveConversations(conv);
  renderChat();
  let note;
  try {
    note = await requestWebTool({ name: "web_search", args: { query } });
    // Model-directed text, appended to the message the model reads next; left untranslated.
    note += `\n\nUsing these results, answer: ${query}\nName the sources you used.`;
  } catch (e) {
    toast(t("slash.webSearchFailed", { message: e.message }), true);
    // Model-directed text (see above), left untranslated.
    note = `[Web search failed: ${e.message}] Tell the user, and answer ` +
           "from your own knowledge if you can.";
  }
  conv.messages.push({ role: "user", content: note, web: true });
  saveConversations(conv);
  renderChat();
  await runCompletion(conv);
}

/** /music <tags> - generate a default-length instrumental inline. */
export async function runMusicInChat(tags) {
  if (!tags) { toast(t("slash.usage.music"), true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/generate-music " + tags });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = t("slash.generatingTrack");
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/music", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ tags }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: t("slash.resultTrackLabel"),
        audio: "/api/music/file/" + encodeURIComponent(end.result),
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = t("slash.musicStatus", { status: jobStatusWord(end.status) });
    }
  } catch (e) {
    body.textContent = t("slash.musicFailed", { message: e.message });
    toast(e.message, true);
  }
}

/** /video <prompt> - generate a default-length (~5s) clip inline. */
export async function runVideoInChat(promptText) {
  if (!promptText) { toast(t("slash.usage.video"), true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/generate-video " + promptText });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = t("slash.generatingClip");
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/video", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ prompt: promptText }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: t("slash.resultClipLabel"),
        video: "/api/video/file/" + encodeURIComponent(end.result),
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = t("slash.videoStatus", { status: jobStatusWord(end.status) });
    }
  } catch (e) {
    body.textContent = t("slash.videoFailed", { message: e.message });
    toast(e.message, true);
  }
}

export function execChatCommand(cmd, arg) {
  switch (cmd) {
    case "generate-image": case "imagine": runImagineInChat(arg); return true;
    case "generate-music": case "music": runMusicInChat(arg); return true;
    case "generate-video": case "video": runVideoInChat(arg); return true;
    case "web": runWebInChat(arg); return true;
    case "clear": {
      const conv = currentConv();
      if (conv) {
        conv.messages = [];
        conv.branches = [];
        saveConversations(conv);
        renderChat();
      }
      return true;
    }
    case "compact": $("compact-conv").onclick(); return true;
    case "export": exportConversation(); return true;
    case "rename": {
      const conv = currentConv();
      if (conv && arg) { conv.title = arg; saveConversations(conv); renderConvList(); }
      else toast(t("slash.usage.rename"), true);
      return true;
    }
    case "remember": rememberFact(arg); return true;
    case "memory": openMemoryModal(); return true;
    case "persona": {
      if (!arg) {
        const names = personaCache.map((p) => p.name);
        toast(names.length ? t("slash.personaList", { names: names.join(", ") })
                           : t("slash.noPersonas"),
              !names.length);
        return true;
      }
      // case-insensitive match
      const hit = personaCache.find(
        (p) => p.name.toLowerCase() === arg.toLowerCase());
      applyPersona(hit ? hit.name : arg);
      return true;
    }
    case "pin": {
      const conv = currentConv();
      if (!conv) return true;
      conv.pinned = !conv.pinned;
      if (!conv.pinned) delete conv.pinned;
      saveConversations(conv);
      renderConvList();
      toast(conv.pinned ? t("slash.pinned") : t("slash.unpinned"));
      return true;
    }
    case "folder": {
      const conv = currentConv();
      if (!conv) return true;
      if (arg) conv.folder = arg;
      else delete conv.folder;
      saveConversations(conv);
      renderConvList();
      toast(arg ? t("slash.movedToFolder", { name: arg }) : t("slash.removedFromFolder"));
      return true;
    }
    case "system":
      $("params").classList.add("open");
      $("p-system").focus();
      return true;
    case "new": newConversation(); return true;
  }
  return false;
}

export function execCoderCommand(cmd) {
  switch (cmd) {
    case "undo": $("coder-undo").onclick(); return true;
    case "files": openFilesModal(); return true;
    case "compact": $("coder-compact").onclick(); return true;
    case "export": exportCoderSession(); return true;
    case "log": $("coder-log").onclick(); return true;
    case "stop": $("coder-stop").onclick(); return true;
    case "end": $("coder-end").onclick(); return true;
    case "help":
      openModal(t("slash.coderCommandsTitle"), (body) => {
        for (const c of CODER_COMMANDS) {
          const row = el("div", "log-entry");
          row.appendChild(el("span", "t", "/" + c.cmd));
          row.appendChild(document.createTextNode(t(c.hint)));
          body.appendChild(row);
        }
        body.appendChild(el("div", "sub", t("slash.coderCommandsHint")));
      });
      return true;
  }
  return false;
}

/** Attach a slash-command dropdown to a composer textarea. */
export function attachSlashMenu(textarea, commands, execute) {
  const menu = el("div", "slash-menu");
  menu.style.display = "none";
  textarea.closest(".composer-wrap").appendChild(menu);
  let selected = 0;
  let visible = [];

  function close() { menu.style.display = "none"; visible = []; }

  function render() {
    const value = textarea.value;
    if (!value.startsWith("/") || value.includes("\n")) { close(); return; }
    // A space ends the command token: close the menu so Enter sends the whole
    // line rather than picking a command.
    const rest = value.slice(1);
    if (rest.includes(" ")) { close(); return; }
    const typed = rest.toLowerCase();
    visible = commands.filter((c) => c.cmd.startsWith(typed));
    if (!visible.length) { close(); return; }
    selected = Math.min(selected, visible.length - 1);
    menu.replaceChildren();
    visible.forEach((c, i) => {
      const row = el("div", "slash-item" + (i === selected ? " selected" : ""));
      row.appendChild(el("span", "cmd", "/" + c.cmd + (c.args ? " " + t(c.args) : "")));
      row.appendChild(el("span", "hint", t(c.hint)));
      row.onmousedown = (e) => { e.preventDefault(); pick(c); };
      menu.appendChild(row);
    });
    menu.style.display = "block";
  }

  function pick(c) {
    if (!c) { close(); return; }   // empty list in a render/keydown race
    if (c.args) {
      textarea.value = "/" + c.cmd + " ";
      textarea.focus();
      close();
    } else {
      textarea.value = "";
      autoGrow(textarea);
      close();
      execute(c.cmd, "");
    }
  }

  textarea.addEventListener("input", () => { selected = 0; render(); });
  textarea.addEventListener("blur", () => setTimeout(close, 150));
  textarea.addEventListener("keydown", (e) => {
    if (menu.style.display === "none") return;
    if (e.key === "ArrowDown") {
      e.preventDefault(); selected = (selected + 1) % visible.length; render();
    } else if (e.key === "ArrowUp") {
      e.preventDefault(); selected = (selected - 1 + visible.length) % visible.length; render();
    } else if (e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
      e.preventDefault(); pick(visible[selected]);
    } else if (e.key === "Escape") {
      close();
    }
  });
}

/** Intercept "/cmd arg" on submit. Returns true when handled (not for the model). */
export function handleSlashSubmit(text, execute) {
  if (!text.startsWith("/")) return false;
  const space = text.indexOf(" ");
  const cmd = (space === -1 ? text.slice(1) : text.slice(1, space)).toLowerCase();
  const arg = space === -1 ? "" : text.slice(space + 1).trim();
  const hint = pluginSuggestion(cmd);   // known plugin, just not active yet
  if (hint) { toast(hint, true); return true; }
  if (!execute(cmd, arg)) {
    toast(t("slash.unknownCommand", { cmd }), true);
  }
  return true;   // never send slash input to the model
}

attachSlashMenu($("chat-input"), CHAT_COMMANDS, execChatCommand);
attachSlashMenu($("coder-input"), CODER_COMMANDS, (c) => execCoderCommand(c));

