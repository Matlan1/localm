// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - slash commands. */
"use strict";

// --- ES module imports ---
import { addMessageRow, chat, chatBusy, currentConv, newConversation, newToolEvent, renderChat, renderConvList, saveConversations } from "./chat.js";
import { activeSession, exportCoderSession, openFilesModal, switchActiveSessionModel } from "./coder.js";
import { $, CHAT_SURFACE_HEADERS, authHeaders, autoGrow, checkModelsBeforeGenerate, createMissingModelOffer, el, jobStatusWord, nearBottom, openModal, streamJob, toast } from "./helpers.js";
import { t } from "./i18n.js";
import { GROUNDING_PAGE_BACKED, applyPersona, exportConversation, openMemoryModal, personaCache, pluginSuggestion, refuseWithoutModel, rememberFact, requestWebTool, runCompletion } from "./settings-perf.js";

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
  { cmd: "model", hint: "slash.coderCmd.model.hint", args: "slash.coderCmd.model.args" },
  { cmd: "compact", hint: "slash.coderCmd.compact.hint" },
  { cmd: "export", hint: "slash.coderCmd.export.hint" },
  { cmd: "log", hint: "slash.coderCmd.log.hint" },
  { cmd: "stop", hint: "slash.coderCmd.stop.hint" },
  { cmd: "end", hint: "slash.coderCmd.end.hint" },
  { cmd: "help", hint: "slash.coderCmd.help.hint" },
];

/** Per media, an async function returning the model fields that media page's
 *  Generate would send, or null when that page would refuse to generate (set
 *  by the page; the entry is null when the page is not loaded). */
export const chatPageModelChoice = { image: null, music: null, video: null };

/** The model fields *media*'s page would send with its Generate ({} when the
 *  page is not loaded), or null when that page would refuse to generate. */
async function chatModelFields(media) {
  const pick = chatPageModelChoice[media];
  return pick ? pick() : {};
}

/** The preflight body for the model fields *picked*. */
function preflightBody(picked) {
  return { model_overrides: picked.model_overrides, native_models: picked.native_models };
}

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
    const picked = await chatModelFields("image");
    if (!picked || !(await checkModelsBeforeGenerate("image", null, preflightBody(picked)))) {
      body.textContent = t("slash.modelsNotDownloaded");
      return;
    }
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(CHAT_SURFACE_HEADERS),
      body: JSON.stringify({ prompt: promptText, ...picked }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const missing = createMissingModelOffer(null);
    const end = await streamJob(data.job_id, (line) => {
      if (!missing.show(line)) return;
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    }, undefined, undefined, missing.onEvent);
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
      await missing.offer();
    }
  } catch (e) {
    body.textContent = t("slash.imageFailed", { message: e.message });
  }
}

/** /web <query> - search, read the top result pages (the shared retrieval
 *  controller behind POST /api/web/retrieve), inject the evidence into the
 *  conversation, and let the model answer from it citing source IDs. The
 *  explicit command is the consent for that one search and its bounded page
 *  reads; net_mode=off, the domain lists and the SSRF guard still apply on
 *  the server, only the per-chat web toggle is bypassed. */
export async function runWebInChat(query) {
  if (!query) { toast(t("slash.usage.web"), true); return; }
  if (chatBusy()) { toast(t("chat.waitForReply"), true); return; }
  if (refuseWithoutModel()) return;
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/web " + query });
  if (conv.messages.length === 1) {
    conv.title = query.slice(0, 42) + (query.length > 42 ? "…" : "");
    renderConvList();
  }
  saveConversations(conv);
  renderChat();
  const ev = newToolEvent({ tool: "search", status: "running", query,
                            started_at: Date.now() });
  conv.messages.push(ev);
  chat.webCall = ev;
  renderChat();
  try {
    Object.assign(ev, await requestWebTool({ name: "web_search", args: { query } }));
    // Model-directed text, appended to the result the model reads next; left untranslated.
    ev.note = `Using this evidence, answer: ${query}\n` +
              "Cite the source IDs (S1, S2, ...) you relied on; never cite a URL " +
              "whose page was not read.";
    if (ev.grounding !== GROUNDING_PAGE_BACKED) {
      ev.note += "\nNo page could be read: the evidence above is search snippets " +
                 "only, or empty. Say so plainly instead of implying you read the pages.";
      toast(t("slash.webNoPageRead"), true);
    }
  } catch (e) {
    toast(t("slash.webSearchFailed", { message: e.message }), true);
    ev.status = "failed";
    ev.error = e.message;
  } finally {
    chat.webCall = null;
  }
  ev.finished_at = Date.now();
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
    const picked = await chatModelFields("music");
    if (!picked || !(await checkModelsBeforeGenerate("music", null, preflightBody(picked)))) {
      body.textContent = t("slash.modelsNotDownloaded");
      return;
    }
    const r = await fetch("/api/music", {
      method: "POST", headers: authHeaders(CHAT_SURFACE_HEADERS),
      body: JSON.stringify({ tags, ...picked }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const missing = createMissingModelOffer(null);
    const end = await streamJob(data.job_id, (line) => {
      if (!missing.show(line)) return;
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    }, undefined, undefined, missing.onEvent);
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
      await missing.offer();
    }
  } catch (e) {
    body.textContent = t("slash.musicFailed", { message: e.message });
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
    const picked = await chatModelFields("video");
    if (!picked || !(await checkModelsBeforeGenerate("video", null, preflightBody(picked)))) {
      body.textContent = t("slash.modelsNotDownloaded");
      return;
    }
    const r = await fetch("/api/video", {
      method: "POST", headers: authHeaders(CHAT_SURFACE_HEADERS),
      body: JSON.stringify({ prompt: promptText, ...picked }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const missing = createMissingModelOffer(null);
    const end = await streamJob(data.job_id, (line) => {
      if (!missing.show(line)) return;
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    }, undefined, undefined, missing.onEvent);
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
      await missing.offer();
    }
  } catch (e) {
    body.textContent = t("slash.videoFailed", { message: e.message });
  }
}

/** Other names a chat command answers to, mapped to its CHAT_COMMANDS name. */
export const CHAT_ALIASES = { imagine: "generate-image", music: "generate-music", video: "generate-video" };

/** Run chat command *cmd* with *arg*. Returns false for a name that is not a
 *  chat command. While a reply streams every command is refused with the
 *  "wait for the current reply" toast, whether it was typed and sent or picked
 *  from the slash menu. */
export function execChatCommand(cmd, arg) {
  cmd = CHAT_ALIASES[cmd] || cmd;
  if (!CHAT_COMMANDS.some((c) => c.cmd === cmd)) return false;
  if (chatBusy()) { toast(t("chat.waitForReply"), true); return true; }
  switch (cmd) {
    case "generate-image": runImagineInChat(arg); return true;
    case "generate-music": runMusicInChat(arg); return true;
    case "generate-video": runVideoInChat(arg); return true;
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
      if (!arg) { toast(t("slash.usage.rename"), true); return true; }
      if (!currentConv()) newConversation();
      const conv = currentConv();
      conv.title = arg;
      saveConversations(conv);
      renderConvList();
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

export function execCoderCommand(cmd, arg = "") {
  switch (cmd) {
    case "undo": $("coder-undo").onclick(); return true;
    case "files": openFilesModal(); return true;
    case "model": {
      const s = activeSession();
      if (!s) { toast(t("coder.session.startFirst"), true); return true; }
      if (!arg) {
        toast(t("slash.coderCmd.currentModel", { model: s.info.model || t("chat.none") }));
      } else {
        switchActiveSessionModel(arg);
      }
      return true;
    }
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

/** Intercept "/cmd arg" on submit. Returns true when handled (not for the model).
 *  The "needs the X plugin" hint is given only to the chat composer
 *  (*execute* is execChatCommand), for the command an alias stands for. */
export function handleSlashSubmit(text, execute) {
  if (!text.startsWith("/")) return false;
  const space = text.indexOf(" ");
  const cmd = (space === -1 ? text.slice(1) : text.slice(1, space)).toLowerCase();
  const arg = space === -1 ? "" : text.slice(space + 1).trim();
  const hint = execute === execChatCommand ? pluginSuggestion(CHAT_ALIASES[cmd] || cmd) : null;
  if (hint) { toast(hint, true); return true; }
  if (!execute(cmd, arg)) {
    toast(t("slash.unknownCommand", { cmd }), true);
  }
  return true;   // never send slash input to the model
}

attachSlashMenu($("chat-input"), CHAT_COMMANDS, execChatCommand);
attachSlashMenu($("coder-input"), CODER_COMMANDS, (c, a) => execCoderCommand(c, a));

