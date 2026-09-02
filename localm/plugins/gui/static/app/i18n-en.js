// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - the English interface catalog.
 *
 * English is the source language: every key is defined here, this file is
 * imported rather than fetched, and app/i18n.js falls back to it for any key
 * a translation catalog does not carry. A translation lives in
 * static/i18n/<id>.json and needs the same keys; tests-js/i18n.test.mjs
 * checks that, and that every data-i18n attribute in index.html resolves.
 *
 * {name} is substituted by t(); a `.one` / `.other` pair is selected by tn().
 */
"use strict";

export const I18N_EN = {
  // ---- Sidebar navigation ----
  "nav.chat": "Chat",
  "nav.models": "Models",
  "nav.plugins": "Plugins",
  "nav.settings": "Settings",

  // ---- Sidebar ----
  "sidebar.activityDetails": "Click for details",
  "sidebar.conversations": "Conversations",
  "sidebar.model": "Model",
  "sidebar.newConversation": "New conversation",
  "sidebar.noModelLoaded": "No model loaded",
  "sidebar.searchChats": "search chats…",
  "sidebar.status.loadFailed": "load failed",
  "sidebar.status.loading": "loading {model}…",
  "sidebar.status.modelsUnavailable": "models unavailable (HTTP {status})",
  "sidebar.status.noModel": "no model",
  "sidebar.status.pageStale": "page out of date - reload to reconnect",
  "sidebar.status.serverUnreachable": "server unreachable",
  "sidebar.status.unloadFailed": "unload failed",
  "sidebar.status.unloading": "unloading {model}…",
  "sidebar.systemLoad": "Live system load (CPU / RAM / VRAM / GPU)",
  "sidebar.theme": "theme",
  "sidebar.toggleTheme": "Toggle light/dark",
  "sidebar.unloadModel": "unload model",
  "sidebar.unloadModelTitle":
    "Unload the active model from memory (it reloads automatically on the next chat request)",

  // ---- Mobile top bar ----
  "topbar.newChat": "New chat",
  "topbar.openMenu": "Open menu",

  // ---- Chat view ----
  "chat.advanced": "Advanced",
  "chat.advanced.hint": "sampling, seed, grammar",
  "chat.attach": "Attach images or documents (pdf, docx, txt, md, …)",
  "chat.camera": "Take a photo with the camera",
  "chat.compact": "compact",
  "chat.compact.archived.one":
    "{base} ({count} alternative branch from the older messages was archived)",
  "chat.compact.archived.other":
    "{base} ({count} alternative branches from the older messages were archived)",
  "chat.compact.summarised": "Older messages summarised to free context",
  "chat.compact.trimmed": "Older messages trimmed (summarisation unavailable)",
  "chat.compactTitle": "Summarise older messages to free context",
  "chat.conv.delete": "Delete conversation",
  "chat.conv.empty": "No conversations yet",
  "chat.conv.empty.hint": "Start one with the + above.",
  "chat.conv.deleteFailed": "Could not delete the conversation on the server - it may reappear",
  "chat.conv.noMatches": "no matching chats",
  "chat.copied": "copied",
  "chat.copy": "copy",
  "chat.copyBlocked": "Could not copy - your browser blocked clipboard access",
  "chat.default": "default",
  "chat.empty.blurb": "Chat with {model}. Everything stays on this machine.",
  "chat.empty.tip": "Type / for commands - /generate-image creates images locally.",
  "chat.empty.yourModel": "your local model",
  "chat.export": "export",
  "chat.exportTitle": "Download this conversation",
  "chat.grammar": "GBNF grammar (constrains output; local models only)",
  "chat.image.close": "Close",
  "chat.image.copied": "Image copied",
  "chat.image.copy": "Copy image",
  "chat.image.copyFailedOpen": "Could not copy the image - open it and use Save",
  "chat.image.copyFailedSave": "Could not copy the image - use Save instead",
  "chat.image.save": "Save",
  "chat.image.viewTitle": "Click to view, save, or copy",
  "chat.inputPlaceholder": "Message the model…",
  "chat.inputTitle": "Enter to send · Shift+Enter for a new line",
  "chat.knowledge":
    "Knowledge - ground replies in an indexed collection (manage on the Knowledge page)",
  "chat.maxTokens": "Max tokens",
  "chat.memory": "Memory - remind the model what it knows about you",
  "chat.memory.title":
    "Recalls facts localm has learned about you and adds them to the prompt. Add facts with /remember, view/edit with /memory. Blocked in privacy mode.",
  "chat.mic":
    "Hold a thought, speak it - click to record, click again to transcribe (needs the [voice] extra)",
  "chat.none": "(none)",
  "chat.parameters": "parameters",
  "chat.params.intro":
    "Fine-tune <b>this chat</b>. Blank fields use your defaults from <b>Settings › Chat</b> (system prompt, sampling); set one here to override it for this conversation only.",
  "chat.persistHint": "history saved on this machine",
  "chat.persistHint.title":
    "Conversations are stored in the localm data directory (chats/) because the server runs in log or full mode. They survive browser reloads and profile wipes.",
  "chat.persona.delete": "delete",
  "chat.persona.deleteTitle": "Delete the selected persona",
  "chat.persona.label": "Persona - a saved system prompt + sampling defaults",
  "chat.persona.save": "save…",
  "chat.persona.saveTitle": "Save the current system prompt and sampling values under a name",
  "chat.privacyHint": "privacy mode - this session only",
  "chat.privacyHint.title":
    "The server runs in privacy mode: conversations are not saved (here or on disk) and vanish on reload. Export still works.",
  "chat.random": "random",
  "chat.repeatPenalty": "Repeat penalty",
  "chat.seed": "Seed",
  "chat.send": "Send",
  "chat.share.clearFailed.one": "{count} shared item could not be cleared from the server",
  "chat.share.clearFailed.other": "{count} shared items could not be cleared from the server",
  "chat.share.imagesIn.one": "{count} image shared into chat",
  "chat.share.imagesIn.other": "{count} images shared into chat",
  "chat.speak": "Speak replies aloud",
  "chat.speak.title":
    "Reads each finished reply aloud. Install the tts plugin for neural Kokoro voices; otherwise the browser's built-in offline voice is used. Every assistant message also has a speaker button.",
  "chat.switchedTo": "switched to {model}",
  "chat.systemPrompt": "System prompt",
  "chat.systemPrompt.hint": "(blank = your Settings › Chat default)",
  "chat.systemPrompt.placeholder": "Blank = use your default from Settings",
  "chat.temperature": "Temperature",
  "chat.topK": "Top-k",
  "chat.topP": "Top-p",
  "chat.variant.next": "Next variant",
  "chat.variant.previous": "Previous variant",
  "chat.voice": "Voice",
  "chat.voice.hint": "(this browser)",
  "chat.voice.title":
    "Which voice reads replies aloud, remembered in this browser. Neural Kokoro voices with the tts plugin installed, otherwise your browser's offline voices.",
  "chat.waitForReply": "Wait for the current reply to finish",
  "chat.web": "Web access - the model may search and read pages",
  "chat.web.title":
    "Lets the model search the web and read pages mid-conversation (uses the server's network policy). Off = fully offline chat.",

  // ---- Plugins page ----
  "plugins.col.description": "Description",
  "plugins.col.name": "Name",
  "plugins.col.status": "Status",
  "plugins.col.tools": "Tools",
  "plugins.col.version": "Version",
  "plugins.disable": "Disable",
  "plugins.enable": "Enable",
  "plugins.enterSourcePath": "Enter the plugin folder path",
  "plugins.error.failed": "failed",
  "plugins.error.noSuchPlugin": "no such plugin",
  "plugins.error.notAllowed": "not allowed in the current state",
  "plugins.external": "External plugins",
  "plugins.external.empty.hint": "Install one from the catalog above to add capabilities.",
  "plugins.external.empty.text": "No external plugins installed",
  "plugins.externalIntro":
    "A third-party plugin is a folder with a <code>plugin.toml</code> manifest under <code>~/.localm/plugins/</code>; it can add a CLI command and export tools to the coder agent. Installation is a local directory copy.",
  "plugins.firstParty": "First-party plugins",
  "plugins.install": "Install",
  "plugins.installDependencies": "Install dependencies",
  "plugins.installFailed": "Install failed",
  "plugins.installRequirements": "Install requirements",
  "plugins.installedRestartHint":
    "Installed '{name}' {version} - restart localm gui to load its command",
  "plugins.intro":
    "LocaLM is a model loader plus a plugin engine. Only chat is active out of the box; install the others as you need them. Enabling or disabling a plugin takes effect instantly - no restart, no model reload.",
  "plugins.loadFailed": "Could not load plugins",
  "plugins.loadingProgress": "Loading plugins… ({current}/{total})",
  "plugins.missingDeps": "needs Python packages: {missing}",
  "plugins.missingRequires": "requires {requires} (missing: {missing})",
  "plugins.pill.active": "active",
  "plugins.pill.available": "available",
  "plugins.pill.installedOff": "installed (off)",
  "plugins.pill.protected": "protected",
  "plugins.refresh": "Refresh",
  "plugins.remove": "Remove",
  "plugins.removeButton": "remove",
  "plugins.removeConfirm.body": "Its folder in the data directory's plugins/ is deleted.",
  "plugins.removeConfirm.title": "Remove plugin '{name}'?",
  "plugins.removeFailed": "Remove failed",
  "plugins.removed": "Removed '{name}'",
  "plugins.sourcePlaceholder": "path to a plugin folder",
  "plugins.summary.active": "{active}/{total} plugins active",
  "plugins.summary.failed.one": "{count} failed",
  "plugins.summary.failed.other": "{count} failed",
  "plugins.title": "Plugins",
  "plugins.uninstall": "Uninstall",
  "plugins.uninstallConfirm.body": "Its plugin files are removed (your data is kept).",
  "plugins.uninstallConfirm.title": "Uninstall '{name}'?",

  // ---- Settings page ----
  "settings.group.media": "Media",
  "settings.group.model": "Model",
  "settings.group.plugins": "Plugins",
  "settings.group.privacy": "Privacy & data",
  "settings.group.security": "Security",
  "settings.group.server": "Server & network",
  "settings.group.system": "System",
  "settings.intro":
    "Stored in <code>config.json</code> in your data directory. Pick a section on the left; each one saves on its own. Engine values apply on the next model load.",
  "settings.search": "Search all settings",
  "settings.sections": "Settings sections",
  "settings.title": "Settings",

  // ---- Settings > Appearance ----
  "appearance.language": "Interface language",
  "appearance.language.help":
    "The language the interface is shown in. Saved on the server, so it follows this instance to every browser that connects to it. Parts of the app that are not translated yet stay in English.",
  "appearance.language.saveFailed":
    "Could not save the language on the server - it applies in this browser but may not on other devices.",
  "appearance.logoStyle": "Logo style",
  "appearance.logoStyle.help":
    "How the wordmark is drawn in the sidebar. Saved in this browser only.",
  "appearance.logoStyle.tile": "Use the {name} wordmark",
  "appearance.showMmproj": "Show vision projector (.mmproj) files in repository search results",
  "appearance.title": "Appearance",
};
