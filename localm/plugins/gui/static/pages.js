/* localm GUI — Models / Images / Plugins / Settings pages.
   Relies on helpers from app.js ($, el, authHeaders, toast, streamJob,
   fetchImageURL, openModal, refreshModels, modelCache, switchModel).
   Untrusted strings only ever reach the DOM via textContent. */

"use strict";

/* ================================================================ */
/*  View refresh dispatcher                                          */
/* ================================================================ */

window.onViewShown = (name) => {
  if (name === "coder") { populateSetupModels(); presetCoderMode(); }
  if (name === "models") refreshModelsPage();
  if (name === "images") refreshImageHistory();
  if (name === "music") refreshMusicHistory();
  if (name === "knowledge") refreshKnowledgePage();
  if (name === "plugins") refreshPluginsPage();
  if (name === "settings") refreshSettingsPage();
};

/** Pre-select the configured coder session mode in the setup form. */
async function presetCoderMode() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    const sel = $("setup-mode");
    if (sel && cfg.effective_coder_mode) sel.value = cfg.effective_coder_mode;
  } catch (e) { /* keep form default */ }
}

/* ================================================================ */
/*  Models page                                                      */
/* ================================================================ */

function fmtSize(bytes) {
  if (bytes == null) return "";
  return (bytes / 1e9).toFixed(2) + " GB";
}

async function refreshModelsPage() {
  await refreshModels();
  const box = $("models-table");
  box.replaceChildren();
  if (!modelCache.models.length) {
    box.appendChild(el("div", "sub", "No models registered yet — pull one above."));
    return;
  }
  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of ["Name", "Source", "Size", ""]) hr.appendChild(el("th", "", h));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  for (const m of modelCache.models) {
    const tr = el("tr");
    const nameTd = el("td");
    nameTd.appendChild(el("span", "name", m.name));
    if (m.active) nameTd.appendChild(el("span", "active-tag", "active"));
    tr.appendChild(nameTd);
    tr.appendChild(el("td", "mono", m.source || ""));
    tr.appendChild(el("td", "mono", fmtSize(m.size_bytes)));

    const actions = el("td");
    actions.style.textAlign = "right";
    const detail = el("button", "", "info");
    detail.onclick = () => showModelDetail(m.name);
    actions.appendChild(detail);
    if (!m.active) {
      const use = el("button", "", "use");
      use.onclick = async () => {
        use.disabled = true;
        try {
          await switchModel(m.name);
          toast("Model switched to " + m.name);
          refreshModelsPage();
        } catch (e) {
          toast("Load failed: " + e.message, true);
        } finally { use.disabled = false; }
      };
      actions.appendChild(use);
    }
    const alias = el("button", "", "alias");
    alias.onclick = async () => {
      const name = prompt(`New alias for '${m.name}':`);
      if (!name) return;
      const r = await fetch("/api/models/alias", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ model: m.name, alias: name.trim() }),
      });
      const data = await r.json();
      if (r.ok) { toast(`Aliased as '${name.trim()}'`); refreshModelsPage(); }
      else toast(data.detail || "Alias failed", true);
    };
    actions.appendChild(alias);
    if (!m.active) {
      const rm = el("button", "danger", "remove");
      rm.onclick = async () => {
        if (!confirm(`Remove '${m.name}'? The file is deleted only when this is its last name and it lives in ~/.localm.`)) return;
        const r = await fetch("/api/models/remove", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name }),
        });
        const data = await r.json();
        if (!r.ok) { toast(data.detail || "Remove failed", true); return; }
        const end = await streamJob(data.job_id, null);
        toast(end.status === "done" ? `Removed '${m.name}'` : "Remove failed",
              end.status !== "done");
        refreshModelsPage();
      };
      actions.appendChild(rm);
    }
    tr.appendChild(actions);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  box.appendChild(table);
}

async function showModelDetail(name) {
  const r = await fetch(`/v1/models/${encodeURIComponent(name)}`, {
    headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "Lookup failed", true); return; }
  openModal("Model — " + name, (body) => {
    const rows = [
      ["Path", data.path],
      ["Source", data.source],
      ["Size", fmtSize(data.size_bytes)],
      ["SHA256", data.sha256 || "(not computed yet — hashes lazily on use)"],
      ["Aliases", data.aliases.length ? data.aliases.join(", ") : "(none)"],
      ["Status", data.active ? (data.loaded ? "active, loaded" : "active, not loaded") : "registered"],
    ];
    for (const [k, v] of rows) {
      const row = el("div", "log-entry");
      row.appendChild(el("span", "t", k));
      row.appendChild(document.createTextNode(String(v)));
      body.appendChild(row);
    }
  });
}

/* ---- model discovery (HuggingFace search + VRAM fit badges) ---- */

function fmtCount(n) {
  if (n == null) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

const FIT_TEXT = { "fits": "fits your VRAM", "tight": "tight fit",
                   "too-big": "needs partial CPU offload" };

async function discoverSearch() {
  const box = $("disc-results");
  box.replaceChildren(el("div", "sub", "searching HuggingFace…"));
  $("disc-search").disabled = true;
  try {
    const q = $("disc-query").value.trim();
    const r = await fetch("/api/discover/search?q=" + encodeURIComponent(q),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    $("disc-vram").textContent = data.vram.total
      ? `Badges compare each file against your ${(data.vram.total / 1e9).toFixed(0)} GB total VRAM (weights + ~1.5 GB overhead).`
      : "No GPU VRAM detected — sizes shown without fit badges.";
    box.replaceChildren();
    if (!data.results.length) {
      box.appendChild(el("div", "sub", "(no GGUF repos found)"));
      return;
    }
    for (const m of data.results) {
      const row = el("div", "disc-repo");
      const head = el("div", "head");
      head.appendChild(el("span", "name", m.id));
      head.appendChild(el("span", "meta",
        `⬇ ${fmtCount(m.downloads)}  ♥ ${fmtCount(m.likes)}`));
      const btn = el("button", "", "files");
      const filesBox = el("div", "files");
      btn.onclick = () => discoverFiles(m.id, filesBox, btn);
      head.appendChild(btn);
      row.appendChild(head);
      row.appendChild(filesBox);
      box.appendChild(row);
    }
  } catch (e) {
    box.replaceChildren(el("div", "sub", "Search failed: " + e.message));
  } finally {
    $("disc-search").disabled = false;
  }
}

async function discoverFiles(repo, filesBox, btn) {
  if (filesBox.childElementCount) {            // toggle collapse
    filesBox.replaceChildren();
    return;
  }
  btn.disabled = true;
  filesBox.replaceChildren(el("div", "sub", "loading file list…"));
  try {
    const r = await fetch("/api/discover/files?repo=" + encodeURIComponent(repo),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    filesBox.replaceChildren();
    for (const f of data.files) {
      const row = el("div", "disc-file");
      row.appendChild(el("span", "quant", f.quant || "?"));
      const desc = `${(f.size_bytes / 1e9).toFixed(1)} GB` +
        (f.n_parts > 1 ? ` (${f.n_parts} parts)` : "");
      row.appendChild(el("span", "mono", desc));
      if (f.fit) row.appendChild(el("span", "fit " + f.fit, FIT_TEXT[f.fit]));
      row.appendChild(el("span", "fname", f.file));
      const pull = el("button", "", "pull");
      pull.onclick = () => {
        $("pull-spec").value = `${repo}:${f.file}`;
        $("pull-name").value = "";
        $("pull-start").click();
        $("pull-log").scrollIntoView({ behavior: "smooth", block: "nearest" });
      };
      row.appendChild(pull);
      filesBox.appendChild(row);
    }
  } catch (e) {
    filesBox.replaceChildren(el("div", "sub", "Failed: " + e.message));
  } finally {
    btn.disabled = false;
  }
}

$("disc-search").onclick = discoverSearch;
$("disc-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter") discoverSearch();
});

$("pull-start").onclick = async () => {
  const spec = $("pull-spec").value.trim();
  if (!spec) { toast("Enter a model spec", true); return; }
  const name = $("pull-name").value.trim();
  $("pull-start").disabled = true;
  const log = $("pull-log");
  log.style.display = "block";
  log.textContent = "";
  try {
    const r = await fetch("/api/models/pull", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ spec, name: name || null }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done") {
      toast("Pull finished");
      $("pull-spec").value = "";
      $("pull-name").value = "";
      refreshModelsPage();
    } else {
      toast("Pull " + end.status, true);
    }
  } catch (e) {
    toast("Pull failed: " + e.message, true);
  } finally {
    $("pull-start").disabled = false;
  }
};

/* ================================================================ */
/*  Images page                                                      */
/* ================================================================ */

async function refreshImageHistory() {
  refreshReloadToggle();
  const grid = $("img-history");
  grid.replaceChildren();
  try {
    const r = await fetch("/api/imagine/history", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    if (!data.images.length) {
      grid.appendChild(el("div", "sub", "No generated images yet."));
      return;
    }
    for (const item of data.images.slice(0, 24)) {
      const thumb = el("div", "thumb");
      const img = document.createElement("img");
      fetchImageURL(`/api/imagine/file/${encodeURIComponent(item.name)}`)
        .then((url) => (img.src = url))
        .catch(() => thumb.remove());
      thumb.appendChild(img);
      const capText = item.meta?.prompt
        ? item.meta.prompt.slice(0, 60) : item.name;
      thumb.appendChild(el("div", "cap", capText));
      thumb.onclick = () => showImageDetail(item);
      grid.appendChild(thumb);
    }
  } catch (e) { /* server unreachable */ }
}

function closeModal() { $("modal").style.display = "none"; }

function showImageDetail(item) {
  openModal(item.name, (body) => {
    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.borderRadius = "8px";
    fetchImageURL(`/api/imagine/file/${encodeURIComponent(item.name)}`)
      .then((url) => (img.src = url));
    body.appendChild(img);

    const actions = el("div", "actions");

    const useInput = el("button", "btn-quiet", "use as input");
    useInput.title = "Use this image as the img2img input";
    useInput.onclick = () => {
      $("img-input").value = item.path || item.name;
      closeModal();
      toast("Set as img2img input — adjust denoise and generate");
    };
    actions.appendChild(useInput);

    const toChat = el("button", "btn-quiet", "send to chat");
    toChat.title = "Attach this image to the chat composer";
    toChat.onclick = async () => {
      try {
        const url = await fetchImageURL(
          `/api/imagine/file/${encodeURIComponent(item.name)}`);
        const blob = await (await fetch(url)).blob();
        const dataUri = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
        chat.attachments.push({ name: item.name, dataUri });
        renderAttachChips();
        closeModal();
        showView("chat");
        toast("Image attached — type your message");
      } catch (e) {
        toast("Attach failed: " + e.message, true);
      }
    };
    actions.appendChild(toChat);

    const move = el("button", "btn-quiet", "move to folder…");
    move.onclick = async () => {
      const dest = prompt("Destination folder:", localStorage.getItem("localm.imgMoveDest") || "");
      if (!dest) return;
      const r = await fetch(
        `/api/imagine/file/${encodeURIComponent(item.name)}/move`, {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ dest: dest.trim() }),
        });
      const data = await r.json();
      if (r.ok) {
        localStorage.setItem("localm.imgMoveDest", dest.trim());
        toast("Moved to " + data.path);
        closeModal();
        refreshImageHistory();
      } else {
        toast(data.detail || "Move failed", true);
      }
    };
    actions.appendChild(move);

    const del = el("button", "danger", "delete");
    del.onclick = async () => {
      if (!confirm(`Delete ${item.name}? This removes the file from disk.`)) return;
      const r = await fetch(
        `/api/imagine/file/${encodeURIComponent(item.name)}`, {
          method: "DELETE", headers: authHeaders() });
      const data = await r.json();
      if (r.ok) {
        toast("Deleted " + item.name);
        closeModal();
        refreshImageHistory();
      } else {
        toast(data.detail || "Delete failed", true);
      }
    };
    actions.appendChild(del);

    body.appendChild(actions);

    if (item.meta && Object.keys(item.meta).length) {
      for (const [k, v] of Object.entries(item.meta)) {
        const row = el("div", "log-entry");
        row.appendChild(el("span", "t", k));
        row.appendChild(document.createTextNode(
          typeof v === "string" ? v : JSON.stringify(v)));
        body.appendChild(row);
      }
    }
  });
}

/* reload-after-generation toggle — mirrors the reload_llm_after_imagine
   server config key */

async function refreshReloadToggle() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    $("img-reload-llm").checked = cfg.reload_llm_after_imagine !== false;
  } catch (e) { /* server unreachable */ }
}

$("img-reload-llm").onchange = async () => {
  const value = $("img-reload-llm").checked;
  const r = await fetch("/v1/config", {
    method: "PATCH", headers: authHeaders(),
    body: JSON.stringify({ reload_llm_after_imagine: value }),
  });
  if (r.ok) {
    toast(value ? "Chat model reloads after each generation"
                : "ComfyUI stays loaded — chat model reloads on next message");
  } else {
    toast("Could not save setting", true);
    $("img-reload-llm").checked = !value;
  }
};

$("img-generate").onclick = async () => {
  const promptText = $("img-prompt").value.trim();
  if (!promptText) { toast("Enter a prompt", true); return; }
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  const body = {
    prompt: promptText,
    negative_prompt: $("img-negative").value.trim() || null,
    seed: num("img-seed"),
    guidance: num("img-guidance"),
    denoise: num("img-denoise"),
    input_image: $("img-input").value.trim() || null,
  };
  $("img-generate").disabled = true;
  const log = $("img-log");
  log.style.display = "block";
  log.textContent = "";
  $("img-result").replaceChildren();
  try {
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      const img = document.createElement("img");
      img.src = await fetchImageURL(
        `/api/imagine/file/${encodeURIComponent(end.result)}`);
      $("img-result").appendChild(img);
      toast("Image generated");
      refreshImageHistory();
    } else {
      toast("Generation " + end.status, true);
    }
  } catch (e) {
    toast("Generation failed: " + e.message, true);
  } finally {
    $("img-generate").disabled = false;
  }
};

/* ================================================================ */
/*  Plugins page                                                     */
/* ================================================================ */

async function refreshPluginsPage() {
  const box = $("plugins-table");
  box.replaceChildren();
  try {
    const r = await fetch("/v1/plugins", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    const data = await r.json();
    if (!data.plugins.length) {
      box.appendChild(el("div", "sub", "No external plugins installed."));
    } else {
      const table = el("table", "data-table");
      const thead = el("thead");
      const hr = el("tr");
      for (const h of ["Name", "Version", "Description", "Tools", ""])
        hr.appendChild(el("th", "", h));
      thead.appendChild(hr);
      table.appendChild(thead);
      const tbody = el("tbody");
      for (const p of data.plugins) {
        const tr = el("tr");
        const nameTd = el("td");
        nameTd.appendChild(el("span", "name", p.name));
        tr.appendChild(nameTd);
        tr.appendChild(el("td", "mono", p.version));
        tr.appendChild(el("td", "", p.description));
        tr.appendChild(el("td", "mono",
          p.tool_exports.length ? p.tool_exports.join(", ") : ""));
        const actions = el("td");
        actions.style.textAlign = "right";
        const rm = el("button", "danger", "remove");
        rm.onclick = async () => {
          if (!confirm(`Remove plugin '${p.name}'? Its folder under ~/.localm/plugins/ is deleted.`)) return;
          const rr = await fetch(`/v1/plugins/${encodeURIComponent(p.name)}`, {
            method: "DELETE", headers: authHeaders() });
          const dd = await rr.json();
          if (rr.ok) { toast(`Removed '${p.name}'`); refreshPluginsPage(); }
          else toast(dd.detail || "Remove failed", true);
        };
        actions.appendChild(rm);
        tr.appendChild(actions);
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      box.appendChild(table);
    }
    for (const err of data.errors) {
      box.appendChild(el("div", "sub", "⚠ " + err));
    }
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load plugins: " + e.message));
  }
}

$("plugin-install").onclick = async () => {
  const source = $("plugin-source").value.trim();
  if (!source) { toast("Enter the plugin folder path", true); return; }
  const r = await fetch("/v1/plugins/install", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ source }),
  });
  const data = await r.json();
  if (r.ok) {
    toast(`Installed '${data.name}' ${data.version} — restart localm gui to load its command`);
    $("plugin-source").value = "";
    refreshPluginsPage();
  } else {
    toast(data.detail || "Install failed", true);
  }
};

/* ================================================================ */
/*  Settings page                                                    */
/* ================================================================ */

// Keys hidden from the form (structured values the GUI shouldn't edit blind,
// plus read-only extras the server reports)
const _CONFIG_SKIP = new Set(["cors_origins", "effective_mode",
                              "effective_coder_mode"]);

let _configSnapshot = {};

async function refreshSettingsPage() {
  $("gui-api-key").value = localStorage.getItem("localm.apiKey") || "";
  const form = $("config-form");
  form.replaceChildren();
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    _configSnapshot = await r.json();
    for (const [key, value] of Object.entries(_configSnapshot)) {
      if (_CONFIG_SKIP.has(key)) continue;
      const wrap = el("div");
      wrap.appendChild(el("label", "", key));
      let input;
      if (typeof value === "boolean") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = value;
        input.style.width = "auto";
      } else if (typeof value === "number") {
        input = document.createElement("input");
        input.type = "number";
        input.step = Number.isInteger(value) ? "1" : "0.05";
        input.value = value;
      } else {
        input = document.createElement("input");
        input.type = "text";
        input.value = value ?? "";
      }
      input.dataset.key = key;
      wrap.appendChild(input);
      form.appendChild(wrap);
    }
  } catch (e) {
    form.appendChild(el("div", "sub", "Could not load config: " + e.message));
  }
}

$("config-save").onclick = async () => {
  const updates = {};
  for (const input of $("config-form").querySelectorAll("input")) {
    const key = input.dataset.key;
    const old = _configSnapshot[key];
    let value;
    if (input.type === "checkbox") value = input.checked;
    else if (input.type === "number") {
      if (input.value.trim() === "") continue;
      value = Number(input.value);
      if (Number.isInteger(old) && Number.isInteger(value)) value = Math.trunc(value);
    } else {
      value = input.value.trim() === "" ? null : input.value.trim();
    }
    if (value !== old) updates[key] = value;
  }
  if (!Object.keys(updates).length) { toast("Nothing changed"); return; }
  const r = await fetch("/v1/config", {
    method: "PATCH", headers: authHeaders(),
    body: JSON.stringify(updates),
  });
  const data = await r.json();
  if (r.ok) {
    toast("Saved — engine values apply on the next model load");
    _configSnapshot = data;
  } else {
    toast(data.detail || "Save failed", true);
  }
};

$("gui-key-save").onclick = () => {
  const key = $("gui-api-key").value.trim();
  if (key) localStorage.setItem("localm.apiKey", key);
  else localStorage.removeItem("localm.apiKey");
  toast("Key saved — reloading");
  setTimeout(() => location.reload(), 600);
};

/* ================================================================ */
/*  Music page                                                       */
/* ================================================================ */

$("music-generate").onclick = async () => {
  const tags = $("music-tags").value.trim();
  if (!tags) { toast("Enter style tags first", true); return; }
  const body = { tags };
  const lyrics = $("music-lyrics").value.trim();
  if (lyrics) body.lyrics = lyrics;
  const duration = Number($("music-duration").value);
  if (duration > 0) body.duration_seconds = duration;
  for (const [field, id] of [["seed", "music-seed"], ["steps", "music-steps"],
                             ["cfg", "music-cfg"]]) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) body[field] = Number(v);
  }

  $("music-generate").disabled = true;
  const log = $("music-log");
  log.style.display = "block";
  log.textContent = "";
  $("music-result").replaceChildren();
  try {
    const r = await fetch("/api/music", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      toast("Track finished");
      const player = document.createElement("audio");
      player.controls = true;
      player.style.width = "100%";
      const url = await fetchImageURL(
        "/api/music/file/" + encodeURIComponent(end.result));
      player.src = url;
      $("music-result").appendChild(player);
      refreshMusicHistory();
    } else {
      toast("Generation " + end.status, true);
    }
  } catch (e) {
    toast("Music generation failed: " + e.message, true);
  } finally {
    $("music-generate").disabled = false;
  }
};

async function refreshMusicHistory() {
  const box = $("music-history");
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/music/history", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load history: " + e.message));
    return;
  }
  if (!data.tracks.length) {
    box.appendChild(el("div", "sub", "No tracks yet — generate one above."));
    return;
  }
  for (const item of data.tracks) {
    const row = el("div", "disc-repo");
    const head = el("div", "head");
    head.appendChild(el("span", "name", item.name));
    const bits = [];
    if (item.meta?.tags) bits.push(item.meta.tags.slice(0, 60));
    if (item.meta?.duration_seconds) bits.push(`${item.meta.duration_seconds}s`);
    bits.push(`${(item.size_bytes / 1e6).toFixed(1)} MB`);
    head.appendChild(el("span", "meta", bits.join(" · ")));

    const play = el("button", "", "play");
    let player = null;
    play.onclick = async () => {
      if (player) { player.remove(); player = null; play.textContent = "play"; return; }
      play.disabled = true;
      try {
        const url = await fetchImageURL(
          "/api/music/file/" + encodeURIComponent(item.name));
        player = document.createElement("audio");
        player.controls = true;
        player.autoplay = true;
        player.style.width = "100%";
        player.src = url;
        row.appendChild(player);
        play.textContent = "hide";
      } catch (e) {
        toast("Could not load track: " + e.message, true);
      } finally {
        play.disabled = false;
      }
    };
    head.appendChild(play);

    const move = el("button", "", "move…");
    move.onclick = async () => {
      const dest = prompt("Destination folder:",
        localStorage.getItem("localm.musicMoveDest") || "");
      if (!dest) return;
      const r = await fetch(
        `/api/music/file/${encodeURIComponent(item.name)}/move`, {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ dest: dest.trim() }),
        });
      const data2 = await r.json();
      if (r.ok) {
        if (!chat.privacy) localStorage.setItem("localm.musicMoveDest", dest.trim());
        toast("Moved to " + data2.path);
        refreshMusicHistory();
      } else {
        toast(data2.detail || "Move failed", true);
      }
    };
    head.appendChild(move);

    const del = el("button", "danger", "delete");
    del.onclick = async () => {
      if (!confirm(`Delete ${item.name}?`)) return;
      const r = await fetch("/api/music/file/" + encodeURIComponent(item.name),
                            { method: "DELETE", headers: authHeaders() });
      if (r.ok) { toast("Deleted"); refreshMusicHistory(); }
      else toast("Delete failed", true);
    };
    head.appendChild(del);

    row.appendChild(head);
    box.appendChild(row);
  }
}

/* ================================================================ */
/*  Knowledge page                                                   */
/* ================================================================ */

async function refreshKnowledgePage() {
  refreshKbSelect();   // keep the chat drawer selector in sync
  const box = $("kb-table");
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/rag/collections", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load collections: " + e.message));
    return;
  }
  if (!data.collections.length) {
    box.appendChild(el("div", "sub",
      "No collections yet — create one above, then add files or folders to it."));
    return;
  }
  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of ["Name", "Docs", "Chunks", "Retrieval", ""]) {
    hr.appendChild(el("th", "", h));
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  for (const c of data.collections) {
    const tr = el("tr");
    const nameTd = el("td");
    nameTd.appendChild(el("span", "name", c.name));
    tr.appendChild(nameTd);
    tr.appendChild(el("td", "mono", String(c.n_docs)));
    tr.appendChild(el("td", "mono", String(c.n_chunks)));
    tr.appendChild(el("td", "mono", c.has_vectors ? "hybrid" : "BM25"));
    const actions = el("td");
    actions.style.textAlign = "right";

    const add = el("button", "", "add docs");
    add.onclick = () => kbAddDocs(c.name);
    actions.appendChild(add);

    const search = el("button", "", "search");
    search.onclick = () => kbSearchModal(c.name);
    actions.appendChild(search);

    const info = el("button", "", "info");
    info.onclick = () => kbInfoModal(c.name);
    actions.appendChild(info);

    const del = el("button", "danger", "delete");
    del.onclick = async () => {
      if (!confirm(`Delete collection '${c.name}'? Only the index is removed — ` +
                   "your original files are untouched.")) return;
      const r = await fetch(
        "/api/rag/collections/" + encodeURIComponent(c.name), {
          method: "DELETE", headers: authHeaders() });
      if (r.ok) { toast("Deleted " + c.name); refreshKnowledgePage(); }
      else toast("Delete failed", true);
    };
    actions.appendChild(del);
    tr.appendChild(actions);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  box.appendChild(table);
}

$("kb-create").onclick = async () => {
  const name = $("kb-name").value.trim();
  if (!name) { toast("Enter a collection name", true); return; }
  const r = await fetch("/api/rag/collections", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ name }),
  });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "Create failed", true); return; }
  $("kb-name").value = "";
  toast(`Collection '${data.name}' created — now add files or folders`);
  refreshKnowledgePage();
};

async function kbAddDocs(name) {
  const path = prompt(
    `Add documents to '${name}' — file or folder path on this machine\n` +
    "(folders are indexed recursively; txt/md/pdf/docx/html/code):",
    localStorage.getItem("localm.kbAddPath") || "");
  if (!path || !path.trim()) return;
  if (!chat.privacy) localStorage.setItem("localm.kbAddPath", path.trim());
  const log = $("kb-log");
  log.style.display = "block";
  log.textContent = `Indexing ${path.trim()} into '${name}'…\n`;
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/add`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ paths: [path.trim()] }),
      });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    toast(end.status === "done" ? "Indexing finished"
                                : "Indexing " + end.status,
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += "failed: " + e.message + "\n";
    toast("Indexing failed: " + e.message, true);
  }
}

async function kbInfoModal(name) {
  const r = await fetch("/api/rag/collections/" + encodeURIComponent(name),
                        { headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "Load failed", true); return; }
  openModal("Collection — " + name, (body) => {
    body.appendChild(el("div", "sub",
      `${data.n_docs} documents · ${data.n_chunks} chunks · ` +
      (data.has_vectors ? "hybrid retrieval (BM25 + embeddings)"
                        : "lexical retrieval (BM25)")));
    for (const d of data.docs) {
      const row = el("div", "log-entry");
      row.appendChild(el("span", "t", `${d.chunks} chunks`));
      row.appendChild(document.createTextNode(d.path + " "));
      const rm = el("button", "action", "remove");
      rm.onclick = async () => {
        const rr = await fetch(
          `/api/rag/collections/${encodeURIComponent(name)}/remove-doc`, {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ path: d.path }),
          });
        if (rr.ok) { toast("Removed"); kbInfoModal(name); refreshKnowledgePage(); }
        else toast("Remove failed", true);
      };
      row.appendChild(rm);
      body.appendChild(row);
    }
    if (!data.docs.length) {
      body.appendChild(el("div", "sub", "(empty — use add docs)"));
    }
  });
}

function kbSearchModal(name) {
  openModal("Search — " + name, (body) => {
    const row = el("div", "row");
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "test query…";
    input.style.flex = "1";
    const go = el("button", "btn-primary", "Search");
    row.appendChild(input);
    row.appendChild(go);
    body.appendChild(row);
    const results = el("div");
    body.appendChild(results);
    const run = async () => {
      const q = input.value.trim();
      if (!q) return;
      results.replaceChildren(el("div", "sub", "searching…"));
      try {
        const r = await fetch(
          `/api/rag/collections/${encodeURIComponent(name)}/query`, {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ query: q, k: 5 }),
          });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || r.statusText);
        results.replaceChildren();
        if (!data.hits.length) {
          results.appendChild(el("div", "sub", "(no matches)"));
          return;
        }
        for (const h of data.hits) {
          const entry = el("div", "log-entry");
          entry.appendChild(el("span", "t",
            `${h.score}  ${h.source.split(/[\\/]/).pop()}:${h.pos}`));
          entry.appendChild(document.createTextNode(h.text.slice(0, 400)));
          results.appendChild(entry);
        }
      } catch (e) {
        results.replaceChildren(el("div", "sub", "failed: " + e.message));
      }
    };
    go.onclick = run;
    input.onkeydown = (e) => { if (e.key === "Enter") run(); };
    input.focus();
  });
}

$("gui-clear-convs").onclick = async () => {
  const where = chat.persist
    ? "from this browser AND the server store" : "from this browser";
  if (!confirm(`Delete all saved conversations ${where}?`)) return;
  if (chat.persist) {
    // Server store is the source of truth — clear it too or they come back.
    await Promise.allSettled(chat.conversations.map((c) =>
      fetch("/api/conversations/" + encodeURIComponent(c.id), {
        method: "DELETE", headers: authHeaders(),
      })));
  }
  localStorage.removeItem("localm.conversations");
  location.reload();
};
