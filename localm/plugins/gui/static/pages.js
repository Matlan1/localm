/* localm GUI — Models / Images / Plugins / Settings pages.
   Relies on helpers from app.js ($, el, authHeaders, toast, streamJob,
   fetchImageURL, openModal, refreshModels, modelCache, switchModel).
   Untrusted strings only ever reach the DOM via textContent. */

"use strict";

/* ================================================================ */
/*  View refresh dispatcher                                          */
/* ================================================================ */

window.onViewShown = (name) => {
  if (name === "coder") populateSetupModels();
  if (name === "models") refreshModelsPage();
  if (name === "images") refreshImageHistory();
  if (name === "plugins") refreshPluginsPage();
  if (name === "settings") refreshSettingsPage();
};

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

function showImageDetail(item) {
  openModal(item.name, (body) => {
    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.borderRadius = "8px";
    fetchImageURL(`/api/imagine/file/${encodeURIComponent(item.name)}`)
      .then((url) => (img.src = url));
    body.appendChild(img);
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

// Keys hidden from the form (structured values the GUI shouldn't edit blind)
const _CONFIG_SKIP = new Set(["cors_origins"]);

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

$("gui-clear-convs").onclick = () => {
  if (!confirm("Delete all saved conversations from this browser?")) return;
  localStorage.removeItem("localm.conversations");
  location.reload();
};
