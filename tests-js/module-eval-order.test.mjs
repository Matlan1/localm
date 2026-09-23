// SPDX-License-Identifier: AGPL-3.0-or-later
// The GUI's real ES module graph, in the order a browser evaluates it: no
// module reads an imported let/const/class binding at its top level before the
// module that declares it has run.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Linter } from "eslint";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = join(ROOT, "localm", "plugins", "gui", "static");
const ENTRY = join(STATIC, "app", "main.js");

function analyze(file) {
  const code = readFileSync(file, "utf-8");
  const info = { imports: [], bindings: new Map(), lexical: new Set(), topReads: [] };
  const linter = new Linter({ configType: "flat" });
  const grab = {
    create(ctx) {
      return {
        "Program:exit"(program) {
          const sm = ctx.sourceCode.scopeManager;
          for (const node of program.body) {
            if (node.type === "ImportDeclaration") {
              const from = resolve(dirname(file), node.source.value);
              info.imports.push(from);
              for (const s of node.specifiers) {
                if (s.type === "ImportSpecifier") {
                  info.bindings.set(s.local.name, { from, name: s.imported.name });
                }
              }
            }
            const decl = node.type === "ExportNamedDeclaration" ? node.declaration : node;
            if (decl && decl.type === "VariableDeclaration" && decl.kind !== "var") {
              for (const d of decl.declarations) {
                if (d.id.type === "Identifier") info.lexical.add(d.id.name);
              }
            }
            if (decl && decl.type === "ClassDeclaration" && decl.id) info.lexical.add(decl.id.name);
          }
          const moduleScope = sm.globalScope.childScopes.find((s) => s.type === "module");
          const walk = (scope) => {
            for (const ref of scope.references) {
              const v = ref.resolved;
              if (!v || !info.bindings.has(v.name) || v.scope !== moduleScope) continue;
              if (ref.from.variableScope === moduleScope) {
                info.topReads.push({ name: v.name, line: ref.identifier.loc.start.line });
              }
            }
            for (const child of scope.childScopes) walk(child);
          };
          walk(moduleScope);
        },
      };
    },
  };
  const messages = linter.verify(code, [{
    files: ["**/*.js"],
    languageOptions: { ecmaVersion: "latest", sourceType: "module" },
    plugins: { t: { rules: { grab } } },
    rules: { "t/grab": "error" },
  }], { filename: file });
  const fatal = messages.filter((m) => m.fatal);
  assert.deepEqual(fatal, [], `${file} does not parse`);
  return info;
}

function evaluationOrder(entry) {
  const infos = new Map();
  const order = [];
  const state = new Map();
  const visit = (file) => {
    if (state.has(file)) return;
    state.set(file, "in-progress");
    if (!infos.has(file)) infos.set(file, analyze(file));
    for (const dep of infos.get(file).imports) visit(dep);
    state.set(file, "done");
    order.push(file);
  };
  visit(entry);
  return { infos, order };
}

function earlyReads(entry) {
  const { infos, order } = evaluationOrder(entry);
  const position = new Map(order.map((f, i) => [f, i]));
  const found = [];
  for (const file of order) {
    const info = infos.get(file);
    for (const read of info.topReads) {
      const b = info.bindings.get(read.name);
      const source = infos.get(b.from);
      if (!source || !source.lexical.has(b.name)) continue;
      if (position.get(b.from) > position.get(file)) {
        found.push(`${file.slice(STATIC.length + 1)}:${read.line} reads ${read.name} `
          + `from ${b.from.slice(STATIC.length + 1)} before that module has run`);
      }
    }
  }
  return { found, order };
}

test("no GUI module reads an imported binding before its module has run", () => {
  const { found, order } = earlyReads(ENTRY);
  assert.ok(order.length > 20, `the module graph was walked (${order.length} modules)`);
  assert.deepEqual(found, []);
});
