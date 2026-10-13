# Gemma 4 chat templates

`chat_template.jinja` files from Google's Gemma 4 model repositories on Hugging Face, licensed under
the Apache License 2.0, (c) Google LLC:

| File | Repository | Revision |
|---|---|---|
| `gemma-4-E2B-it.jinja` | google/gemma-4-E2B-it | 3e22461f65e8 (2026-07-20) |
| `gemma-4-31B-it.jinja` | google/gemma-4-31B-it | 842da3794eaa (2026-07-20) |
| `gemma-4-31B-it-2026-05-18.jinja` | google/gemma-4-31B-it | fcf2302760ae (2026-05-18) |
| `gemma-4-E2B-it-2026-04-02.jinja` | google/gemma-4-E2B-it | 4742fe843cc0 (2026-04-02) |

Modification: an em dash inside a Jinja comment of the two 2026-07-20 files is replaced by a
hyphen. Comments are not rendered, so the output is unchanged.

`expected.json` holds each template's output for the conversations in its `cases`, rendered with
Jinja2 the way Hugging Face transformers renders chat templates (sandboxed environment,
`trim_blocks`, `lstrip_blocks`, `bos_token` `<bos>`, `add_generation_prompt` true, thinking not
enabled).
