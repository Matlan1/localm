# Harmony (gpt-oss) chat template

`gpt-oss-20b.jinja` is the `tokenizer.chat_template` embedded in the `gpt-oss-20b` GGUF
(`general.license` apache-2.0), unmodified.

`expected.json` holds the template's output for the conversations in its `cases`, rendered with
Jinja2 the way Hugging Face transformers renders chat templates (sandboxed environment,
`trim_blocks`, `lstrip_blocks`, `add_generation_prompt` true, no tools), with `strftime_now`
returning the `today` value of the file.
