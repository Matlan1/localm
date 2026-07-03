# localm GUI design standard

The web GUI (`localm/plugins/gui/static/`) should feel like one coherent, quiet,
professional dark app. The file/folder picker (`app/picker.js`) and the grouped
settings page (`pages/settings.js`) are the reference surfaces: match them.

This is the checklist. When you add or touch a GUI surface, it should already
follow these eight rules. Everything here uses tokens and helpers that already
exist, so "consistent" is almost never "new CSS".

## 1. Icons are inline SVG, never emoji

Use the shared set in `app/icons.js`. Emoji and HTML-entity glyphs are blurry,
off-baseline, and cannot follow the theme, so they are banned on the shipping
surface.

- JS-built DOM: `iconEl("send", "cls")` returns a `<span>` with the SVG.
- Static markup: put `<span data-icon="send"></span>` in `index.html`; it is
  hydrated on load. Call `hydrateIcons(root)` after inserting a fragment.
- Every icon is a 24x24 `viewBox` stroke SVG drawn with `stroke="currentColor"`,
  so it inherits color and both themes. Add a new one to `APP_ICONS`.
- ASCII `x` (close) and `+` are fine as text; they are crisp and conventional.

## 2. Every interactive row has hover + an inset accent bar when active

The picker row is canon:

```css
.thing:hover  { background: var(--bg-input); }
.thing.active { background: var(--accent-soft); box-shadow: inset 2px 0 0 var(--accent); color: var(--accent); font-weight: 600; }
```

Nav buttons, table rows, list items, tabs: reuse this exact pattern.

## 3. Two button tiers plus danger, one vocabulary

- `.btn-primary` (filled accent, white text) for the one main action.
- `.btn-secondary` (outline, dim text) for everything else.
- `.btn-danger` (red outline that FILLS red on hover) for destructive actions.

Inside a dense `.data-table`, the compact `.data-table button` styling wins;
there, use `.primary` / `.danger` modifiers (accent / red that fill on hover).
Never ship a class-less `el("button", "", ...)`.

## 4. Sections are cards

`bg-raised`, `1px var(--border)`, 12px radius, generous padding, an `<h3>` head,
and `.sub` helper text under labels. The settings page is built entirely from
these; loose inline clusters should adopt the same shell so weight is consistent.

## 5. Inputs are one primitive

`bg-input` fill, `1px var(--border)`, 8px radius, ~8-10px padding, ~13.5px sans,
and a focus state that sets `border-color: var(--accent)`. The base
`input, select, textarea` rule gives every field the focus accent for free; a
filter field adds a leading search/type SVG via the `.picker-filter` shell.

## 6. State is a pill, not bare colored text

ok / fail / running / active read as a `.job-state`-shaped badge: ~11px, 999px
radius, `1px` border, a tinted background per state. Not bare `color: green`.

## 7. Empty and unsupported states are designed

Every list renders a real empty state (a centered icon + a one-line "do this
next" hint), never a blank scroll area or a lone `.sub` line. Use the
`emptyState(icon, text, hint)` helper (renders the `.empty-state` block).

## 8. Shared spacing and corner rhythm

8px for controls and rows, 12px for cards, generous section padding. Radii come
from the token scale (8px controls, 12px cards). No one-off `7px` selects or
bespoke badge radii.

---

Tokens live in `:root` / `[data-theme="light"]` at the top of `style.css`
(`--bg`, `--bg-raised`, `--bg-input`, `--border`, `--text`, `--text-dim`,
`--accent`, `--accent-soft`, `--green`/`--red`/`--yellow`). Use them; never
hardcode a hex that will not follow the theme (the one exception is the pairing
QR, which is deliberately pinned dark-on-light so a scanner can read it).
