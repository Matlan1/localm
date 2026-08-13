# localm GUI design standard

The web GUI (`localm/plugins/gui/static/`) should feel like one coherent, quiet,
professional dark app. The file/folder picker (`app/picker.js`) and the grouped
settings page (`pages/settings.js`) are the reference surfaces: match them.

This is the checklist. When you add or touch a GUI surface, it should already
follow these nine rules. Everything here uses tokens and helpers that already
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
- Action buttons use an icon, including "new/add" ones (the `plus` icon, not a
  literal `+`), so the whole chrome reads as one icon set. A close/remove `x` may
  stay ASCII text on a dense inline affordance (a chip/tag remove): it is crisp
  and conventional there.

Color carries meaning in two independent registers, and they must not be mixed:

- CATEGORY hue = which area (nav icons, and a card-head / section-head icon). Muted
  and shared-lightness (`--cat-blue/-cyan/-green/-teal/-violet/-amber/-slate`), applied
  via `.nav-ic[data-icon=...]` or a `.cat-ic.cat-*` class. Chrome only.
- CONTENT-TYPE hue = what kind of thing a row is (a model, a doc, a video, a track,
  a plugin). Applied to a row's leading icon via `.ic-model/-folder/-doc/-media/-audio/
  -video/-code/-plugin`. Content only. A data-table name cell leads with one inside
  `.name-cell`; a `.disc-repo` history row leads with one in its head.

## 2. Every interactive row has hover + an inset accent bar when active

The picker row is canon for a plain list row: hover and active both tint the
background, and active adds the inset accent bar:

```css
.thing:hover, .thing.active { background: var(--bg-input); }
.thing.active { box-shadow: inset 2px 0 0 var(--accent); }
```

A `.data-table` active row (`tr:has(.active-tag)`) uses the same inset bar but
with `background: var(--accent-soft)` instead of `--bg-input`, so an active
table row reads a shade stronger than an active picker row. Nav buttons and
settings-nav links (primary navigation, not a data row) go further still,
pairing that same accent-soft fill and inset bar with `color: var(--accent)`
and `font-weight: 600`:

```css
.thing:hover  { background: var(--bg-input); color: var(--text); }
.thing.active { background: var(--accent-soft); box-shadow: inset 2px 0 0 var(--accent); color: var(--accent); font-weight: 600; }
```

## 3. Two button tiers plus danger, one vocabulary

- `.btn-primary` (filled accent, white text) for the one main action.
- `.btn-secondary` (outline, dim text) for everything else.
- `.btn-danger` (red outline that FILLS red on hover) for destructive actions.

Inside a dense `.data-table`, the compact `.data-table button` styling wins;
there, use `.primary` / `.danger` modifiers (accent / red that fill on hover).
Never ship a class-less `el("button", "", ...)`.

## 4. Sections are cards, with a `.card-head`

`bg-raised`, `1px var(--border)`, 12px radius, generous padding, and `.sub` helper
text under labels. The head is a `.card-head`, not a lone `<h3>`: a category-hued
leading icon + `.card-head-text` (the `<h3>`, optionally a one-line `.card-desc`) +
a bottom divider, so no card shows a bare grey title. Every content-page card and
settings section uses it (the schema-driven settings sections wrap their existing
`.settings-section-head` in a `.card-head`). Loose inline clusters should adopt the
same shell so weight is consistent.

## 5. Inputs are one primitive

`bg-input` fill, `1px var(--border)`, 8px radius, ~8-10px padding, ~13.5px sans,
and a focus state that sets `border-color: var(--accent)`. The base
`input, select, textarea` rule gives every field the focus accent for free; a
filter field adds a leading search/type SVG via the `.picker-filter` shell.

## 6. State is a pill, not bare colored text

ok / fail / running / active read as a `.job-state`-shaped badge: ~11px, 999px
radius, `1px` border, a tinted background per state. Not bare `color: green`, and
not a `(status)` parenthetical. The run-status variants are `.job-state.st-ok` /
`.st-error` / `.st-skipped` / `.st-running` / `.st-pending` / `.st-paused`, their
tints `color-mix`ed from the theme tokens so both themes follow the palette.

## 7. Empty and unsupported states are designed

Every list renders a real empty state (a centered icon + a one-line "do this
next" hint), never a blank scroll area or a lone `.sub` line. Use the
`emptyState(icon, text, hint)` helper (renders the `.empty-state` block).

## 8. Shared spacing and corner rhythm

8px for controls and rows, 12px for cards, generous section padding. Radii come
from the token scale (8px controls, 12px cards). No one-off `7px` selects or
bespoke badge radii.

## 9. Help text says what the control does, and what changes if you alter it

Nothing else. Rationale, threat models, upstream issue numbers, history and
"why it is off by default" belong in a code comment beside the field, or in the
docs. A control's help is read while deciding; a paragraph is not read at all,
so a 452-character warning protects nobody.

- **Target 150 characters. Hard cap 200**, enforced over `CORE_FIELDS`,
  `MEDIA_PLUGIN_FIELDS` and `TTS_FIELDS` by `tests/test_settings_help_budget.py`.
  The cap covers `HIDDEN` fields too: HIDDEN is a rendering decision that gets
  reversed, and a field that becomes visible must not bring a wall of text with
  it.
- **Lead with the consequence.** "Lowering this risks a native crash" before the
  explanation of what the number reserves. If the reader stops after one
  sentence, that sentence should be the one that matters.
- **Name the setting you mean, never its position.** No "the option above", no
  "the toggles below". `.settings-fields` is a two-column grid, so the next
  field renders to the *right*; and a setting can move to another nav group
  entirely. Positional copy is false the moment anything moves, and it has been
  false while standing still.
- **Do not explain the UI's own gating to the only people who can read it.**
  "Shown only when more than one GPU is detected", rendered under a control that
  appears only on a multi-GPU box, tells its whole audience what they already
  know.
- **A shared explanation goes in the panel intro once**, not repeated per field.
  Five load/timeout fields once carried the same paragraph five times.

When you cut, move the removed reasoning into a why-comment at the schema site.
It is usually the most valuable text in the field; it was just in the wrong
place.

---

Tokens live in `:root` / `[data-theme="light"]` at the top of `style.css`
(`--bg`, `--bg-raised`, `--bg-input`, `--border`, `--text`, `--text-dim`,
`--accent`, `--accent-soft`, `--green`/`--red`/`--yellow`). Use them; never
hardcode a hex that will not follow the theme (the one exception is the pairing
QR, which is deliberately pinned dark-on-light so a scanner can read it).
