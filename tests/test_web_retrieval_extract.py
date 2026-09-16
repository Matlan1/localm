# SPDX-License-Identifier: AGPL-3.0-or-later
"""Main-content extraction (localm.web_retrieval.extract), including the
audited WEB-FUNC-002 reproduction: about 7,000 characters of site chrome before
a short article."""

from __future__ import annotations

import time

import pytest

from localm.netpolicy import html_to_text
from localm.web_retrieval import extract_page, html_to_main_text
from tests._web_retrieval_fixtures import (
    ANSWER,
    article_html,
    html_page,
    nav_heavy_page,
)

_LONG = " ".join(f"Sentence number {i} of the body text." for i in range(12))


class TestRegionSelection:
    def test_main_preferred_and_chrome_outside_it_dropped(self):
        page = extract_page(html_page(
            "<header>Site Name Login</header><nav>Home Products About</nav>"
            f"<main><p>{_LONG}</p></main>"
            "<aside>Sidebar promo</aside><footer>Imprint Copyright</footer>"))
        assert page.region == "main"
        assert _LONG in page.text
        for chrome in ("Site Name", "Home Products", "Sidebar", "Imprint"):
            assert chrome not in page.text

    def test_role_main_counts_as_main(self):
        page = extract_page(html_page(
            f"<nav>Menu one two</nav><div role='main'><p>{_LONG}</p></div>"))
        assert page.region == "main"
        assert "Menu one" not in page.text

    def test_largest_main_wins(self):
        page = extract_page(html_page(
            f"<main><p>tiny</p></main><main><p>{_LONG}</p></main>"))
        assert page.region == "main" and _LONG in page.text

    def test_article_used_when_no_main_and_its_header_kept(self):
        page = extract_page(html_page(
            "<header>Site chrome header</header>"
            f"<article><header><h1>The Headline</h1></header><p>{_LONG}</p>"
            "<nav>in-article nav</nav><footer>article footer</footer></article>"
            "<footer>Site footer</footer>"))
        assert page.region == "article"
        assert "The Headline" in page.text
        assert _LONG in page.text
        for dropped in ("Site chrome header", "in-article nav",
                        "article footer", "Site footer"):
            assert dropped not in page.text

    def test_largest_article_wins_on_a_listing_page(self):
        page = extract_page(html_page(
            "<article><p>teaser one</p></article>"
            f"<article><p>{_LONG}</p></article>"
            "<article><p>teaser two</p></article>"))
        assert page.region == "article"
        assert _LONG in page.text and "teaser one" not in page.text

    def test_forms_removed_inside_main(self):
        page = extract_page(html_page(
            f"<main><p>{_LONG}</p><form><label>Email</label>"
            "<input name='e'><button>Subscribe now</button></form></main>"))
        assert "Subscribe now" not in page.text and "Email" not in page.text

    def test_short_main_falls_back_to_body(self):
        page = extract_page(html_page(
            f"<main><p>Loading...</p></main><div><p>{_LONG}</p></div>"))
        assert page.region == "body"
        assert _LONG in page.text


class TestBodyFallback:
    def test_header_footer_and_link_dense_menus_removed(self):
        menu = "".join(f"<li><a href='/{i}'>Menu item {i}</a></li>"
                       for i in range(20))
        page = extract_page(html_page(
            "<header>Top header text</header>"
            f"<div class='menu'><ul>{menu}</ul></div>"
            f"<div class='content'><p>{_LONG}</p></div>"
            "<footer>Bottom footer text</footer>"))
        assert page.region == "body"
        assert _LONG in page.text
        assert "Menu item" not in page.text
        assert "Top header" not in page.text and "Bottom footer" not in page.text

    def test_paragraph_with_a_few_links_is_kept(self):
        page = extract_page(html_page(
            f"<div><p>{_LONG} See <a href='/a'>this</a>, <a href='/b'>that</a> "
            "and <a href='/c'>more</a>.</p></div>"))
        assert _LONG in page.text and "this" in page.text

    def test_all_link_directory_falls_back_to_plain_text(self):
        links = "".join(f"<li><a href='/{i}'>Directory entry {i}</a></li>"
                        for i in range(30))
        page = extract_page(html_page(f"<div><ul>{links}</ul></div>"))
        assert "Directory entry 3" in page.text

    def test_nested_menu_pruned_but_wrapping_div_kept(self):
        menu = "".join(f"<li><a href='/{i}'>Nav {i}</a></li>" for i in range(10))
        page = extract_page(html_page(
            f"<div class='wrap'><div class='nav'><ul>{menu}</ul></div>"
            f"<p>{_LONG}</p></div>"))
        assert _LONG in page.text and "Nav 3" not in page.text


class TestNoiseRemoval:
    @pytest.mark.parametrize("markup", [
        "<div hidden>HIDDEN TEXT</div>",
        "<div aria-hidden='true'>HIDDEN TEXT</div>",
        "<div style='display:none'>HIDDEN TEXT</div>",
        "<div style='color:red; visibility: hidden'>HIDDEN TEXT</div>",
        "<div role='navigation'>HIDDEN TEXT</div>",
        "<div role='Banner'>HIDDEN TEXT</div>",
        "<div role='contentinfo'>HIDDEN TEXT</div>",
        "<div role='complementary'>HIDDEN TEXT</div>",
        "<script>HIDDEN TEXT</script>",
        "<style>HIDDEN TEXT</style>",
        "<noscript>HIDDEN TEXT</noscript>",
        "<template>HIDDEN TEXT</template>",
        "<svg><text>HIDDEN TEXT</text></svg>",
        "<!-- HIDDEN TEXT -->",
    ])
    def test_removed_everywhere(self, markup):
        page = extract_page(html_page(f"<main>{markup}<p>{_LONG}</p></main>"))
        assert "HIDDEN TEXT" not in page.text
        assert _LONG in page.text

    def test_head_content_never_leaks_into_text(self):
        page = extract_page(
            "<html><head><title>T</title><meta name='d' content='META'>"
            f"<link rel='x' href='/l'></head><body><p>{_LONG}</p></body></html>")
        assert "META" not in page.text and _LONG in page.text


class TestTextRendering:
    def test_paragraphs_separated_by_blank_lines_and_br_by_newline(self):
        page = extract_page(html_page(
            "<div><p>First para.</p><p>Second<br>line.</p></div>"))
        assert page.text == "First para.\n\nSecond\nline."

    def test_table_cells_spaced_and_rows_on_lines(self):
        page = extract_page(html_page(
            "<table><tr><th>Day</th><th>Time</th></tr>"
            "<tr><td>Mon</td><td>10:00</td></tr></table>"))
        assert "Day Time" in page.text
        assert "Mon 10:00" in page.text
        assert page.text.index("Day Time") < page.text.index("Mon 10:00")
        assert "\n" in page.text

    def test_whitespace_collapsed_and_entities_decoded(self):
        page = extract_page(html_page(
            "<p>a &amp; b\t\t   c&nbsp;d &lt;e&gt;</p>"))
        assert page.text == "a & b c d <e>"

    def test_unicode_preserved(self):
        page = extract_page(html_page("<p>Grüße aus Linz. 東京へようこそ.</p>"))
        assert page.text == "Grüße aus Linz. 東京へようこそ."

    def test_unclosed_and_mismatched_tags(self):
        page = extract_page("<div><p>one<p>two</div>after</span>")
        assert "one" in page.text and "two" in page.text and "after" in page.text

    def test_malformed_markup_never_raises(self):
        for bad in ("<div><p>ok<", "<<<>>>", "<a href='x", "", None):
            assert isinstance(extract_page(bad).text, str)

    def test_thousands_of_nested_unclosed_tags_do_not_overflow(self):
        deep = "<div>" * 5000 + f"<p>{_LONG}</p>" + "</div>" * 5000
        page = extract_page(html_page(deep))
        assert _LONG in page.text

    def test_main_within_the_depth_cap_is_selected(self):
        deep = "<div>" * 150 + f"<main><p>{_LONG}</p></main>" + "</div>" * 150
        page = extract_page(html_page(f"<nav>Menu one two</nav>{deep}"))
        assert page.region == "main"
        assert _LONG in page.text and "Menu one" not in page.text

    def test_main_beyond_the_depth_cap_survives_through_the_body_fallback(self):
        deep = "<div>" * 300 + f"<main><p>{_LONG}</p></main>" + "</div>" * 300
        page = extract_page(html_page(f"<nav>Menu one two</nav>{deep}"))
        assert page.region == "body"
        assert _LONG in page.text and "Menu one" not in page.text

    def test_content_after_a_premature_body_end_tag_is_kept(self):
        page = extract_page(
            f"<html><body><p>intro</p></body><div><p>{_LONG}</p></div></html>")
        assert "intro" in page.text and _LONG in page.text

    def test_adversarial_nesting_is_processed_in_linear_cpu_time(self):
        markup = "<div>" * 150_000 + _LONG
        started = time.process_time()
        page = extract_page(markup)
        cpu = time.process_time() - started
        assert _LONG in page.text
        assert cpu < 6.0, f"extract_page used {cpu:.1f}s of CPU"


class TestFormWrappedPages:
    def test_page_wrapped_in_one_form_is_extracted(self):
        page = extract_page(html_page(
            f"<form id='aspnetForm'><div><h1>Title</h1><p>{_LONG}</p>"
            f"<p>{_LONG}</p></div></form>"))
        assert _LONG in page.text
        assert "Title" in page.text

    def test_main_wrapped_in_one_form_is_extracted(self):
        page = extract_page(html_page(
            f"<nav>Menu one two</nav><main><form><div><p>{_LONG}</p>"
            f"<p>{_LONG}</p></div></form></main>"))
        assert page.region == "main"
        assert _LONG in page.text and "Menu one" not in page.text

    def test_small_form_inside_content_is_still_removed(self):
        page = extract_page(html_page(
            f"<main><p>{_LONG}</p><p>{_LONG}</p><form><label>Email</label>"
            "<button>Subscribe now</button></form></main>"))
        assert "Subscribe now" not in page.text and _LONG in page.text

    def test_html_to_main_text_is_the_text(self):
        markup = html_page(f"<main><p>{_LONG}</p></main>")
        assert html_to_main_text(markup) == extract_page(markup).text


class TestTitle:
    def test_title_tag_wins(self):
        page = extract_page(
            "<html><head><title>  Page   Title </title>"
            "<meta property='og:title' content='OG'></head>"
            "<body><h1>H1</h1></body></html>")
        assert page.title == "Page Title"

    def test_og_title_when_no_title_tag(self):
        page = extract_page(
            "<html><head><meta property='og:title' content='OG Title'></head>"
            "<body><h1>H1</h1></body></html>")
        assert page.title == "OG Title"

    def test_first_heading_when_nothing_else(self):
        assert extract_page("<body><h2>Only Heading</h2></body>").title == \
            "Only Heading"

    def test_empty_when_nothing(self):
        assert extract_page("<body><p>text</p></body>").title == ""

    def test_title_capped(self):
        assert len(extract_page(f"<title>{'x' * 1000}</title>").title) == 300


class TestWebFunc002Reproduction:
    """About 7,000 characters of chrome before a short article. The old
    whole-body stripper puts the answer past the 6,000-character prefix that
    chat forwards; main-content extraction puts it near the start."""

    @pytest.mark.parametrize("mode", ["semantic", "menus"])
    def test_answer_moves_from_past_6000_to_the_front(self, mode):
        markup = nav_heavy_page(mode)
        legacy = html_to_text(markup)
        assert legacy.find(ANSWER) >= 6000, "fixture does not reproduce the audit"
        page = extract_page(markup)
        assert ANSWER in page.text
        assert page.text.find(ANSWER) < 2500
        assert len(page.text) < len(legacy) // 2

    def test_semantic_variant_selects_main(self):
        assert extract_page(nav_heavy_page("semantic")).region == "main"

    def test_menus_variant_uses_body_fallback_with_menus_pruned(self):
        page = extract_page(nav_heavy_page("menus"))
        assert page.region == "body"
        assert "Section 40 overview" not in page.text

    def test_plain_boilerplate_cannot_be_pruned_and_is_kept_for_chunking(self):
        page = extract_page(nav_heavy_page("plain"))
        assert ANSWER in page.text
        assert page.text.find(ANSWER) >= 6000

    def test_article_fixture_alone_is_short(self):
        assert len(extract_page(html_page(article_html())).text) < 3000
