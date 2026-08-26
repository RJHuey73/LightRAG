"""Tests for OMML run properties (m:rPr) -> LaTeX math alphabets."""

from xml.etree.ElementTree import fromstring

import pytest

from lightrag.parser.docx.omml.ommlparser import OMMLParser


MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _omath(run_properties: str, text: str = "R") -> str:
    """Build an m:oMath document holding a single run."""
    return (
        f'<m:oMath xmlns:m="{MATH_NS}">'
        f"<m:r>{run_properties}<m:t>{text}</m:t></m:r>"
        f"</m:oMath>"
    )


def _parse(run_properties: str, text: str = "R") -> str:
    return OMMLParser().parse(fromstring(_omath(run_properties, text)))


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("roman", "\\mathrm{R}"),
        ("script", "\\mathscr{R}"),
        ("fraktur", "\\mathfrak{R}"),
        ("double-struck", "\\mathbb{R}"),
        ("sans-serif", "\\mathsf{R}"),
        ("monospace", "\\mathtt{R}"),
    ],
)
def test_m_scr_selects_the_math_alphabet(script, expected):
    assert _parse(f'<m:rPr><m:scr m:val="{script}"/></m:rPr>') == expected


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("p", "\\mathrm{R}"),
        ("b", "\\mathbf{R}"),
        ("i", "\\mathit{R}"),
        ("bi", "\\boldsymbol{R}"),
    ],
)
def test_m_sty_selects_weight_and_slant(style, expected):
    assert _parse(f'<m:rPr><m:sty m:val="{style}"/></m:rPr>') == expected


@pytest.mark.parametrize("style", ["b", "bi"])
def test_bold_nests_around_a_script_alphabet(style):
    # No single LaTeX command is both blackboard-bold and bold, so the two nest.
    run_properties = (
        f'<m:rPr><m:scr m:val="double-struck"/><m:sty m:val="{style}"/></m:rPr>'
    )
    assert _parse(run_properties) == "\\boldsymbol{\\mathbb{R}}"


def test_non_bold_style_defers_to_the_script_alphabet():
    run_properties = '<m:rPr><m:scr m:val="fraktur"/><m:sty m:val="i"/></m:rPr>'
    assert _parse(run_properties) == "\\mathfrak{R}"


def test_m_nor_marks_literal_upright_text():
    assert _parse("<m:rPr><m:nor/></m:rPr>", text="max") == "\\mathrm{max}"


@pytest.mark.parametrize("off", ["0", "false", "off"])
def test_m_nor_switched_off_is_not_applied(off):
    assert _parse(f'<m:rPr><m:nor m:val="{off}"/></m:rPr>', text="max") == "max"


def test_m_scr_wins_over_m_nor():
    run_properties = '<m:rPr><m:scr m:val="script"/><m:nor/></m:rPr>'
    assert _parse(run_properties) == "\\mathscr{R}"


def test_unknown_values_are_ignored_rather_than_emitted():
    assert _parse('<m:rPr><m:scr m:val="wingdings"/></m:rPr>') == "R"
    assert _parse('<m:rPr><m:sty m:val="q"/></m:rPr>') == "R"


# --- Backwards compatibility -------------------------------------------------
# Runs that carry no styling must render exactly as they did before m:rPr was
# supported, so existing goldens stay valid.


def test_run_without_properties_is_unchanged():
    assert _parse("") == "R"


def test_empty_run_properties_add_nothing():
    assert _parse("<m:rPr></m:rPr>") == "R"


def test_whitespace_only_run_is_never_wrapped():
    # parse_t returns a bare space for an empty m:t; wrapping it would emit
    # a spurious \mathbb{ } into the equation.
    xml = (
        f'<m:oMath xmlns:m="{MATH_NS}">'
        f'<m:r><m:rPr><m:scr m:val="double-struck"/></m:rPr><m:t> </m:t></m:r>'
        f"</m:oMath>"
    )
    assert OMMLParser().parse(fromstring(xml)) == " "


def test_styling_composes_with_surrounding_structure():
    # A styled run used as a superscript base, to prove the wrapping survives
    # nesting. parse_s_sup always brace-wraps its base, hence the outer braces.
    xml = (
        f'<m:oMath xmlns:m="{MATH_NS}"><m:sSup>'
        f'<m:e><m:r><m:rPr><m:scr m:val="double-struck"/></m:rPr><m:t>R</m:t></m:r></m:e>'
        f"<m:sup><m:r><m:t>n</m:t></m:r></m:sup>"
        f"</m:sSup></m:oMath>"
    )
    assert OMMLParser().parse(fromstring(xml)) == "{\\mathbb{R}}^{n}"
