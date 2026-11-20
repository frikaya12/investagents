"""Tests du masquage des lieux et de la construction des sections."""

from investagents.agents.masking import TOKEN, country_pattern, mask_places
from investagents.data.sections_corpus import sections_text


def test_mask_places_ner_and_country_list():
    text = "The project in Chad benefits Chadian farmers along the Niger River in Kenya."
    out = mask_places([text], country_names=["Kenya"])[0]
    for word in ["Chad ", "Chadian", "Kenya"]:
        assert word not in out
    assert TOKEN in out
    assert "farmers" in out  # le reste du texte est conservé


def test_country_pattern_ignores_case_and_short_names():
    pat = country_pattern(["Viet Nam", "Chad", "Mali", "XX"])
    assert pat.sub("#", "VIET NAM and chad") == "# and #"
    assert country_pattern([]) is None


def test_sections_text_masks_project_identifiers():
    raw = "x " * 300 + "The project development objective is to build roads. P123456 Rural Roads."
    out = sections_text(raw, "Rural Roads")
    assert "P123456" not in out and "Rural Roads" not in out
    assert "development objective" in out
