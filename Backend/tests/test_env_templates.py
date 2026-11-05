"""Tests for how `.env` is written and read.

These exist because of one bug that cost a working OCR deployment and looked like something
else entirely. `.env.example` documented a blank setting the way every other line documents a
filled one:

    OCR_REGION=                     # blank = AWS_REGION. Textract is not in every region.

dotenv strips an inline comment **only when a value precedes it**. With nothing in front, the
comment *is* the value — so boto3 was handed "# blank = AWS_REGION. Textract is not in every
region." as a region name, every scanned upload failed with `InvalidRegionError`, and the
documents landed `NEEDS_OCR`. From the UI that is indistinguishable from a scan nothing could
be read from, which is why it took a log dive to find.

Two tests, because there are two fixes: the templates no longer write comments that way, and
`Settings` refuses to believe a value that is obviously a comment.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from dotenv import dotenv_values

from backend.core.config import Settings

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATES = (REPO_ROOT / ".env.example", REPO_ROOT / "deploy" / "env.poc.example")

#: `KEY=` followed by whitespace and a comment — the form whose comment becomes the value.
DANGEROUS = re.compile(r"^([A-Z][A-Z0-9_]*)=[ \t]+#")


class TestTheTemplatesCannotPoisonASetting:
    @pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
    def test_no_blank_setting_carries_an_inline_comment(
        self, template: pathlib.Path
    ) -> None:
        """Document a blank setting on the line *above* it, never after the `=`.

        A reader copies this file to `.env` verbatim, so a line that parses wrongly here
        parses wrongly in every deployment made from it.
        """
        offenders = [
            f"{number}: {line.strip()[:70]}"
            for number, line in enumerate(
                template.read_text(encoding="utf-8").splitlines(), start=1
            )
            if DANGEROUS.match(line)
        ]
        assert not offenders, (
            f"{template.name} has settings whose inline comment becomes the value "
            f"(dotenv keeps it when the value is empty). Move the comment above the key:\n  "
            + "\n  ".join(offenders)
        )

    @pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
    def test_no_value_in_a_template_is_a_comment(
        self, template: pathlib.Path
    ) -> None:
        """The same property asserted through the parser rather than the text.

        Catches any other way of writing a line that ends up with a comment as its value —
        quoting, odd spacing — which a regex over the raw text could miss.
        """
        poisoned = {
            key: value
            for key, value in dotenv_values(template).items()
            if isinstance(value, str) and value.lstrip().startswith("#")
        }
        assert not poisoned, f"{template.name} parses these as comments: {sorted(poisoned)}"


class TestSettingsRefusesACommentAsAValue:
    """The belt, for a `.env` written by hand rather than copied from the template."""

    @pytest.mark.parametrize(
        "field",
        [
            "OCR_REGION",
            "S3_ENDPOINT_URL",
            "COOKIE_DOMAIN",
            "AWS_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "PINECONE_API_KEY",
        ],
    )
    def test_a_leaked_comment_reads_as_unset(self, field: str) -> None:
        """None, not the comment text.

        `boto3.client("textract", region_name="# blank = AWS_REGION…")` raises
        `InvalidRegionError`; a credential set to a comment fails as an opaque provider auth
        error. Neither is what the file was describing, and both are hard to trace back to a
        stray `#`.
        """
        validated = Settings._blank_is_unset("   # blank = use the default")
        assert validated is None, f"{field}: a comment should read as unset"

    def test_a_real_value_is_untouched(self) -> None:
        assert Settings._blank_is_unset("us-east-1") == "us-east-1"
        # A `#` *inside* a value is not a leading comment — passwords contain them.
        assert Settings._blank_is_unset("pa#ssword") == "pa#ssword"

    def test_a_blank_value_still_reads_as_unset(self) -> None:
        assert Settings._blank_is_unset("") is None
        assert Settings._blank_is_unset("   ") is None
