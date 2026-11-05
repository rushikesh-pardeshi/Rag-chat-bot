"""backend/tests/regression/test_coverage_map.py

Role: Keeps `BR_COVERAGE` honest.

`conftest.py` maps each requirement family in `WORKBREAKDOWN.md` § "Document requirements
covered" to the module that regresses it. A table like that decays in one direction only: a
module gets renamed, split or deleted, the table keeps naming it, and the Phase 6 gate
("Regression suites pass for BR-01 to BR-09") reads as satisfied because nothing contradicts
it. A missing file is not a failing test — it is no test at all, which is the harder kind of
gap to notice.

So these assertions are about the *map*, not about any requirement:

  * every module the table names exists on disk, and is in this package;
  * every regression module on disk is named by the table, so a suite cannot be added without
    saying which requirement it answers;
  * every module actually contains tests, because an empty file satisfies the check above
    while covering nothing;
  * the six numbered requirements are all present, and the two named families use the
    identifiers the repository actually uses rather than invented BR-07..BR-09 numbers.

The last point is the one worth stating twice. `WORKBREAKDOWN.md`'s gate says "BR-01 to BR-09"
and only BR-01..BR-06 are numbered anywhere in this workspace; see the `conftest.py` docstring.
This module asserts the table's *shape*, and deliberately does not assert nine numbered
requirements exist — that would encode a mapping nobody can check against a source document.
"""

from __future__ import annotations

import pathlib

import pytest

from backend.tests.regression.conftest import BR_COVERAGE

#: This package. `__file__` rather than a path built from the working directory, for the same
#: reason `harness.backend_modules()` does it that way: a scan of an empty tree passes every
#: assertion made over it.
PACKAGE = pathlib.Path(__file__).resolve().parent

#: Modules in this package that are not requirement suites, and why each is exempt.
NOT_A_SUITE = frozenset(
    {
        "__init__.py",
        "conftest.py",
        # This module. It regresses the map, not a requirement.
        "test_coverage_map.py",
    }
)


def suite_files() -> set[str]:
    found = {
        path.name
        for path in PACKAGE.glob("*.py")
        if path.name not in NOT_A_SUITE and path.name != "__pycache__"
    }
    assert found, f"no regression modules found under {PACKAGE}"
    return found


@pytest.mark.parametrize(("family", "entry"), sorted(BR_COVERAGE.items()))
def test_every_named_module_exists(family: str, entry: tuple[str, str]) -> None:
    """An entry pointing at a deleted module is a coverage claim with nothing behind it."""
    _, module = entry
    assert (PACKAGE / module).is_file(), f"{family} names {module}, which does not exist"


@pytest.mark.parametrize(("family", "entry"), sorted(BR_COVERAGE.items()))
def test_every_named_module_contains_tests(family: str, entry: tuple[str, str]) -> None:
    """An empty file satisfies "the module exists" and covers nothing."""
    _, module = entry
    source = (PACKAGE / module).read_text(encoding="utf-8")
    assert "async def test_" in source or "def test_" in source, f"{module} has no tests"


def test_every_suite_is_claimed_by_a_requirement() -> None:
    """The other direction: a module nobody's requirement points at.

    Not pedantry about tidiness. The gate is phrased as "the requirements are regressed", and a
    suite that answers no requirement means either the table is stale or the requirement is
    missing from `WORKBREAKDOWN.md` — both worth finding at the moment the file is added, while
    whoever added it still knows which requirement they had in mind.
    """
    claimed = {module for _, module in BR_COVERAGE.values()}
    unclaimed = suite_files() - claimed
    assert not unclaimed, f"regression modules not named in BR_COVERAGE: {sorted(unclaimed)}"


def test_the_six_numbered_requirements_are_all_covered() -> None:
    """BR-01..BR-06 — the only numbered business requirements in this workspace."""
    numbered = {identifier for identifier, _ in BR_COVERAGE.values()}
    for n in range(1, 7):
        assert f"BR-0{n}" in numbered, f"BR-0{n} has no regression module"


def test_no_invented_requirement_numbers() -> None:
    """BR-07, BR-08 and BR-09 are named by the Phase 6 gate and defined nowhere.

    If a numbered mapping is ever agreed it belongs in `WORKBREAKDOWN.md` first and in
    `BR_COVERAGE` second — at which point this test is the one to delete, deliberately, rather
    than a table that quietly grew three identifiers no document defines.
    """
    numbered = {identifier for identifier, _ in BR_COVERAGE.values()}
    assert not numbered & {"BR-07", "BR-08", "BR-09"}


def test_the_two_unnumbered_families_are_covered_by_name() -> None:
    """Analytics & Reporting is `BR-AN`; the rest carry NFR/C/R references."""
    assert BR_COVERAGE["Analytics & Reporting"][0] == "BR-AN"
    assert "Data Integrity" in BR_COVERAGE
    assert "Query Accuracy" in BR_COVERAGE
