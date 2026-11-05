"""backend/tests/test_theme_default.py

Role: Pin the two things the light/dark theme duplicates across files it cannot share.

`frontend/public/theme-init.js` runs before React exists, as a blocking `<script src>` in the
document head, so it cannot `import` anything. It therefore repeats two facts that
`frontend/lib/theme.ts` also states, and a duplicated fact drifts:

  1. **The storage key.** If the two disagree, the pre-paint script reads a key nobody writes:
     a user who chose dark gets a light flash on every single navigation, and nothing fails.
  2. **The default.** Light, with the operating system not consulted. If the script regained a
     `prefers-color-scheme` fallback, a dark-machine visitor would get a dark compliance tool
     they never asked for — and the only symptom would be screenshots that differ between two
     people with no setting to explain it.

Both are invisible when broken, which is exactly why they are asserted here rather than left
to review. These are source-text checks, in the same style as
`backend/tests/security/test_xss.py`, because this repo has no JavaScript test runner — the
only frontend gate installed is `tsc --noEmit`, and neither of these is a type error.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BOOTSTRAP = REPO_ROOT / "frontend" / "public" / "theme-init.js"
THEME_MODULE = REPO_ROOT / "frontend" / "lib" / "theme.ts"


@pytest.fixture(scope="module")
def bootstrap() -> str:
    assert BOOTSTRAP.exists(), f"missing {BOOTSTRAP}"
    return BOOTSTRAP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def theme_module() -> str:
    assert THEME_MODULE.exists(), f"missing {THEME_MODULE}"
    return THEME_MODULE.read_text(encoding="utf-8")


def _code_lines(source: str) -> list[str]:
    """Lines with the `//` and `/* */` comments stripped out.

    Every claim below is about what the file *does*. The prose in these files discusses
    `prefers-color-scheme` at length precisely to explain why it is not used, so a naive
    substring search over the whole text would fail on the documentation.
    """
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    lines = []
    for line in without_blocks.splitlines():
        code = line.split("//", 1)[0].strip()
        if code:
            lines.append(code)
    return lines


class TestTheStorageKeyMatches:
    def test_the_bootstrap_reads_the_key_the_module_writes(
        self, bootstrap: str, theme_module: str
    ) -> None:
        """A mismatch is silent: the script reads a key nobody writes and light always wins."""
        declared = re.search(
            r"THEME_STORAGE_KEY\s*=\s*'([^']+)'", theme_module
        )
        assert declared is not None, "THEME_STORAGE_KEY is no longer a single-quoted literal"
        key = declared.group(1)

        used = [line for line in _code_lines(bootstrap) if "localStorage.getItem" in line]
        assert len(used) == 1, f"expected exactly one storage read, found {used}"
        assert f"'{key}'" in used[0], (
            f"the bootstrap reads {used[0]!r}, which is not THEME_STORAGE_KEY ({key!r}). "
            f"They are duplicated on purpose — a public/ asset cannot import — so both have "
            f"to change together."
        )


class TestLightIsTheDefault:
    """The requirement: dark happens only when somebody asks for it."""

    def test_the_bootstrap_adds_dark_only_for_a_stored_dark_choice(
        self, bootstrap: str
    ) -> None:
        toggles = [line for line in _code_lines(bootstrap) if "classList.toggle" in line]
        assert len(toggles) == 1, f"expected one class toggle, found {toggles}"
        statement = toggles[0]
        assert "'dark'" in statement
        assert "stored === 'dark'" in statement, (
            f"the pre-paint script no longer keys dark mode on an explicit stored choice: "
            f"{statement!r}"
        )

    def test_the_bootstrap_does_not_consult_the_operating_system(
        self, bootstrap: str
    ) -> None:
        """The regression that would be hardest to notice and easiest to reintroduce."""
        code = " ".join(_code_lines(bootstrap))
        for forbidden in ("matchMedia", "prefers-color-scheme"):
            assert forbidden not in code, (
                f"{forbidden} is back in the pre-paint script, so the OS decides the theme "
                f"again for anyone who has never chosen. Light is the default."
            )

    def test_resolve_theme_sends_everything_but_dark_to_light(
        self, theme_module: str
    ) -> None:
        """The same rule on the React side, where the toggle and the remount repair read it."""
        body = re.search(
            r"export function resolveTheme\([^)]*\)[^{]*\{(.*?)\n\}",
            theme_module,
            flags=re.DOTALL,
        )
        assert body is not None, "resolveTheme is no longer a plain exported function"
        statements = _code_lines(body.group(1))
        assert statements == ["return preference === 'dark' ? 'dark' : 'light';"], (
            f"resolveTheme does something other than 'dark means dark, everything else means "
            f"light': {statements}"
        )

    def test_no_frontend_file_reads_the_os_colour_preference(self) -> None:
        """Nowhere, in code, across the whole frontend.

        Narrow on purpose: the forbidden thing is `prefers-color-scheme`, not `matchMedia`.
        `components/ui/reveal.tsx` reads `prefers-reduced-motion` and should keep doing so —
        that is an accessibility preference the app genuinely must honour, and banning
        `matchMedia` outright would have caught it. The colour scheme is the one media
        preference this app deliberately ignores.
        """
        offenders: list[str] = []
        for pattern in ("*.ts", "*.tsx", "*.js", "*.css"):
            for path in sorted((REPO_ROOT / "frontend").rglob(pattern)):
                if "node_modules" in path.parts or ".next" in path.parts:
                    continue
                for number, line in enumerate(
                    _code_lines(path.read_text(encoding="utf-8")), 1
                ):
                    if "prefers-color-scheme" in line:
                        offenders.append(f"{path.name}:{number}: {line}")
        assert offenders == [], (
            "prefers-color-scheme is back in frontend code — the theme must not depend on "
            "the OS, and light is the default:\n" + "\n".join(offenders)
        )
