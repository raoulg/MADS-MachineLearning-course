"""The loop of a study comes with the repository as a skill, in the folder each assistant
reads, and the skill points at the one file that holds the steps."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ".claude/skills/your-study/SKILL.md"
SKILLS = [ROOT / SKILL, ROOT / ".agents/skills/your-study/SKILL.md"]


def frontmatter_and_body(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"---\n(.*?)\n---\n(.*)", text, flags=re.DOTALL)
    assert match is not None, "the skill opens with a frontmatter block"
    keys = dict(
        line.split(": ", 1) for line in match.group(1).splitlines() if ": " in line
    )
    return keys, match.group(2)


def test_both_assistants_read_the_same_skill() -> None:
    claude, agents = (path.read_bytes() for path in SKILLS)
    assert claude == agents


def test_the_skill_names_itself_and_when_to_use_it() -> None:
    keys, _ = frontmatter_and_body(SKILLS[0].read_text())
    assert keys["name"] == "your-study"
    assert keys["description"].strip()


def test_the_skill_points_at_the_steps_and_holds_none_of_its_own() -> None:
    _, body = frontmatter_and_body(SKILLS[0].read_text())
    named = re.findall(r"`(docs/[^`]+\.md)`", body)
    assert named == ["docs/your-study.md"]
    assert (ROOT / named[0]).is_file()
    assert not re.search(r"^\s*\d+[.)]\s", body, flags=re.MULTILINE)


def ignored(path: str) -> bool:
    check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, check=False
    )
    return check.returncode == 0


def test_git_keeps_the_skill_and_ignores_the_rest_of_claude() -> None:
    assert not ignored(SKILL)
    assert ignored(".claude/settings.local.json")
    assert ignored("notebooks/1_representation_and_head/.claude/launch.json")
