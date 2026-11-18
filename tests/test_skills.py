"""The loop of a study comes with the repository as a skill, in the folder each assistant
reads, and the skill points at the one file that holds the steps."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = [
    ROOT / ".claude/skills/your-study/SKILL.md",
    ROOT / ".agents/skills/your-study/SKILL.md",
]


def test_both_assistants_read_the_same_skill() -> None:
    claude, agents = (path.read_bytes() for path in SKILLS)
    assert claude == agents


def test_the_skill_names_when_to_use_it() -> None:
    text = SKILLS[0].read_text()
    frontmatter = re.match(r"---\n(.*?)\n---\n", text, flags=re.DOTALL)
    assert frontmatter is not None
    assert "name: your-study" in frontmatter.group(1)
    assert "description: " in frontmatter.group(1)


def test_the_file_the_skill_points_to_exists() -> None:
    named = re.findall(r"`(docs/[^`]+\.md)`", SKILLS[0].read_text())
    assert named == ["docs/your-study.md"]
    assert (ROOT / named[0]).is_file()
