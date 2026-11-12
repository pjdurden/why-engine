"""Asset resolution. Missing assets must degrade, never raise."""
from pathlib import Path

import pytest
from PIL import Image

from why_engine.art.direct import Shot
from why_engine.assets import Assets


@pytest.fixture
def library(tmp_path):
    icons, lib, scenes = tmp_path / "i", tmp_path / "l", tmp_path / "s"
    for d in (icons, lib, scenes):
        d.mkdir()
    for name in ("carp", "pig", "cow"):
        Image.new("RGB", (8, 8), (10, 10, 10)).save(icons / f"{name}.png")
    for name in ("everyman_disgusted", "bureaucrat_stamping"):
        Image.new("RGB", (8, 8), (10, 10, 10)).save(lib / f"{name}.png")
    for name in ("scene_a", "scene_b"):
        Image.new("RGB", (8, 8), (10, 10, 10)).save(scenes / f"{name}.png")
    return icons, lib, scenes


def make(library, topic="why we do not eat carp"):
    return Assets(*library, topic=topic)


def shot(archetype, sentence, index=0):
    return Shot(index=index, segment_id=1, sentence=sentence, archetype=archetype)


def test_icon_matches_a_noun_in_the_sentence(library):
    assert make(library).icon_for("twelve pigs went out").stem == "pig"


def test_icon_matches_a_plural(library):
    assert make(library).icon_for("the cows ate").stem == "cow"


def test_icon_returns_none_when_nothing_matches(library):
    assert make(library).icon_for("nothing relevant here") is None


def test_subject_falls_back_to_the_topic(library):
    """'more protein than twelve pigs' never says carp, but the video is about carp."""
    a = make(library).for_shot(shot("icon_multiply", "More protein than twelve pigs."))
    assert Path(a["subject_icon"]).stem == "carp"
    assert Path(a["count_icon"]).stem == "pig"


def test_comparison_resolves_the_two_sides_separately(library):
    a = make(library).for_shot(
        shot("measure_compare", "A cow needs eight kilograms and a carp needs one point two."))
    assert Path(a["left_icon"]).stem == "cow"
    assert Path(a["right_icon"]).stem == "carp"


def test_comparison_right_side_falls_back_to_the_topic(library):
    a = make(library).for_shot(shot("measure_compare", "A cow needs eight kilograms."))
    assert Path(a["right_icon"]).stem == "carp"


def test_versus_split_uses_the_cast(library):
    a = make(library).for_shot(shot("versus_split", "They chose pork instead."))
    assert Path(a["left_cast"]).stem == "everyman_disgusted"
    assert a["left_label"] and a["right_label"]


def test_scene_choice_is_deterministic_by_index(library):
    a = make(library)
    first = a.for_shot(shot("painted_scene", "down by the river", index=3))
    again = a.for_shot(shot("painted_scene", "down by the river", index=3))
    assert first["scene"] == again["scene"]


def test_different_indices_can_take_different_scenes(library):
    a = make(library)
    one = a.for_shot(shot("painted_scene", "river", index=0))["scene"]
    two = a.for_shot(shot("painted_scene", "river", index=1))["scene"]
    assert one != two


def test_missing_directories_do_not_raise(tmp_path):
    a = Assets(tmp_path / "none", tmp_path / "none", tmp_path / "none", topic="carp")
    assert a.for_shot(shot("icon_multiply", "twelve pigs")) is not None
    assert a.icon_for("carp") is None


def test_missing_scene_directory_leaves_the_scene_unset(tmp_path):
    a = Assets(tmp_path / "x", tmp_path / "y", tmp_path / "z", topic="carp")
    assert "scene" not in a.for_shot(shot("painted_scene", "by the river"))


def test_archival_plate_carries_its_licence_line(library):
    s = shot("archival_plate", "The fish had not changed.")
    s.claim_source = "usfc"
    a = make(library).for_shot(s)
    assert a["archival_licence"] == "PUBLIC DOMAIN"
    assert a["archival_caption"] == "USFC"


def test_scenes_come_only_from_the_directory_given(tmp_path):
    """Scenes are cycled by shot index with no regard to content, so a shared
    pool puts the previous video's pictures in this one: the carp scenes were
    sitting in the same directory as the school bus ones."""
    this_topic = tmp_path / "school-bus"
    other_topic = tmp_path / "carp"
    this_topic.mkdir()
    other_topic.mkdir()
    (this_topic / "scene_bus.png").write_bytes(b"")
    (other_topic / "scene_river.png").write_bytes(b"")

    assets = Assets(scenes=this_topic, topic="t")

    assert [p.name for p in assets.scenes] == ["scene_bus.png"]
