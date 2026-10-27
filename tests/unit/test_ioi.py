from kerneltrace.tasks import ioi


def test_clean_and_corrupt_differ_only_at_second_subject():
    axes = dict(template=0, order="IO_first", io="Mary", s="John", place="store", object="drink")
    c = ioi.render(axes)
    assert c.clean == "When Mary and John went to the store, John gave a drink to"
    assert c.corrupt == "When Mary and John went to the store, Mary gave a drink to"
    assert (c.answer, c.distractor) == (" Mary", " John")
    words_clean, words_corrupt = c.clean.split(), c.corrupt.split()
    assert sum(a != b for a, b in zip(words_clean, words_corrupt)) == 1


def test_order_axis_swaps_first_mentions():
    axes = dict(template=0, order="S_first", io="Mary", s="John", place="store", object="drink")
    assert ioi.render(axes).clean.startswith("When John and Mary")


def test_every_template_has_all_slots():
    for t in ioi.TEMPLATES:
        for slot in ("{N1}", "{N2}", "{S2}", "{PLACE}", "{OBJECT}"):
            assert slot in t, (t, slot)
        assert t.endswith(" to")


def test_splits_are_disjoint():
    assert not set(ioi.DISCOVERY_NAMES) & set(ioi.HOLDOUT_NAMES)
    d = ioi.axis_space("discovery")
    h = ioi.axis_space("holdout")
    assert not set(d.values["template"]) & set(h.values["template"])
    assert not set(d.values["io"]) & set(h.values["io"])


def test_generation_is_deterministic_and_in_split():
    a = ioi.generate("holdout", 30, seed=0)
    b = ioi.generate("holdout", 30, seed=0)
    assert [c.case_id for c in a] == [c.case_id for c in b]
    assert len({c.case_id for c in a}) == 30
    assert all(c.template in ioi.HOLDOUT_TEMPLATES and c.split == "holdout" for c in a)
    assert all(c.axes["io"] != c.axes["s"] for c in a)


def test_names_filter_applies():
    space = ioi.axis_space("discovery", names_ok=lambda n: n != "Mary")
    assert "Mary" not in space.values["io"]
