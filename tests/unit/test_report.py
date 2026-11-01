from kerneltrace.reports import report as R


def test_table_cells_with_pipes_survive_markdown_and_html():
    md = R._table(["gate", "detail"], [["identity", "max |a - b| = 0"]])
    assert "\|a - b\|" in md
    html = R.markdown_to_html(md, "t")
    assert "<td>max |a - b| = 0</td>" in html
    assert html.count("<td>") == 2


def test_markdown_subset_renders():
    html = R.markdown_to_html("# T\n\n- a `x`\n\n```text\n<b>\n```\n\n![f](figures/a.svg)", "t")
    assert "<h1>T</h1>" in html and "<li>a <code>x</code></li>" in html
    assert "&lt;b&gt;" in html and '<img alt="f" src="figures/a.svg">' in html


def test_bundle_writes_lf_only(tmp_path):
    import pandas as pd

    from kerneltrace.reports.bundle import Bundle

    b = Bundle(tmp_path)
    paths = [b.json("a.json", {"x": [1, 2]}), b.jsonl("b.jsonl", [{"x": 1}, {"x": 2}]),
             b.text("c.md", "one\ntwo\n"), b.parquet("d.parquet", pd.DataFrame({"x": [1, 2]})).with_suffix(".csv")]
    for p in paths:
        assert b"\r\n" not in p.read_bytes(), p
