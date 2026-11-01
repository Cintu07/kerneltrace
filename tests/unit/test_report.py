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
