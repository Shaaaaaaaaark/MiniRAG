from minirag.integrations.feishu import (
    FeishuDocument,
    blocks_to_markdown,
    find_feishu_document_url,
    parse_feishu_xml,
)


def test_parse_feishu_xml_preserves_structured_blocks() -> None:
    xml = """
    <title>CloudWAN Design</title>
    <h1 id="heading-1">API Design</h1>
    <p id="paragraph-1">Create <b>CloudWAN</b> resources.</p>
    <table id="table-1">
      <tbody>
        <tr><th>Name</th><th>Type</th></tr>
        <tr><td>project_id</td><td>string</td></tr>
      </tbody>
    </table>
    <pre id="code-1" lang="json"><code>{"enabled": true}</code></pre>
    <ul id="list-1"><li>Keep API names</li><li>Keep error codes</li></ul>
    <img id="image-1" token="img-token"/>
    <synced_reference id="sync-1" src-token="source-token"/>
    """

    title, blocks = parse_feishu_xml(xml)

    assert title == "CloudWAN Design"
    assert blocks[0].type == "heading"
    assert blocks[0].block_id == "heading-1"
    assert "project_id" in blocks[2].text
    assert "| Name | Type |" in blocks[2].text
    assert blocks[3].text.startswith("```json")
    assert "- Keep API names" in blocks[4].text
    assert blocks[5].text == "[图片: img-token]"
    assert blocks[6].text == "[同步块: source-token]"


def test_blocks_to_markdown_keeps_revision_and_block_anchor() -> None:
    title, blocks = parse_feishu_xml(
        '<title>Doc</title><h1 id="h1">Section</h1><p id="p1">Content</p>'
    )
    markdown = blocks_to_markdown(
        FeishuDocument(
            token="doc-token",
            url="https://example.feishu.cn/docx/doc-token",
            title=title,
            revision_id="42",
            blocks=blocks,
            raw_xml="",
        )
    )

    assert 'doc_token: "doc-token"' in markdown
    assert 'revision_id: "42"' in markdown
    assert "<!-- feishu-block-id: p1 -->" in markdown


def test_find_feishu_document_url() -> None:
    url = "https://bytedance.larkoffice.com/docx/AbCd1234?from=foo"
    assert find_feishu_document_url(f"请读取 {url} 的灰度章节") == url
