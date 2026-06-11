import pytest

from deep_researcher.tools.literature import _parse_arxiv_feed, _reconstruct_abstract

pytestmark = pytest.mark.asyncio(loop_scope="function")

_ARXIV_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2502.13138v1</id>
    <title>AIDE: AI-Driven Exploration
     in the Space of Code</title>
    <summary>We frame ML engineering as a code optimization problem.</summary>
    <published>2025-02-18T00:00:00Z</published>
    <author><name>Alice A</name></author>
    <author><name>Bob B</name></author>
    <arxiv:primary_category term="cs.AI"/>
  </entry>
</feed>"""


def test_parse_arxiv_feed():
    papers = _parse_arxiv_feed(_ARXIV_SAMPLE)
    assert len(papers) == 1
    p = papers[0]
    assert p["arxiv_id"] == "2502.13138v1"
    assert p["title"] == "AIDE: AI-Driven Exploration in the Space of Code"
    assert p["authors"] == ["Alice A", "Bob B"]
    assert p["primary_category"] == "cs.AI"
    assert p["published"] == "2025-02-18"


def test_reconstruct_abstract():
    inverted = {"Hello": [0], "world": [1], "again": [2]}
    assert _reconstruct_abstract(inverted) == "Hello world again"
    assert _reconstruct_abstract(None) is None
