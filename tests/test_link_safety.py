from backend.mineru.link_safety import assess_link


def test_assess_link_flags_plain_http() -> None:
    result = assess_link("http://example.com/manual")

    assert result.label == "需要谨慎打开"
    assert "不是 HTTPS 链接" in result.warnings


def test_assess_link_accepts_normal_https() -> None:
    result = assess_link("https://example.com/manual")

    assert result.label == "未发现明显风险"
    assert result.warnings == []

