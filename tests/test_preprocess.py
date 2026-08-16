from src.data.preprocess import sent_preprocess


def test_lowercase_and_punctuation():
    assert sent_preprocess("Hello, World!") == "hello world"


def test_options_can_be_disabled():
    out = sent_preprocess("Hello, World!", lower=False, remove_punct=False)
    assert "Hello" in out


def test_emoji_removal():
    assert "🙂" not in sent_preprocess("nice day 🙂")


def test_number_handling():
    out = sent_preprocess("I have 3 cats", handle_nums=True)
    assert "<num>" in out or "<NUM>" in out
