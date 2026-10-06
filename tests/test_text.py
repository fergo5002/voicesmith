from voicesmith import text


def test_normalise_expands_numbers_and_strips_punctuation():
    assert text.normalise("It's 5 PM, and 25% of 1,200 people!") == "it's five pm and twenty five percent of one thousand two hundred people"


def test_normalise_reads_years_like_people_do():
    assert text.normalise("In 1984 and 2026.") == "in nineteen eighty four and twenty twenty six"


def test_normalise_drops_paralinguistic_tags():
    assert text.normalise("Well [laugh] that worked") == "well that worked"


def test_wer_and_cer_ignore_case_and_punctuation():
    assert text.wer("Hello, world.", "hello world") == 0.0
    assert text.cer("Hello, world.", "hello world") == 0.0
    assert text.wer("the cat sat", "the cat sat sat") == 1 / 3


def test_cer_is_gentler_than_wer_on_near_misses():
    ref, hyp = "brightwater runs a winback", "brightwatter runs a win back"
    assert text.cer(ref, hyp) < text.wer(ref, hyp)


def test_repeated_ngram_flags_a_loop_but_not_a_scripted_repeat():
    assert text.repeated_ngram("we will win", "we will win we will win") == "we will win"
    assert text.repeated_ngram("go go go go", "go go go go") is None


def test_sentences_keep_abbreviations_together():
    s = text.sentences("Mr. Smith arrived. He said hello! Did it work? Yes.")
    assert s == ["Mr. Smith arrived.", "He said hello!", "Did it work?", "Yes."]


def test_chunk_packs_whole_sentences_and_never_exceeds_limit():
    para = " ".join(f"Sentence number {i} is here." for i in range(20))
    chunks = text.chunk(para, 80)
    assert all(len(c) <= 80 for c in chunks)
    assert " ".join(chunks) == para


def test_chunk_splits_an_overlong_sentence_at_a_comma():
    long = "This clause is first, " * 10 + "and this is the end."
    chunks = text.chunk(long, 60)
    assert all(len(c) <= 60 for c in chunks)
    assert text.normalise(" ".join(chunks)) == text.normalise(long)
