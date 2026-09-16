"""Tests de la heurística de fusión de segmentos (merge_sentence_segments):
evita que una oración cortada por una pausa breve llegue partida al TTS."""

from dubbing_poc.segments import Segment, merge_sentence_segments


def test_merges_segment_cut_mid_sentence_by_short_gap():
    segments = [
        Segment(start=0.0, end=1.0, text="Esto es una frase"),
        Segment(start=1.2, end=2.0, text="que sigue después de una coma."),
    ]
    merged = merge_sentence_segments(segments)
    assert len(merged) == 1
    assert merged[0].text == "Esto es una frase que sigue después de una coma."
    assert merged[0].start == 0.0
    assert merged[0].end == 2.0


def test_does_not_merge_when_previous_ends_with_terminal_punctuation():
    segments = [
        Segment(start=0.0, end=1.0, text="Primera frase completa."),
        Segment(start=1.2, end=2.0, text="Segunda frase completa."),
    ]
    merged = merge_sentence_segments(segments)
    assert len(merged) == 2
    assert [s.text for s in merged] == [s.text for s in segments]


def test_does_not_merge_across_long_gap():
    segments = [
        Segment(start=0.0, end=1.0, text="Frase cortada"),
        Segment(start=3.0, end=4.0, text="pero después de un silencio largo."),
    ]
    merged = merge_sentence_segments(segments, max_gap=0.6)
    assert len(merged) == 2


def test_does_not_merge_if_result_exceeds_max_duration():
    segments = [
        Segment(start=0.0, end=8.0, text="Frase larga que empieza"),
        Segment(start=8.1, end=13.0, text="y sigue por mucho más tiempo."),
    ]
    merged = merge_sentence_segments(segments, max_merged_duration=12.0)
    assert len(merged) == 2


def test_chains_multiple_consecutive_fragments():
    segments = [
        Segment(start=0.0, end=1.0, text="Uno,"),
        Segment(start=1.1, end=2.0, text="dos,"),
        Segment(start=2.1, end=3.0, text="y tres."),
        Segment(start=3.5, end=4.5, text="Frase nueva."),
    ]
    merged = merge_sentence_segments(segments)
    assert len(merged) == 2
    assert merged[0].text == "Uno, dos, y tres."
    assert merged[0].start == 0.0
    assert merged[0].end == 3.0
    assert merged[1].text == "Frase nueva."


def test_empty_list_returns_empty():
    assert merge_sentence_segments([]) == []


def test_single_segment_unchanged():
    segments = [Segment(start=0.0, end=1.0, text="Sola.")]
    merged = merge_sentence_segments(segments)
    assert merged == segments
