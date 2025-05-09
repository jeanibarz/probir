import pytest
from common_utils import chunk_text

def test_chunk_text_empty():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   ", 100, 10) == ["   "] # Test with whitespace only

def test_chunk_text_single_chunk():
    text = "This is a short text."
    chunks = chunk_text(text, 100, 10)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_multiple_chunks_no_overlap():
    text = "This is a longer text that will be split into multiple chunks."
    #         012345678901234567890123456789012345678901234567890123456789
    #         0         1         2         3         4         5
    chunk_size = 20
    overlap = 0
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 4
    assert chunks[0] == "This is a longer tex"
    assert chunks[1] == "t that will be split"
    assert chunks[2] == " into multiple chunk" # Corrected expected text
    assert chunks[3] == "s." # Added 4th chunk assertion

def test_chunk_text_multiple_chunks_with_overlap():
    text = "This is a longer text that will be split into multiple chunks with some overlap."
    #         01234567890123456789012345678901234567890123456789012345678901234567890123456789
    #         0         1         2         3         4         5         6         7
    chunk_size = 30
    overlap = 10
    chunks = chunk_text(text, chunk_size, overlap)
    
    # Expected:
    # Chunk 1 (0-30): "This is a longer text that wil" (start: 0)
    # Next start: 30 - 10 = 20
    # Chunk 2 (20-50): "l be split into multiple chun" (start: 20)
    # Next start: 50 - 10 = 40
    # Chunk 3 (40-70): "ks with some overlap." (start: 40)
    # Next start: 70 - 10 = 60
    # Chunk 4 (60-eof): " overlap." (start: 60) -> text[60:78]

    assert len(chunks) == 4
    assert chunks[0] == text[0:30] # "This is a longer text that wil"
    assert chunks[1] == text[20:50] # "l be split into multiple chun"
    assert chunks[2] == text[40:70] # "ks with some overlap."
    assert chunks[3] == text[60:]   # " overlap."

def test_chunk_text_exact_multiple_no_overlap():
    text = "onetwothreefour" # 15 chars
    chunk_size = 5
    overlap = 0
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 3
    assert chunks[0] == "onetw"
    assert chunks[1] == "othre"
    assert chunks[2] == "efour"

def test_chunk_text_exact_multiple_with_overlap():
    text = "onetwothreefourfive" # 19 chars
    chunk_size = 7
    overlap = 3 # step = 4
    # 0-7: onetwot (0)
    # 4-11: wothree (4)
    # 8-15: reefour (8)
    # 12-19: urfive (12)
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 4
    assert chunks[0] == text[0:7]
    assert chunks[1] == text[4:11]
    assert chunks[2] == text[8:15]
    assert chunks[3] == text[12:19]

def test_chunk_text_chunk_size_larger_than_text():
    text = "short"
    chunks = chunk_text(text, 10, 2)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_invalid_chunk_size():
    # Expect warning and returning the original text as a single chunk
    assert chunk_text("some text", 0, 0) == ["some text"]
    assert chunk_text("some text", -5, 0) == ["some text"]

def test_chunk_text_invalid_overlap():
    # Expect warning or specific behavior based on current implementation
    # chunk_size <= chunk_overlap is the main guard.
    # For chunk_text("some text", 10, -1), 10 <= -1 is False. It will chunk.
    # Expected: ["some text"] because step will be 11, so only one chunk.
    assert chunk_text("some text", 10, -1) == ["some text"]
    # For chunk_text("some text", 10, 10), 10 <= 10 is True. Returns [text].
    assert chunk_text("some text", 10, 10) == ["some text"]
    # For chunk_text("some text", 10, 11), 10 <= 11 is True. Returns [text].
    assert chunk_text("some text", 10, 11) == ["some text"]

def test_chunk_text_unicode():
    text = "こんにちは世界" # Hello World in Japanese
    chunk_size = 3
    overlap = 1
    # Chars: こ ん に ち は 世 界
    # 0-3: こんに (0)
    # 2-5: にちは (2)
    # 4-7: は世界 (4)
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 3
    assert chunks[0] == "こんに"
    assert chunks[1] == "にちは" # Start index is by character, not byte
    assert chunks[2] == "は世界"

def test_chunk_text_overlap_equals_chunk_size_minus_one():
    text = "abcdefghij" # len 10
    chunk_size = 5
    overlap = 4 # step = 1
    # 0-5: abcde (0)
    # 1-6: bcdef (1)
    # 2-7: cdefg (2)
    # 3-8: defgh (3)
    # 4-9: efghi (4)
    # 5-10: fghij (5)
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 6
    assert chunks[0] == "abcde"
    assert chunks[1] == "bcdef"
    assert chunks[2] == "cdefg"
    assert chunks[3] == "defgh"
    assert chunks[4] == "efghi"
    assert chunks[5] == "fghij"

def test_chunk_text_final_chunk_smaller_than_chunk_size():
    text = "This is a test text." # 20 chars
    chunk_size = 8
    overlap = 2 # step = 6
    # 0-8: This is  (0)
    # 6-14: is a tes (6)
    # 12-20: test text. (12)
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 3
    assert chunks[0] == text[0:8]
    assert chunks[1] == text[6:14]
    assert chunks[2] == text[12:20]
    assert len(chunks[2]) == 8

    text2 = "This is a test text A" # 21 chars
    # 0-8: This is 
    # 6-14: is a tes
    # 12-20: test tex
    # 18-21: xt A
    chunks2 = chunk_text(text2, chunk_size, overlap) # step = 6
    # text2[0:8] = "This is "
    # text2[6:14] = "is a tes"
    # text2[12:20] = "test tex"
    # text2[18:21] = "xt A"
    assert len(chunks2) == 4
    assert chunks2[0] == text2[0:8]
    assert chunks2[1] == text2[6:14]
    assert chunks2[2] == text2[12:20]
    assert chunks2[3] == text2[18:21]
    assert len(chunks2[3]) == 3
