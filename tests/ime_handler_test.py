from multilingual_ime.ime_handler import IMEHandler

ime_handler = IMEHandler()

def test_reconstruct_sentence():
    test_text = "t g3bjo4dk4apple wathc"
    label_tokens = ["t ", "g3", "bjo4", "dk4", "apple", " ", "wathc"]

    possbile_sentences = ime_handler._reconstruct_sentence(test_text)
    for sentence in possbile_sentences:
        assert "".join(sentence) == test_text, "Reconstructed sentence is not same as input text"

    assert label_tokens in possbile_sentences, "Label tokens not in possible sentence"


def test_get_token_pool():
    test_text = "t g3bjo4dk4apple wathc"
    label_tokens = ["t ", "g3", "bjo4", "dk4", "apple", " ", "wathc"]

    token_pool = ime_handler._get_token_pool(test_text)
    token_pool = set([token for token in token_pool])
    for label_token in label_tokens:
        assert label_token in token_pool, f"{label_token} not in token pool"

def test_get_candidate():
    test_text = "t g3bjo4dk4apple wathc"
    label_tokens = ["t ", "g3", "bjo4", "dk4", "apple", " ", "wathc"]

    candidates = ime_handler.get_candidate(test_text)
    print(candidates)
    # assert False, f"candidates: {candidates}"
