from multilingual_ime.keystroke_map_db import KeystrokeMappingDB

bopomofo_db = KeystrokeMappingDB(".\\multilingual_ime\\src\\bopomofo_keystroke_map.db")


def test_word_to_keystroke():
    assert bopomofo_db.word_to_keystroke("你") == "su3"
    assert bopomofo_db.word_to_keystroke("好") == "cl3"

    assert bopomofo_db.word_to_keystroke("你好") == None
    assert bopomofo_db.word_to_keystroke("Hello") == None

    assert bopomofo_db.word_to_keystroke("") == None
    assert bopomofo_db.word_to_keystroke(" ") == None
