from data_preprocess.typo_generater import TypoGenerater
from data_preprocess.keystroke_converter import KeyStrokeConverter


def test_generate():
    assert TypoGenerater.generate("su3cl3" , error_type="random", error_rate=0) == "su3cl3"

    assert TypoGenerater.generate("你好", error_type="random", error_rate=0) == "你好"

def test_keystroke_converter():
    assert KeyStrokeConverter.convert("你好", convert_type="bopomofo") == "su3cl3"
    assert KeyStrokeConverter.convert("大市民", convert_type="bopomofo") == "284g4aup6"