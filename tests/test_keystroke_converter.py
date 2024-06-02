
from multilingual_ime.data_preprocess.keystroke_converter import KeyStrokeConverter

def test_keystroke_converter():
    assert KeyStrokeConverter.convert("你好棒", convert_type="cangjie") == "onf vnd dqkq "
    
def test_keystroke_converter2():
    assert KeyStrokeConverter.convert("僅頒行政院長陳建仁今\n（16）日出席\n「112年鳳凰獎楷模表揚典禮」，頒獎表揚74名獲獎義消",
                                    convert_type="cangjie") == "otlm chmbc hommn mmok nljmu smv nldw nklq omm oin \nyyyai 16yyyaj a uu itlb \nyyyaa 112oq hnmaf hnhag viik dppa dtak qmv qamh tbc iftwt yyyab zxab chmbc viik qmv qamh 74nir khtoe viik tghqi efb "
