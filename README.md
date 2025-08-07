# Multilingual IME (MIME)

![pypi](https://img.shields.io/pypi/v/multilingual_ime)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multilingual_ime)

Multilingual IME is a package of input method editor (IME) core that leverage AI and algorithms to enable cross-typing between 3+ different input methods.
There are two derivative user interface implementations compatible with both Windows and the Chrome browser ([See related projects](#related-projects)).

Current supported input methods

| Input Methods | Languages | Dictionary size |
| ---- | ----- | :---- |
| English | English | 10,000 words |
| Bopomofo (Zhuyin) 注音 | Chinese | 15,7744 characters |
| Cangjie 倉頡 | Chinese | 6,308 characters |
| Pinyin 無聲調拼音 | Chinese | 6,235 characters|
| Japanese 日文 | Japanese | x |

| Languages | Vocabulary database |
| ---- | :---- |
| English | X |
| Chinese (Mandarin) | 12,645 phrases|

### Related Projects

* [MIME-win](https://github.com/Zen-Transform/MIME-win) : Input method editor on Windows
* [MIME-web](https://github.com/Zen-Transform/MIME-web) : Input method editor as Chrome extension

## Install

```shell
> pip install multilingual_ime
```

### Run Example

```shell
# Install package
> pip install multilingual_ime
# Run cli version of input method
> python -m multilingual_ime.ime_cli
```

## Development

* Package manager: [Poetry](https://python-poetry.org/)

### Project Structure

* Datasets
  * Keystroke_Datasets
  * Plain_Text_Datasets
  * Test_Datasets
  * Train_Datasets
* multilingual_ime
  * core: core functions
  * src: location for none codes source object
  * \*.py: main IME handler codes
* data_precrocess: data preprocessing library
* reports:  system test reports and log files
* scripts: short scripts for data generations and others
* tests: system performance test codes
* unit_tests: unit test codess

### How to run script

```shell
# install all dependencies
poetry install

# Add package to dependencies
poetry add [package]

# run module as script
python -m [module_name].[script]
```

## Bug Report

Please report any issue to [here](https://github.com/Zen-Transform/Multilingual-IME/issues).

## License

Multilingual IME is release under [MIT License](https://github.com/Zen-Transform/Multilingual-IME?tab=MIT-1-ov-file).
