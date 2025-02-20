# Multilingual IME

![pypi](https://img.shields.io/pypi/v/multilingual_ime)

// Tag on release, oenssf, lastes build, ci/cd

Multilingual IME is a package of input method editor (IME) cores that support cross-typing between 3+ different input methods.

Current supported input method

* English
* Bopomofo (Zhuyin) 注音
* Cangjie 倉頡
* Pinyin 無聲調拼音

## Install

```shell
> pip install multlingual_ime
```

### Run Example

```shell
> pip install multlingual_ime
```



## Development

### Dependency

Package manager: [Poetry](https://python-poetry.org/)

### Project Structure

* Datasets
  * Keystroke_Datasets
  * Plain_Text_Datasets
  * Test_Datasets
  * Train_Datasets
* multilingual_ime
  * core: core functions
  * data_preprocess: codes for data preprocessing
  * src: location for none code source object
  * \*.py: main IME handler code
* references: storing referece paper or documents
* reports: storing system tesing report or log files
* scripts: short script for data generations or others
* tests: storing unit test code

### How to run script

```shell
# install package
poetry add [package]

# run module as script
python -m [module_name].[script]
```

## Related Project

* PolyKey: Input method editor on Windows
* PolyKey-Web: Input method editor as Chrome extension

## Bug Report

Please report any issue to [here](https://github.com/Zen-Transform/Multilingual-IME/issues).
