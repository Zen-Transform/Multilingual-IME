# Multilingual IME

## Dependency

Package manager: [Poetry](https://python-poetry.org/)

## Project Structure

- Datasets
  - Keystroke_Datasets
  - Plain_Text_Datasets
  - Test_Datasets
  - Train_Datasets
- multilingual_ime
  - core: core functions
  - data_preprocess: codes for data preprocessing
  - src: location for none code source object
  - *.py: main IME handler code
- references: storing referece paper or documents
- reports: storing system tesing report or log files
- scripts: short script for data generations or others
- tests: storing unit test code

### How to run script

```shell
# install package
poetry add [package]

# run module as script
python -m [module_name].[script]
```
## Contributer

1. 
