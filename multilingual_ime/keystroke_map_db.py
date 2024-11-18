from pathlib import Path
import sqlite3

from .trie import modified_levenshtein_distance
from .core.custom_decorators import lru_cache_with_doc

# from fuzzywuzzy import fuzz

MAX_LEVENSHTEIN_DISTANCE = 1


class KeystrokeMappingDB:
    def __init__(self, db_path: str):
        if not pathlib.Path(db_path).exists():
            raise FileNotFoundError(f"Database file {db_path} not found")

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._conn.create_function("levenshtein", 2, modified_levenshtein_distance)

    def get(self, keystroke: str) -> list[str]:
        self._cursor.execute(
            "SELECT keystroke, word, frequency FROM keystroke_map WHERE keystroke = ?",
            (keystroke,),
        )
        return self._cursor.fetchall()

    @lru_cache_with_doc(maxsize=128)
    def fuzzy_get(
        self, keystroke: str, max_distance: int = MAX_LEVENSHTEIN_DISTANCE
    ) -> list[str]:
        self._cursor.execute(
            f"SELECT keystroke, word, frequency FROM keystroke_map WHERE levenshtein(keystroke, ?) <= {max_distance}",
            (keystroke,),
        )
        return self._cursor.fetchall()

    @lru_cache_with_doc(maxsize=128)
    def get_closest(self, keystroke: str) -> list[tuple[str, str, int]]:
        """
        Get the closest words to the given keystroke.

        Args:
            keystroke (str): The keystroke to search for

        Returns:
            list: A list of **tuples (keystroke, word, frequency)** containing the closest words
        """

        for i in range(MAX_LEVENSHTEIN_DISTANCE + 1):
            result = self.fuzzy_get(keystroke, i)
            if result:
                return result
        return []

    def create_table(self):
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS keystroke_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keystroke TEXT,
                word TEXT,
                frequency INTEGER
            )
            """
        )
        self._conn.commit()

    def insert(self, keystroke: str, word: str, frequency: int):
        if (keystroke, word, frequency) not in self.get(keystroke):
            self._cursor.execute(
                "INSERT INTO keystroke_map (keystroke, word, frequency) VALUES (?, ?, ?)",
                (keystroke, word, frequency),
            )
            self._conn.commit()

    def insert_many(self, data: list[tuple[str, str, int]]):
        for keystroke, word, frequency in data:
            self.insert(keystroke, word, frequency)

    def __del__(self):
        self._conn.close()

    def keystroke_exists(self, keystroke: str) -> bool:
        return bool(self.get(keystroke))

    def is_word_within_distance(
        self, keystroke: str, distance: int = MAX_LEVENSHTEIN_DISTANCE
    ) -> bool:
        return self.closest_word_distance(keystroke) <= distance

    def closest_word_distance(self, keystroke: str) -> int:
        distance = 0
        while True:
            if self.fuzzy_get(keystroke, distance):
                return distance
            else:
                distance += 1

    def word_to_keystroke(self, word: str) -> str:
        self._cursor.execute(
            "SELECT keystroke FROM keystroke_map WHERE word = ?", (word,)
        )
        if word := self._cursor.fetchone():
            return word[0]
        else:
            return None


import pathlib

if __name__ == "__main__":
    db = KeystrokeMappingDB(
        pathlib.Path(__file__).parent / "src" / "bopomofo_keystroke_map.db"
    )
    print(db.get_closest("u04counsel"))
    print(db.closest_word_distance("u04counsel"))
    print(db.is_word_within_distance("u04counsel"))
