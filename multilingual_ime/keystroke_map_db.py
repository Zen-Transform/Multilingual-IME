"""
A module for interacting with the keystroke mapping database.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Union

from .core.F import modified_levenshtein_distance
from .core.custom_decorators import lru_cache_with_doc


class KeystrokeMappingDB:
    """
    A class for interacting with the keystroke mapping database.
    Database file must be in SQLite format.

    Args:
        db_path (str): The path to the database (Sqlite DB) file.
    """

    def __init__(self, db_path: Union[str, Path]):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database file {db_path} not found")

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._lock = threading.Lock()
        self._conn.create_function("levenshtein", 2, modified_levenshtein_distance)

    def get_word(self, keystroke: str) -> list[tuple[str, str, int]]:
        """
        Get the **(keystroke, word, frequency)** tuple with the given keystroke from the database.

        Args:
            keystroke (str): The keystroke to search for.

        Returns:
            list: A list of **tuples (keystroke, word, frequency)**
        """

        with self._lock:
            self._cursor.execute(
                "SELECT keystroke, word, frequency FROM keystroke_map WHERE keystroke = ?",
                (keystroke,),
            )
            return self._cursor.fetchall()

    @lru_cache_with_doc(maxsize=128)
    def fuzzy_get_word(
        self, keystroke: str, max_distance: int
    ) -> list[tuple[str, str, int]]:
        """
        Get the **(keystroke, word, frequency)** tuple entry with the given keystroke \
        from the database with a Levenshtein distance less than or equal to the given distance.

        Args:
            keystroke (str): The keystroke to search for.
            max_distance (int): The maximum Levenshtein distance to search for.

        Returns:
            list: A list of **tuples (keystroke, word, frequency)**
        """
        with self._lock:
            self._cursor.execute(
                "SELECT keystroke, word, frequency FROM keystroke_map \
                WHERE levenshtein(keystroke, ?) <= ?",
                (keystroke, max_distance),
            )
            return self._cursor.fetchall()

    def fuzzy_get_exist(self, keystroke: str, distance: int) -> bool:
        """
        Check if there is any entry in the database with a Levenshtein distance \
        less than or equal to the given distance.

        Args:
            keystroke (str): The keystroke to search for.
            distance (int): The maximum Levenshtein distance to search for.

        Returns:
            bool: True if there is an entry with a Levenshtein distance less than \
            or equal to the given distance, False otherwise.
        """
        with self._lock:
            self._cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM keystroke_map \
                WHERE levenshtein(keystroke, ?) <= ?)",
                (keystroke, distance),
            )

            return bool(self._cursor.fetchone()[0])

    @lru_cache_with_doc(maxsize=128)
    def get_closest_word(
        self, keystroke: str, max_search_distance: int = 2
    ) -> list[tuple[str, str, int]]:
        """
        Get the **(keystroke, word, frequency)** tuple entry with smallest \
        Levenshtein distance to the given keystroke from the database.

        Args:
            keystroke (str): The keystroke to search for
            max_search_distance (int, optional): The maximum Levenshtein distance to search for. \
            Defaults to 1.

        Returns:
            list: A list of **tuples (keystroke, word, frequency)**
        """

        # Search for the direct match first
        if result := self.get_word(keystroke):
            return result

        closest_distance = self.get_closest_word_distance(keystroke)
        if closest_distance > max_search_distance:
            return []

        return self.fuzzy_get_word(keystroke, closest_distance)

    def create_keystroke_map_table(self):
        """
        Create the keystroke map table if it does not exist.

        The table has the following columns:
        - id: The primary key of the table.
        - keystroke: The keystroke of the word.
        - word: The word.
        - frequency: The frequency of the word.

        And an index(idx_keystroke) is on the keystroke column.
        """
        with self._lock:
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
            self._cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_keystroke ON keystroke_map (keystroke)"
            )
            self._conn.commit()

    def insert(self, keystroke: str, word: str, frequency: int):
        """
        Insert a new entry (keystroke, word, frequency) into the database.

        Args:
            keystroke (str): The keystroke to insert.
            word (str): The word to insert.
            frequency (int): The frequency of the word.
        """

        with self._lock:
            if (keystroke, word, frequency) not in self.get_word(keystroke):
                self._cursor.execute(
                    "INSERT INTO keystroke_map (keystroke, word, frequency) VALUES (?, ?, ?)",
                    (keystroke, word, frequency),
                )
                self._conn.commit()

    def __del__(self):
        self._conn.close()

    def keystroke_exists(self, keystroke: str) -> bool:
        """
        Check if a keystroke exists in the database.

        Args:
            keystroke (str): The keystroke to check for.

        Returns:
            bool: True if the keystroke exists in the database, False otherwise.
        """
        return bool(self.get_word(keystroke))

    @lru_cache_with_doc(maxsize=128)
    def get_closest_word_distance(self, keystroke: str) -> int:
        """
        Get the smallest Levenshtein distance between \
        the given keystroke and the words in the database.

        Args:
            keystroke (str): The keystroke to search for.

        Returns:
            int: The smallest Levenshtein distance between \
            the given keystroke and the words in the database.
        """
        with self._lock:
            self._cursor.execute(
                "SELECT MIN(levenshtein(keystroke, ?)) FROM keystroke_map",
                (keystroke,),
            )
            min_distance = self._cursor.fetchone()[0]
            assert (
                min_distance is not None
            ), "No entry found in the database, Min distance is None"
            return min_distance

    def word_to_keystroke(self, keystroke_results: str) -> list[str]:
        """
        Get the keystroke of a word in the database.

        Args:
            word (str): The word to get the keystroke of.

        Returns:
            list: A list of keystrokes that correspond to the word.
        """
        with self._lock:
            self._cursor.execute(
                "SELECT keystroke FROM keystroke_map WHERE word = ?",
                (keystroke_results,),
            )
            return self._cursor.fetchall()

    def word_exists(self, word: str) -> bool:
        """
        Check if a word exists in the database.

        Args:
            word (str): The word to check for.

        Returns:
            bool: True if the word exists in the database, False otherwise.
        """
        return bool(self.word_to_keystroke(word))

    def increment_word_frequency(self, word: str):
        """
        Increment the frequency of a word in the database.

        Args:
            word (str): The word to increment the frequency of.
        """

        with self._lock:
            self._cursor.execute(
                "UPDATE keystroke_map SET frequency = frequency + 1 WHERE word = ?",
                (word,),
            )
            self._conn.commit()

    def get_word_frequency(self, word: str) -> int:
        """
        Get the frequency of a word in the database.

        Args:
            word (str): The word to get the frequency of.

        Returns:
            int: The frequency of the word.
        """
        with self._lock:
            self._cursor.execute(
                "SELECT frequency FROM keystroke_map WHERE word = ?", (word,)
            )
            if frequency := self._cursor.fetchone():
                return frequency[0]
            else:
                return 0

    def update_word_frequency(self, keystroke: str, word: str, frequency: int):
        """
        Update the frequency of a word in the database. The keystroke and the word must match.

        Args:
            keystroke (str): The keystroke of the word.
            word (str): The word to update the frequency of.
            frequency (int): The new frequency of the word.
        """
        with self._lock:
            self._cursor.execute(
                "UPDATE keystroke_map SET frequency = ? WHERE keystroke = ? AND word = ?",
                (frequency, keystroke, word),
            )
            self._conn.commit()

    def to_csv(self, file_path: str):
        """
        Export the keystroke map to a CSV file.

        Args:
            file_path (str): The path to the CSV file to write to.
        """
        with self._lock:
            self._cursor.execute("SELECT keystroke, word, frequency FROM keystroke_map")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("keystroke,word,frequency\n")
                for row in self._cursor.fetchall():
                    f.write(f'"{row[0]}","{row[1]}",{int(row[2])}\n')


if __name__ == "__main__":
    # Example usage of the KeystrokeMappingDB class
    import pathlib

    path = pathlib.Path(__file__).parent / "src" / "english_keystroke_map.db"
    db = KeystrokeMappingDB(path)

    # print(db.get_closest_word("su32", 2))
    # print(db.get_closest_word_distance("u04counsel"))
    # print(db.word_to_keystroke("你"))

    total = [
        ("`", "`"),
        ("~", "~"),
        ("!", "!"),
        ("@", "@"),
        ("#", "#"),
        ("$", "$"),
        ("%", "%"),
        ("^", "^"),
        ("&", "&"),
        ("*", "*"),
        ("(", "("),
        (")", ")"),
        ("-", "-"),
        ("_", "_"),
        ("=", "="),
        ("+", "+"),
        ("[", "["),
        ("{", "{"),
        ("]", "]"),
        ("}", "}"),
        ("\\", "\\"),
        ("|", "|"),
        (";", ";"),
        (":", ":"),
        ("'", "'"),
        ('"', '"'),
        (",", ","),
        ("<", "<"),
        (".", "."),
        (">", ">"),
        ("/", "/"),
        ("?", "?"),
    ]

    for keystroke, words in total:
        for word in words:
            db.insert(keystroke, word, 1)
            print(f"Inserted {keystroke} -> {word}")
