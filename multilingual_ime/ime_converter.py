import json

class TrieNode():
    ...

class Trie():
    ...

class IMEConverter():
    ...

class EnglishIMEConverter():
    ...


class TrieNode():
    def __init__(self):
        self.children = {}
        self.value = None

class Trie():
    def __init__(self, data_dict: dict = None):
        self.root = TrieNode()
        self.keyStrokeCatch = {}
        self.startSet = set()
        self.endSet = set()

        if data_dict is not None:
            for key, value in data_dict.items():
                self.insert(key, value)

    def insert(self, key:str, value:list)->None:
        if key[0] not in self.startSet:
            self.startSet.add(key[0])
        if key[-1] not in self.endSet:
            self.endSet.add(key[-1])

        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if node.value is None:
            node.value = [{"word": element[0], "frequency": element[1]} for element in value]
        else:
            node.value.extend([{"word": element[0], "frequency": element[1]} for element in value])

    def search(self, key:str) -> list:
        node = self.root
        for char in key:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.value

    def findClosestMatches(self, query: str, num_of_result: int) -> list:
        if query in self.keyStrokeCatch:
            return self.keyStrokeCatch[query]

        minHeap = []

        def dfs(node: TrieNode, keySoFar: str) -> None:
            if node.value is not None:
                distance = levenshteinDistance(query, keySoFar)
                minHeap.append((distance, keySoFar, node.value))

        def traverse(node: TrieNode, keySoFar: str) -> None:
            dfs(node, keySoFar)
            for char, child_node in node.children.items():
                traverse(child_node, keySoFar + char)

        def levenshteinDistance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshteinDistance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))

            for i, char1 in enumerate(s1):
                current_row = [i + 1]

                for j, char2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (char1 != char2)

                    current_row.append(min(insertions, deletions, substitutions))

                previous_row = current_row

            return previous_row[-1]

        traverse(self.root, "")
        minHeap.sort(key=lambda x: x[0])

        result = [{"distance": res[0], "keySoFar": res[1], "value": res[2]} for res in minHeap[:num_of_result]]
        self.keyStrokeCatch[query] = result

        return result


class IMEConverter():
    def __init__(self, data_dict_path: str):
        self.trie = Trie(data_dict=json.load(open(data_dict_path, "r", encoding="utf-8")))

    def get_candidates(self, key_stroke_query: str, num_of_result: int = 3, distance: int = 3) -> list:
        candidates = self.trie.findClosestMatches(key_stroke_query, num_of_result)
        candidates = [candidate for candidate in candidates if candidate["distance"] <= distance]
        candidates = sorted(candidates, key=lambda x: x["distance"])

        word_candidates = []
        for candidate in candidates:
            for value in candidate["value"]:
                word_candidates.append({
                    "word": value["word"],
                    "distance": candidate["distance"],
                    "frequency": value["frequency"],
                    "key": candidate["keySoFar"],
                    "user_key": key_stroke_query
                })
        return word_candidates

class EnglishIMEConverter():
    def __init__(self, data_dict_path: str):
        self.trie = Trie(data_dict=json.load(open(data_dict_path, "r", encoding="utf-8")))

    def get_candidates(self, key_stroke_query: str, num_of_result: int = 3, distance: int = 3) -> list:
        is_upper = False
        if key_stroke_query[0].isupper():
            key_stroke_query_lower_case = key_stroke_query.lower()
            is_upper = True
        else:
            key_stroke_query_lower_case = key_stroke_query

        candidates = self.trie.findClosestMatches(key_stroke_query_lower_case, num_of_result)
        candidates = [candidate for candidate in candidates if candidate["distance"] <= distance]
        candidates = sorted(candidates, key=lambda x: x["distance"])

        word_candidates = []
        for candidate in candidates:
            for value in candidate["value"]:
                word_candidates.append({
                    "word": value["word"].capitalize() if is_upper else value["word"],
                    "distance": candidate["distance"],
                    "frequency": value["frequency"],
                    "key": candidate["keySoFar"],
                    "user_key": key_stroke_query
                })
        return word_candidates


if __name__ == "__main__":
    my_bopomofo_IMEConverter = IMEConverter(".\\src\\keystroke_mapping_dictionary\\bopomofo_dict_with_frequency.json")
    my_cangjie_IMEConverter = IMEConverter(".\\src\\keystroke_mapping_dictionary\\cangjie_dict_with_frequency.json")
    my_pinyin_IMEConverter = IMEConverter(".\\src\\keystroke_mapping_dictionary\\pinyin_dict_with_frequency.json")
    my_english_IMEConverter = EnglishIMEConverter(".\\src\\keystroke_mapping_dictionary\\english_dict_with_frequency.json")


    while True:
        input_text = input('Enter text: ')
        print("Bopomofo: ", my_bopomofo_IMEConverter.get_candidates(input_text, 3, 2))
        print("Cangjie: ", my_cangjie_IMEConverter.get_candidates(input_text, 3, 2))
        print("Pinyin: ", my_pinyin_IMEConverter.get_candidates(input_text, 3, 2))
        print("English: ", my_english_IMEConverter.get_candidates(input_text, 3, 2))
        print("\n")

    # my_converter = IMEConverter(".\\keystroke_mapping_dictionary\\bopomofo_dict_with_frequency.json")
    # input = "su3"

    # def levenshteinDistance(s1: str, s2: str) -> int:
    #     if len(s1) < len(s2):
    #         return levenshteinDistance(s2, s1)

    #     if len(s2) == 0:
    #         return len(s1)

    #     previous_row = list(range(len(s2) + 1))

    #     for i, char1 in enumerate(s1):
    #         current_row = [i + 1]

    #         for j, char2 in enumerate(s2):
    #             insertions = previous_row[j + 1] + 1
    #             deletions = current_row[j] + 1
    #             # substitutions = previous_row[j] + (char1 != char2)
    #             if char1 != char2:
    #                 if i > 0 and j > 0 and s1[i-1] == char2 and s1[i] == char1:
    #                     substitutions = previous_row[j-1]
    #                 else:
    #                     substitutions = previous_row[j] + 1
    #             else:
    #                 substitutions = previous_row[j]

    #             current_row.append(min(insertions, deletions, substitutions))

    #         previous_row = current_row

    #     return previous_row[-1]
    # print(levenshteinDistance("su3", "s3u"))