import heapq
import json

from .core.custom_decorators import lru_cache_with_doc
from .core.F import modified_levenshtein_distance


class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = None


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.keyStrokeCatch = {}

    def insert(self, key: str, input_value: any) -> None:
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if node.value is None:
            node.value = [input_value]
        else:
            node.value.extend([input_value])

    def search(self, query_key: str) -> list:
        node = self.root
        for char in query_key:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.value

    def _dfs_traverse(self, node: TrieNode, query: str, keySoFar: str) -> list:
        min_heap = []
        if node.children == {}:
            current_distance = modified_levenshtein_distance(query, keySoFar)
            heapq.heappush(min_heap, (current_distance, keySoFar))
            return min_heap
        else:
            for char, child_node in node.children.items():
                min_heap = list(
                    heapq.merge(
                        min_heap, self._dfs_traverse(child_node, query, keySoFar + char)
                    )
                )
                if len(min_heap) > 5:
                    min_heap = min_heap[:5]
                if len(min_heap) > 0 and min_heap[0][0] == 0:  # found exact match
                    return min_heap
            if node.value is not None:
                current_distance = modified_levenshtein_distance(query, keySoFar)
                heapq.heappush(min_heap, (current_distance, keySoFar))
            return min_heap

    @lru_cache_with_doc(maxsize=128, typed=False)
    def find_closest_match(self, query: str) -> list[dict]:
        """
        Find the closest match to the query string in the trie.
        Args:
            query (str): The query string.

        Returns:
            [dict]: A list of dictionaries containing the distance, keySoFar, and value of the closest match to the query string.

        """
        quick_search = self.search(query)
        if quick_search is not None:
            return [
                {
                    "distance": 0,
                    "keySoFar": query,
                    "value": quick_search,
                }
            ]

        minHeap = self._dfs_traverse(self.root, query, "")
        min_distance_candidate = minHeap[0]
        return [
            {
                "distance": candidate[0],
                "keySoFar": candidate[1],
                "value": self.search(candidate[1]),
            }
            for candidate in minHeap
            if candidate[0] <= min_distance_candidate[0]
        ]


from .candidate import CandidateWord

if __name__ == "__main__":
    data_dict_path = ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\english_dict_with_frequency.json"
    keystroke_mapping_dict = json.load(open(data_dict_path, "r", encoding="utf-8"))
    trie = Trie()
    if keystroke_mapping_dict is not None:
        for key, value in keystroke_mapping_dict.items():
            Candidate_words = [
                CandidateWord(
                    word=element[0], keystrokes=key, word_frequency=element[1]
                )
                for element in value
            ]
            for candidate in Candidate_words:
                trie.insert(key, candidate)
    result = trie.find_closest_match("apple")
    print(result)
