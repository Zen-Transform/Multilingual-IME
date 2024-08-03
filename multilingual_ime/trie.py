from .candidate import CandidateWord


class TrieNode():
    def __init__(self):
        self.children = {}
        self.value = None

class Trie():
    def __init__(self, keystroke_mapping_dict: dict = None):
        self.root = TrieNode()
        self.keyStrokeCatch = {}

        if keystroke_mapping_dict is not None:
            for key, value in keystroke_mapping_dict.items():
                self.insert(key, value)

    def insert(self, key:str, value:list)->None:
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if node.value is None:
            node.value = [  # fix: bad code hard to read
                CandidateWord(word=element[0], keystrokes=key, word_frequency=element[1]) for element in value
                ]
        else:
            node.value.extend([
                CandidateWord(word=element[0], keystrokes=key, word_frequency=element[1]) for element in value
                ])

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
                    # substitutions = previous_row[j] + (char1 != char2)
                    if char1 != char2:
                        if i > 0 and j > 0 and s1[i-1] == char2 and s1[i] == char1:
                            substitutions = previous_row[j-1]
                        else:
                            substitutions = previous_row[j] + 1
                    else:
                        substitutions = previous_row[j]

                    current_row.append(min(insertions, deletions, substitutions))

                previous_row = current_row

            return previous_row[-1]

        traverse(self.root, "")
        minHeap.sort(key=lambda x: x[0])

        result = [{"distance": res[0], "keySoFar": res[1], "value": res[2]} for res in minHeap[:num_of_result]]
        self.keyStrokeCatch[query] = result

        return result
    