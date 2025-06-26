import heapq
import re


class SentenceGraph:
    def __init__(self) -> None:
        self._graph = {}
        self._id_maps = {}
        self._heuristic = {}
        self._sentence_length = 0

    def _add_edge(
        self, u_id: str, v_id: str, distance: int, direct: bool = True
    ) -> None:
        if u_id not in self._graph:
            self._graph[u_id] = [(v_id, distance)]
        else:
            if (v_id, distance) not in self._graph[u_id]:
                self._graph[u_id].append((v_id, distance))

        if v_id not in self._graph:
            if direct:
                self._graph[v_id] = []
            else:
                self._graph[v_id] = [(u_id, distance)]

    def _find_shortest_paths(self, start_id: str, end_id: str) -> list[list[str]]:
        # Use A* algorithm with heuristic: number of characters to end node
        predecessor = {node_id: set() for node_id in self._graph}
        distance = {node_id: -1 for node_id in self._graph}
        distance[start_id] = 0

        priority_queue = [(0, 0, start_id)]
        while priority_queue:
            est_total, current_distance, current_id = heapq.heappop(priority_queue)

            if current_id == end_id:
                break

            for neighbor_id, neighbor_weight in self._graph[current_id]:
                new_distance = current_distance + neighbor_weight
                h = self._heuristic[neighbor_id] * 2
                # print(
                #     f"Current: {current_id}, Neighbor: {neighbor_id}, current_distance: {current_distance},est_total: {est_total}, heuristic: {h}, neg weigh: {neighbor_weight}, new_distance: {new_distance}"
                # )

                if distance[neighbor_id] < 0:  # Not visited yet
                    distance[neighbor_id] = new_distance
                    heapq.heappush(
                        priority_queue, (new_distance + h, new_distance, neighbor_id)
                    )
                    predecessor[neighbor_id] = set([current_id])
                else:
                    if new_distance < distance[neighbor_id]:
                        distance[neighbor_id] = new_distance
                        heapq.heappush(
                            priority_queue,
                            (new_distance + h, new_distance, neighbor_id),
                        )
                        predecessor[neighbor_id] = set([current_id])
                    elif new_distance == distance[neighbor_id]:
                        predecessor[neighbor_id].add(current_id)

        # Get the first shortest path
        path = []
        node = end_id
        while node != start_id:
            path.append(node)
            preds = list(predecessor[node])
            if not preds:
                break
            node = preds[0]
        path.append(start_id)
        return [path[::-1]]

    def add_token_path(self, tokens: list[tuple]) -> None:
        new_add_sentence_length = len("".join([token for token, _ in tokens]))
        if self._sentence_length == 0:
            self._sentence_length = new_add_sentence_length
        else:
            if self._sentence_length != new_add_sentence_length:
                raise ValueError(
                    f"The add sentence length({new_add_sentence_length}) does not match the Graph's current sentence length{self._sentence_length}."
                )

        prev_str = ""
        prev_token_id = "<start>"
        for token, distance in tokens:
            empty_token_id = f"<none>_{len(prev_str)}_{len(prev_str)}"
            token_id = f"{token}_{len(prev_str)}_{len(prev_str + token)}"
            self._id_maps[token_id] = token
            self._heuristic[empty_token_id] = self._sentence_length - len(prev_str)
            self._heuristic[token_id] = self._sentence_length - len(prev_str + token)
            self._add_edge(prev_token_id, empty_token_id, 0)
            self._add_edge(empty_token_id, token_id, distance)
            prev_str += token
            prev_token_id = token_id
        self._add_edge(prev_token_id, "<end>", 0)
        self._heuristic["<end>"] = 0

    def get_sentence(self) -> list[list[str]]:
        possible_paths = []
        shortest_paths = self._find_shortest_paths("<start>", "<end>")
        for path in shortest_paths:
            path = list(
                filter(
                    lambda x: x not in ["<start>", "<end>"]
                    and not x.startswith("<none>"),
                    path,
                )
            )
            possible_paths.append([self._id_maps[id] for id in path])

        return possible_paths


if __name__ == "__main__":
    graph = SentenceGraph()
    graph.add_token_path([("hello", 1), ("world", 2)])
    graph.add_token_path([("hello", 1), ("wor", 1), ("ld", 1)])
    print(graph._find_shortest_paths("<start>", "<end>"))
    print(graph.get_sentence())
    # Output: [['hello', 'world']]
