import heapq


class SentenceGraph:
    def __init__(self) -> None:
        self._graph = {}
        self._id_maps = {}

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
        # By Dijkstra
        # TODO: use A* algorithm for better performance
        predecessor = {id: set() for id in self._graph}
        distance = {id: -1 for id in self._graph}
        distance[start_id] = 0

        priority_queue = [(0, start_id)]
        while priority_queue:
            current_distance, current_id = heapq.heappop(priority_queue)

            for neighbor_id, neighbor_weight in self._graph[current_id]:
                neg_new_distance = current_distance + neighbor_weight

                if distance[neighbor_id] < 0:  # Not visited yet
                    distance[neighbor_id] = neg_new_distance
                    heapq.heappush(priority_queue, (neg_new_distance, neighbor_id))
                    predecessor[neighbor_id] = set([current_id])
                else:

                    if neg_new_distance < distance[neighbor_id]:
                        distance[neighbor_id] = neg_new_distance
                        heapq.heappush(priority_queue, (neg_new_distance, neighbor_id))
                        predecessor[neighbor_id] = set([current_id])
                    elif neg_new_distance == distance[neighbor_id]:
                        predecessor[neighbor_id].add(current_id)

        # Get the path
        def dfs(current_id: str) -> list[list[str]]:
            if current_id == start_id:
                return [[start_id]]

            if predecessor[current_id] is None:
                return []

            paths = []
            for pred in predecessor[current_id]:
                paths.extend([path + [current_id] for path in dfs(pred)])
            return paths

        return dfs(end_id)

    def add_token_path(self, tokens: list[tuple]) -> None:
        prev_str = ""
        prev_token_id = "<start>"
        for token, distance in tokens:
            empty_token_id = f"<none>_{len(prev_str)}_{len(prev_str)}"
            token_id = f"{token}_{len(prev_str)}_{len(prev_str + token)}"
            self._id_maps[token_id] = token
            self._add_edge(prev_token_id, empty_token_id, 0)
            self._add_edge(empty_token_id, token_id, distance)
            prev_str += token
            prev_token_id = token_id
        self._add_edge(prev_token_id, "<end>", 0)

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
        possible_paths = sorted(possible_paths, key=len, reverse=False)

        return possible_paths


if __name__ == "__main__":
    graph = SentenceGraph()
    graph.add_token_path([("hello", 1), ("world", 2)])
    graph.add_token_path([("hello", 1), ("everyone", 2)])
    print(graph._find_shortest_paths("<start>", "<end>"))
    print(graph.get_sentence())
    # Output: [['hello', 'world']]
