import heapq

class SentenceGraph:
    def __init__(self) -> None:
        self._graph = {}

    @property
    def num_of_node(self) -> int:
        return len(self._graph)

    def add_edge(
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

    def find_shortest_paths(self, start_id: str, end_id: str) -> list[str]:
        # By Dijkstra
        # TODO: use A* algorithm for better performance
        predecessor = {id: None for id in self._graph}
        distance = {id: None for id in self._graph}
        distance[start_id] = 0

        priority_queue = [(0, start_id)]
        while priority_queue:
            current_distance, current_id = heapq.heappop(priority_queue)

            for neighbor_id, neighbor_weight in self._graph[current_id]:
                neg_new_distance = current_distance + neighbor_weight

                if distance[neighbor_id] is None:
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
        def get_path(predecessor: dict[str, set], end_id: str) -> list[list[str]]:

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

        return get_path(predecessor, end_id)
