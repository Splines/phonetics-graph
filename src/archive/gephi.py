# Out of the graph.csv generate a file nodes.csv with id,label
# and a file edges.csv with sourceid, targetid, weight

import csv


def main():
    with open("./data/ipa/graph.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        nodes = set()
        edges = set()
        node_cache = {}
        node_id = 0
        for row in reader:
            source = row[1]
            target = row[3]
            weight = row[5]
            if source not in node_cache:
                node_cache[source] = node_id
                node_id += 1
            if target not in node_cache:
                node_cache[target] = node_id
                node_id += 1
            nodes.add(source)
            nodes.add(target)
            edges.add((node_cache[source], node_cache[target], weight))

    with open("./data/ipa/nodes.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for node in nodes:
            writer.writerow([node_cache[node], node])

    with open("./data/ipa/edges.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])
        for edge in edges:
            writer.writerow(edge)


if __name__ == "__main__":
    main()
