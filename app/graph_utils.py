from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network


def build_rules_graph(rules: pd.DataFrame, output_path: str | Path) -> Path:
    """Create an interactive HTML network graph from association rules."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = nx.DiGraph()

    for _, row in rules.iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])
        confidence = float(row.get("confidence", 0.0))
        lift = float(row.get("lift", 0.0))
        support = float(row.get("support", 0.0))

        for left in antecedents:
            graph.add_node(left, title=left)
            for right in consequents:
                graph.add_node(right, title=right)
                graph.add_edge(
                    left,
                    right,
                    value=max(lift, 0.1),
                    title=(
                        f"support={support:.3f}<br>"
                        f"confidence={confidence:.3f}<br>"
                        f"lift={lift:.3f}"
                    ),
                )

    net = Network(height="650px", width="100%", directed=True, notebook=False)
    net.from_nx(graph)
    net.repulsion(node_distance=180, central_gravity=0.25, spring_length=160)
    net.save_graph(str(output_path))
    return output_path
