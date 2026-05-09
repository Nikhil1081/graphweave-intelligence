import networkx as nx
import spacy
from pathlib import Path
import pickle

GRAPH_PATH = Path("graph.gpickle")

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def load_graph() -> nx.Graph:
    if GRAPH_PATH.exists():
        with GRAPH_PATH.open("rb") as f:
            return pickle.load(f)
    return nx.Graph()


def save_graph(G: nx.Graph) -> None:
    with GRAPH_PATH.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def ingest_text(doc_id: str, text: str) -> None:
    """
    Build a simple knowledge graph:
    - Nodes: entities (ORG, PERSON, PRODUCT, GPE...)
    - Edges: co-occurrence of entities within a sentence
    - Nodes store supporting sentences and doc_ids.
    """
    nlp = get_nlp()
    G = load_graph()

    doc = nlp(text)
    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in {"ORG", "PERSON", "GPE", "PRODUCT"}]
        if not ents:
            continue

        sent_text = sent.text.strip()

        # Add/update nodes
        for ent in ents:
            node_id = ent.text.strip()
            if node_id not in G:
                G.add_node(
                    node_id,
                    label=ent.label_,
                    sentences=set(),
                    docs=set(),
                )
            G.nodes[node_id]["sentences"].add(sent_text)
            G.nodes[node_id]["docs"].add(doc_id)

        # Connect co-occurring entities
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a = ents[i].text.strip()
                b = ents[j].text.strip()
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    # Convert sets to lists for serialization
    for node in G.nodes:
        G.nodes[node]["sentences"] = list(G.nodes[node]["sentences"])
        G.nodes[node]["docs"] = list(G.nodes[node]["docs"])

    save_graph(G)


def get_context_for_query(query: str, max_hops: int = 2, max_sentences: int = 20) -> str:
    """
    Use entities from the query to walk the graph and collect a focused context.
    If no entities are found in graph, fall back to top-degree nodes.
    """
    nlp = get_nlp()
    G = load_graph()

    if len(G) == 0:
        return ""

    doc = nlp(query)
    query_ents = [ent.text.strip() for ent in doc.ents if ent.text.strip() in G.nodes]

    if not query_ents:
        # Fallback: use top-degree nodes
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:3]
        query_ents = [n for n, _ in top_nodes]

    visited = set()
    sentences = []

    for ent in query_ents:
        if ent not in G:
            continue

        frontier = [(ent, 0)]
        while frontier:
            node, dist = frontier.pop(0)
            if node in visited or dist > max_hops:
                continue
            visited.add(node)

            node_data = G.nodes[node]
            sentences.extend(node_data.get("sentences", []))

            for neigh in G.neighbors(node):
                if neigh not in visited:
                    frontier.append((neigh, dist + 1))

            if len(sentences) >= max_sentences:
                break

        if len(sentences) >= max_sentences:
            break

    unique = list(dict.fromkeys(sentences))[:max_sentences]
    return "\n".join(unique)


def graph_stats():
    G = load_graph()
    if len(G) == 0:
        return 0, 0, 0.0, []

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0.0
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    return num_nodes, num_edges, avg_degree, top_nodes
