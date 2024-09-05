import networkx as nx

def load_graph(path='indexes/graph.gml'):
    """
    Load the graph from a GML file.
    """
    try:
        return nx.read_gml(path)
    except Exception as e:
        print(f"An error occurred while loading the graph: {e}")
        return None

def find_relevant_info(graph, keywords):
    """
    Find relevant nodes and edges in the graph based on keywords.
    """
    relevant_nodes = set()
    relevant_edges = {}
    
    for keyword in keywords:
        if keyword in graph.nodes:
            relevant_nodes.add(keyword)
            # Find edges connected to this node
            for neighbor in graph.neighbors(keyword):
                edge = (keyword, neighbor) if (keyword, neighbor) in graph.edges else (neighbor, keyword)
                if edge not in relevant_edges:
                    relevant_edges[edge] = graph[edge[0]][edge[1]].get('weight', 1)  # Default weight to 1 if not present
    
    print(f"Keywords: {keywords}")  # Debug statement
    print(f"Relevant nodes found: {relevant_nodes}")  # Debug statement
    print(f"Relevant edges found: {relevant_edges}")  # Debug statement
    
    return relevant_nodes, relevant_edges

def query_graph_from_keywords(keywords, graph_path='indexes/graph.gml'):
    """
    Query the graph for relevant information based on extracted keywords.
    """
    graph = load_graph(graph_path)
    if graph is not None:
        relevant_nodes, relevant_edges = find_relevant_info(graph, keywords)
        return relevant_nodes, relevant_edges
    else:
        print("Graph not loaded.")
        return set(), {}
