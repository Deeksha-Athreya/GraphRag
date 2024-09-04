import spacy
import networkx as nx
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load SpaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Load BERT model and tokenizer for question answering
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def query_graph(graph, query):
    """
    Query the graph to retrieve relevant nodes and relationships.
    """
    # Perform entity extraction on the query
    doc = nlp(query)
    query_entities = {ent.text.lower() for ent in doc.ents}

    # Score nodes based on their relevance to the query
    relevant_nodes = set()
    for node in graph.nodes:
        if any(ent.lower() in node.lower() for ent in query_entities):
            relevant_nodes.add(node)

    # Debug: Check all edges for connections
    relevant_edges = []
    for u, v in graph.edges():
        if u in relevant_nodes and v in relevant_nodes:
            relevant_edges.append((u, v))

    print(f"Relevant nodes: {relevant_nodes}")  # Debug statement
    print(f"Relevant edges: {relevant_edges}")  # Debug statement

    # Extract subgraph with relevant nodes and edges
    subgraph = graph.edge_subgraph(relevant_edges).copy()

    return subgraph

def generate_response(query, subgraph):
    """
    Generate a response based on the query and relevant graph data using BERT.
    """
    if not subgraph.nodes():
        return "No relevant information found in the graph."

    # Convert subgraph to text
    nodes_text = "\n".join(subgraph.nodes)
    edges_text = "\n".join([f"{u} - {v}" for u, v in subgraph.edges])
    subgraph_text = f"Nodes:\n{nodes_text}\nEdges:\n{edges_text}"

    # Tokenize and encode the text for question answering
    inputs = tokenizer.encode_plus(query, subgraph_text, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get the answer from the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

    return answer

def handle_query(query, graph_path='indexes/graph.gml'):
    """
    Handle a user query by retrieving relevant information from the graph and generating a response.
    """
    # Load the graph
    graph = nx.read_gml(graph_path)
    print(f"Graph nodes: {graph.nodes()}")  # Debug statement
    print(f"Graph edges: {graph.edges()}")  # Debug statement
    
    # Query the graph
    subgraph = query_graph(graph, query)
    
    # Generate a response
    response = generate_response(query, subgraph)
    
    return response

if __name__ == "__main__":
    # Example query
    query = "What is Toyota Prius?"
    response = handle_query(query)
    print(response)
