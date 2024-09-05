from transformers import BartTokenizer, BartForConditionalGeneration
from query_handler import query_graph_from_keywords
import spacy
import re

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(query):
    """
    Extract keywords (nouns, proper nouns, named entities) from the user query.
    """
    doc = nlp(query)
    keywords = set()

    # Extracting nouns and proper nouns
    for chunk in doc.noun_chunks:
        keywords.add(chunk.root.text.lower())

    # Adding Named Entities
    for entity in doc.ents:
        keywords.add(entity.text.lower())

    return list(keywords)

# Load the BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

def summarize_text(text):
    """
    Summarize the given text using BART model.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_query(query, graph_path='indexes/graph.gml'):
    """
    Process the query to extract relevant information from the graph and summarize it.
    """
    # Extract keywords from the query
    keywords = extract_keywords(query)
    
    # Query the graph for relevant information
    relevant_nodes, relevant_edges = query_graph_from_keywords(keywords, graph_path)
    
    # Format the relevant graph information
    relevant_info = "Relevant information about the query:\n"
    for node in relevant_nodes:
        relevant_info += f"- Node: {node}\n"
    for edge in relevant_edges:
        node1, node2 = edge
        weight = relevant_edges[edge]
        relevant_info += f"- Edge between {node1} and {node2} with weight {weight}\n"
    
    # Add more descriptive information if available
    if not relevant_nodes and not relevant_edges:
        relevant_info += "No relevant information found in the graph."

    # Summarize the relevant graph information
    summary = summarize_text(relevant_info)
    return summary

if __name__ == "__main__":
    query = "Explain how the starter motor functions in a vehicle"
    graph_path = 'indexes/graph.gml'
    summary = process_query(query, graph_path)
    print(f"Query: {query}")
    print(f"Summary: {summary}")
