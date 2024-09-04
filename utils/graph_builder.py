import spacy
import networkx as nx
import os
import re
from collections import defaultdict

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Clean text by removing non-printable characters and extra whitespace.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with a single space
    return text.strip()

def extract_entities_and_relationships(text):
    """
    Extract entities and relationships from the text using SpaCy.
    """
    doc = nlp(text)
    entities = set()
    relationships = []

    # Normalize entity names (e.g., removing "The")
    def normalize_entity(entity):
        return entity.lower().strip()

    # Extract entities
    for ent in doc.ents:
        normalized_ent = normalize_entity(ent.text)
        entities.add(normalized_ent)

    # Extract relationships based on sentence-level co-occurrence
    for sent in doc.sents:
        sent_entities = [normalize_entity(ent.text) for ent in sent.ents]
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                if sent_entities[i] != sent_entities[j]:
                    relationships.append((sent_entities[i], sent_entities[j]))

    print(f"Extracted Entities: {entities}")  # Debug: Print entities
    print(f"Extracted Relationships: {relationships}")  # Debug: Print relationships

    return entities, relationships


def build_graph_from_text(text_file_path):
    """
    Build a graph from the text file, with nodes as entities and edges as relationships.
    """
    G = nx.Graph()

    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Clean the text
    text = clean_text(text)

    # Extract entities and relationships
    entities, relationships = extract_entities_and_relationships(text)

    # Add entities as nodes
    for entity in entities:
        G.add_node(entity)

    # Add relationships as edges
    for entity1, entity2 in relationships:
        G.add_edge(entity1, entity2)

    return G


def save_graph(graph, path='indexes/graph.gml'):
    """
    Save the graph to a file in GML format.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_gml(graph, path)

def load_graph(path='indexes/graph.gml'):
    """
    Load the graph from a GML file.
    """
    return nx.read_gml(path)

def process_extracted_text(text_file_path, graph_output_path):
    """
    Process the extracted text to build and save the graph.
    """
    graph = build_graph_from_text(text_file_path)
    save_graph(graph, graph_output_path)
    print(f"Graph built and saved to {graph_output_path}")

if __name__ == "__main__":
    text_file_path = "data/extracted/book2/text.txt"
    graph_output_path = "indexes/graph.gml"
    process_extracted_text(text_file_path, graph_output_path)
