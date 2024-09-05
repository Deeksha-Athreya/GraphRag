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

def extract_definitions(text):
    """
    Extract definitions from the text based on certain linguistic patterns.
    This uses SpaCy dependency parsing to find patterns like 'X is defined as ...'.
    """
    doc = nlp(text)
    definitions = []

    # Common definition patterns
    definition_keywords = ['defined as', 'refers to', 'is a type of', 'is known as']

    # Check each sentence for patterns
    for sent in doc.sents:
        sentence_text = sent.text.lower()
        if any(keyword in sentence_text for keyword in definition_keywords):
            definitions.append(sent.text)

    print(f"Extracted Definitions: {definitions}")  # Debug: Print extracted definitions
    return definitions

def extract_entities_and_relationships(text):
    """
    Extract entities and relationships from the text using SpaCy.
    """
    doc = nlp(text)
    entities = set()
    relationships = defaultdict(int)

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
                    relationships[(sent_entities[i], sent_entities[j])] += 1

    print(f"Extracted Entities: {entities}")  # Debug: Print entities
    print(f"Extracted Relationships: {relationships}")  # Debug: Print relationships

    return entities, relationships

def build_graph_from_text(text_file_path):
    """
    Build a graph from the text file, with nodes as entities and edges as relationships.
    Also extract and add definitions as node attributes.
    """
    G = nx.Graph()

    try:
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Clean the text
        text = clean_text(text)

        # Extract definitions from the text
        definitions = extract_definitions(text)

        # Extract entities and relationships
        entities, relationships = extract_entities_and_relationships(text)

        # Add entities as nodes with labels and definitions as attributes
        for entity in entities:
            entity_label = entity  # Assuming the entity itself is the label
            entity_definition = next((defn for defn in definitions if entity in defn), "No definition found")
            G.add_node(entity_label, label=entity_label, definition=entity_definition)

        # Add relationships as edges with weights
        for (entity1, entity2), weight in relationships.items():
            G.add_edge(entity1, entity2, weight=weight)

    except Exception as e:
        print(f"An error occurred while building the graph: {e}")

    return G


def save_graph(graph, path='indexes/graph.gml'):
    """
    Save the graph to a file in GML format.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        nx.write_gml(graph, path)
        print(f"Graph saved to {path}")
    except Exception as e:
        print(f"An error occurred while saving the graph: {e}")

def load_graph(path='indexes/graph.gml'):
    """
    Load the graph from a GML file.
    """
    try:
        return nx.read_gml(path)
    except Exception as e:
        print(f"An error occurred while loading the graph: {e}")
        return None

def process_extracted_text(text_file_path, graph_output_path):
    """
    Process the extracted text to build and save the graph.
    """
    graph = build_graph_from_text(text_file_path)
    if graph is not None:
        save_graph(graph, graph_output_path)

if __name__ == "__main__":
    text_file_path = "data/extracted/text.txt"
    graph_output_path = "indexes/graph.gml"
    process_extracted_text(text_file_path, graph_output_path)
