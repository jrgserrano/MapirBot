import os
import asyncio
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

async def get_graph_data():
    uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    user = os.environ.get('NEO4J_USER', 'neo4j')
    password = os.environ.get('NEO4J_PASSWORD', 'neo4j_changed')

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    nodes = []
    edges = []

    async with driver.session() as session:
        # Query all nodes
        result = await session.run("MATCH (n) RETURN id(n) as id, labels(n) as labels, n.name as name, n.content as content")
        async for record in result:
            node_id = record["id"]
            labels = record["labels"]
            name = record["name"] or record["content"] or f"Node_{node_id}"
            nodes.append({
                "id": node_id,
                "label": labels[0] if labels else "Unknown",
                "name": str(name)
            })

        # Query all relationships
        result = await session.run("MATCH (n)-[r]->(m) RETURN id(n) as start_id, id(m) as end_id, type(r) as type, r.relation as relation")
        async for record in result:
            edges.append({
                "start": record["start_id"],
                "end": record["end_id"],
                "type": record["type"],
                "relation": record["relation"] or record["type"]
            })

    await driver.close()
    return nodes, edges

def generate_mermaid(nodes, edges):
    mermaid = "graph TD\n"
    for node in nodes:
        node_id = node["id"]
        label = node["label"]
        name = node["name"].replace('"', "'")[:50]
        if label == "Entity":
            mermaid += f'    n{node_id}["({label}) {name}"]\n'
            mermaid += f'    style n{node_id} fill:#f9f,stroke:#333,stroke-width:2px\n'
        elif label == "Episodic":
            mermaid += f'    n{node_id}["({label}) {name}"]\n'
            mermaid += f'    style n{node_id} fill:#bbf,stroke:#333,stroke-width:1px\n'
        else:
            mermaid += f'    n{node_id}["({label}) {name}"]\n'

    for edge in edges:
        mermaid += f'    n{edge["start"]} -- "{edge["relation"]}" --> n{edge["end"]}\n'
    return mermaid

def generate_png(nodes, edges, output_path="graph_visualization.png"):
    G = nx.MultiDiGraph()
    
    color_map = []
    labels = {}
    
    for node in nodes:
        G.add_node(node["id"])
        labels[node["id"]] = f"{node['name']}\n({node['label']})"
        if node["label"] == "Entity":
            color_map.append('lightpink')
        elif node["label"] == "Episodic":
            color_map.append('lightblue')
        else:
            color_map.append('lightgray')

    for edge in edges:
        G.add_edge(edge["start"], edge["end"], label=edge["relation"])

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    nx.draw(G, pos, with_labels=True, labels=labels, 
            node_color=color_map, node_size=3000, 
            font_size=10, font_weight='bold', 
            arrows=True, connectionstyle='arc3, rad = 0.1')
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Graphiti Knowledge Graph Visualization")
    plt.savefig(output_path, format="PNG", bbox_inches='tight')
    plt.close()
    return output_path

async def main():
    print("[INFO] Fetching graph data from Neo4j...")
    try:
        nodes, edges = await get_graph_data()
        
        # Generate PNG only
        print("[INFO] Generating PNG image...")
        png_path = generate_png(nodes, edges)
        
        print(f"[SUCCESS] PNG saved to {png_path}")
        
        # Abrir auto la imagen si en Mac
        import sys
        if sys.platform == "darwin":
            os.system(f"open {png_path}")
            print(f"[INFO] Opening {png_path}...")
            
    except Exception as e:
        print(f"[ERROR] Failed to generate visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
