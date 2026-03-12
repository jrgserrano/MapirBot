import os
import sys

# Add the project root to sys.path to allow imports from agent.llm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from agent.llm import llm_text
    from langchain_core.messages import HumanMessage
    print("✅ LLM configuration imported successfully.")
except ImportError as e:
    print(f"❌ Error importing llm from agent.llm: {e}")
    sys.exit(1)

def test_llm():
    print("\n--- Testing LLM Connectivity ---")
    question = "Hola, ¿puedes presentarte brevemente?"
    print(f"Pregunta: {question}")
    
    try:
        print("Esperando respuesta del modelo...")
        response = llm_text.invoke([HumanMessage(content=question)])
        print("\n--- Respuesta del Modelo ---")
        print(response.content)
        print("\n--- Fin de la prueba ---")
        print("✅ Prueba de conectividad exitosa.")
    except Exception as e:
        print(f"\n❌ Error al invocar el LLM: {e}")
        print("\nAsegúrate de que LM Studio esté corriendo y el modelo esté cargado en el puerto 1234.")

if __name__ == "__main__":
    test_llm()
