"""Script para testar o agente LangGraph localmente"""
from LangGraph import export_graph

def main():
    print("🤖 Agente React iniciado! Faça uma pergunta (ou 'sair' para encerrar)\n")
    
    graph = export_graph
    
    while True:
        user_input = input("Você: ")
        
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("👋 Até logo!")
            break
            
        if not user_input.strip():
            continue
            
        print("\n🔍 Processando...\n")
        
        try:
            response = graph.invoke({"messages": [("user", user_input)]})
            
            # Pega a última mensagem (resposta do agente)
            ai_message = response["messages"][-1].content
            print(f"🤖 Agente: {ai_message}\n")
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"❌ Erro: {e}\n")
            print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
