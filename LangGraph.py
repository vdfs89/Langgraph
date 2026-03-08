from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage 
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
import os

# 1 - configurações iniciais
load_dotenv(override=True)
API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=API_KEY
)

# 2 - Prompt do sistema
system_message = SystemMessage(content="""
Você é um pesquisador muito sarcástico e ironico,
Use ferramenta 'search' sempre que necessário, especialmente
para perguntas que exigem informações atualizadas~da web ou específicas.
""")

# 3 - Criando a ferramenta search
@tool("search")
def search_web(query: str = "") -> str:
    """
    Busca informações na web baseada na consulta fornecida.

    Args:
    query: Termos para buscar dados na web

    Returns:
        As informações relevantes encontradas na web ou uma mensagem indicando
        que nenhum resultado foi encontrado
    """


    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query)
    return search_docs

# 4 - Criando o agente REACT    
tools = [search_web]
graph = create_react_agent(
    model,
    tools=tools,
    prompt=system_message
)

export_graph = graph

# 5 - Testando o agente
if __name__ == "__main__":
    print("🤖 Agente React iniciado! Faça uma pergunta (ou 'sair' para encerrar)\n")

    while True:
        user_input = input("Você: ")

        if user_input.lower() in ["sair", "exit", "quit"]:
            print("👋 Até logo!")
            break

        print("\n🔍 Processando...\n")

        response = graph.invoke({"messages": [("user", user_input)]})

        # Pega a última mensagem (resposta do agente)
        ai_message = response["messages"][-1].content
        print(f"🤖 Agente: {ai_message}\n")
        print("-" * 80 + "\n")