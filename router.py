from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from models import models

#System Messages
SYSTEM_MESSAGE_ASSISTENTE = SystemMessage( content="""sumary_line
Você é um assistente virtual especializado em ajudar
com diferentes tipos de consulta.
Seja educado e prestativo em suas respostas.
""")

SYSTEM_MESSAGE_TECNICO = SystemMessage( content="""
Você é um especialista técnico que fornece
respostas detalhadas e precisas sobre tecnologia.
Use linguagem técnica apropriada e forneça exemplos
práticos quando possível.
""")

SYSTEM_MESSAGE_SAUDE = SystemMessage( content="""
Você é um consultor de saúde que fornece informações gerais
sobre bem estar e saúde.
Lembre-se de enfatizar quue suas resposta são apenas
informativas e não substituem consultas médicas
""")

# Estados
class State(TypedDict):
    query: str
    category: str
    answer: str

def router(state: State):
    """Roteia a consulta para diferentes categorias baseado no conteúdo."""
    query = state["query"].lower()
    palavras_tecnologia = ["python", "programação", "código", "desenvolvimento","software","tecnologia"]
    palavras_saude = ["saúde","exercício","alimentação","bem-estar","medicina","dieta"]
    if any(palavra in query for palavra in palavras_tecnologia):
        return {"category":"tecnico"}
    elif any(palavra in query for palavra in palavras_saude):
        return {"category":"saude"}
    else:
        return {"category":"assistente"}

    def assistente (state: State):
        """Processa consultas gerais"""
        messages = [
            SYSTEM_MESSAGE_ASSISTENTE
            HumanMessage(content=state["query"])
        ],
        response = models["gpt_4o"].invoke(messages)
        return {"answer": response.content}
    
    def saude (state: State):
        """Processa consultas sobre saúde"""
        messages = [
            SYSTEM_MESSAGE_SAUDE,
            HumanMessage(content=state["query"])
        ],
        response = models["gpt_4o"].invoke(messages)
        return {"answer": response.content}

#contruindo o workflw
workflow_builder = StateGraph(State)
workflow_builder.add_node("router", router)
workflow_builder.add_node("assistente", assistente)
workflow_builder.add_node("tecnico", tecnico)
workflow_builder.add_node("saude", saude)

workflow_builder.set_entry_point("router")

workflow_builder.set_conditional_edges("router",
                                       lambda state: state["category"],{
                                           "assistente": "assistente",,
                                           "tecnico": "tecnico",
                                           "saude": "saude"
                                           })
workflow_builder.add_edge("assistente", END)
workflow_builder.add_edge("tecnico", END)
workflow_builder.add_edge("saude", END)

workflow_router = workflow_builder.compile()
