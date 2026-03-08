from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from models import models

# 1 SystemMessage
SYSTEM_MESSAGE_LLMS = SystemMessage(content="""
Você é um especialista em análise de código e boas 
práticas de programação.
Sua tarefa é analisar o código fornecido
e identificar e sugerir melhorias em termos de:
1. Performance e otimização
2. Boas práticas e padrões de código
3. Segurança, vulnerabilidades e tratamento de erros
4. Legibilidade e manutenibilidade

Forneça suas sugestões de forma clara, objetiva e estruturada,
com exemplos praticos de como implementar as melhorias sugeridas.
Seja específico e detalhado em suas análises, sugestões e recomendações.
"""
)

# 2 Definição do Estado
class State(TypedDict):
    query: str # Código a ser analisado
    llm1: str # Análise do Geminni
    llm2: str # Análise do o4 mini
    best_llm: str # Melhor análise escolhida

# 3  Nós
def call_llm_1(state: State):
    """Recebe o código e retorna a análise do modelo Gemini"""
    messages = [
        SystemMessage(content=SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content=f"Analise o seguinte código e forneça sugestões de melhorias::\n\n{state['query']}")
     ]

    response = models["gemini_2.5_flash"].invoke(messages)
    return {"llm1": response.content}

def call_llm_2(state: State):
    """Recebe o código e retorna a análise do modelo o4 mini"""
    messages = [
        SystemMessage(content=SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content=f"Analise o seguinte código e forneça sugestões de melhorias::\n\n{state['query']}")
     ]

    response = models["o4"].invoke(messages)
    return {"llm2": response.content}

def judge(state: State):
    """Avalie qual das análises é mais completa e útil"""
    msg = f"""
    Aja como revisor técnico sênior e avalie a quantidade e qualidade
    das análises de código fornecidas por dois especialistas.

    Sua tarefa é escolher a análise que:
    1. Identifica mais problemas potenciais
    2. Fornece sugestões mais práticas e implementáveis.
    3. Considera aspectos do código como performance, segurança, legibilidade e manutenibilidade e etc.
    4. Explica melhor o raciocínio por trás de suas sugestões.  


    [Código Analisado]
    {state['query'] }

    [Análise do Especialista A]
    {state['llm1'] }

    [Análise do Especialista B]
    {state['llm2'] }

    Forneça sua avaliação comparativa e conclua com seu veredito
    final usando exatamente um destes formatos:
    '[[A]]' se a análise A for melhor
    '[[B]]' se a análise B for melhor
    '[[C]]' em caso de empate
    """

    messages = [HumanMessage(content=msg)]
    response = models["gpt_4o"].invoke(messages)
    return {"best_llm": response.content}

# 4 Construindo o Workflow

code_analysis_builder = StateGraph(State)

# Adiciona os nós
code_analysis_builder.add_node("call_llm_1", call_llm_1)
code_analysis_builder.add_node("call_llm_2", call_llm_2)
code_analysis_builder.add_node("judge", judge)

# Adiciona as arestas
code_analysis_builder.add_edge(START, "call_llm_1")
code_analysis_builder.add_edge(START, "call_llm_2")
code_analysis_builder.add_edge("call_llm_1", "judge")
code_analysis_builder.add_edge("call_llm_2", "judge")
code_analysis_builder.add_edge("judge", END)

code_analysis_workflow = code_analysis_builder.compile()  