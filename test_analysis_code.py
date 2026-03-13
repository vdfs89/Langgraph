def main():
    try:
        from parallelization import code_analysis_workflow
    except ModuleNotFoundError as exc:
        print(
            "Dependência ausente para executar a análise. "
            f"Instale o pacote: {exc.name}"
        )
        return

    # Código de exemplo para análise
    codigo_teste = """
de calcular_mrfis(lista):
        soma = 0
        for i in range(len(lista)):
            soma = soma + lista[i]
        media = soma / len(lista)
        return media

# Testanndo a função
numeros = [1,2,3,4,5]
resultado = calcular_mrfis(numeros)
print(f'A mmédia é:{resultado}')
"""

    try:
        # Executando o workflow
        resultado = code_analysis_workflow.invoke({
            "query": codigo_teste
        })

        # Exibindo o resultado
        print("\n=== Análise do Gemini ===")
        print(resultado['llm1'])
        print("\n=== Análise do o4 mini ===")
        print(resultado['llm2'])
        print("\n=== Avaliação Final ===")
        print(resultado['best_llm'])
    except Exception as exc:
        print(f"Falha ao executar a análise: {exc}")


if __name__ == "__main__":
    main()


def test_main_existe():
    assert callable(main)
