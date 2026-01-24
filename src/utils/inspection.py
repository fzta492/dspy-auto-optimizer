import dspy

def print_last_prompt():
    """
    Imprime o último prompt enviado ao LLM e a resposta recebida.
    Útil para debugging e para mostrar a 'Alquimia' a transformar-se em Engenharia.
    """
    try:
        print("\n" + "="*40)
        print("🔍 INSPEÇÃO DO ÚLTIMO PROMPT (History)")
        print("="*40)
        lm = dspy.settings.lm
        if lm and len(lm.history) > 0:
            last_call = lm.history[-1]
            print(f"📥 INPUT (Prompt Otimizado):\n\n{last_call['prompt']}\n")
            print("-" * 20)
            print(f"📤 OUTPUT (Resposta do Modelo):\n\n{last_call['response']['choices'][0]['message']['content']}\n")
            print("="*40 + "\n")
        else:
            print("⚠️ Nenhum histórico encontrado. Execute uma predição primeiro.")
    except Exception as e:
        print(f"Erro ao inspecionar histórico: {e}")
