import dspy

def validate_context_and_answer(example, pred, trace=None):
    """
    Uma métrica composta que verifica duas coisas:
    1. Se a resposta está correta (Exact Match).
    2. Se a resposta não é excessivamente longa (brevidade).
    
    Retorna:
        True se passar em ambos os critérios.
    """
    
    # 1. Verifica a exatidão (lógica padrão do DSPy)
    answer_match = dspy.evaluate.answer_exact_match(example, pred)
    
    # 2. Verifica a restrição de negócio (máximo 15 palavras)
    # Isso força o LLM a ser conciso durante a otimização.
    answer_length = len(pred.answer.split())
    length_check = answer_length <= 15
    
    if trace is None:
        # Durante a execução normal
        return answer_match and length_check
    else:
        # Se estivermos debugando, podemos retornar o score (float)
        return answer_match and length_check
