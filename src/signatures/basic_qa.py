import dspy

class BasicQA(dspy.Signature):
    """
    Responda a perguntas complexas com respostas factuais, curtas e diretas.
    O objetivo é ser preciso e evitar verbosidade desnecessária.
    """
    
    question = dspy.InputField(desc="A pergunta do usuário que necessita de raciocínio")
    answer = dspy.OutputField(desc="A resposta final otimizada, geralmente entre 1 a 10 palavras")
