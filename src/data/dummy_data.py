import dspy

# Pequeno conjunto de treino (Golden Set)
# O DSPy usará isso para aprender o padrão de raciocínio.
train_data = [
    dspy.Example(question="Qual é a capital da França?", answer="Paris").with_inputs('question'),
    dspy.Example(question="Quem escreveu Dom Casmurro?", answer="Machado de Assis").with_inputs('question'),
    dspy.Example(question="Quanto é 10 mais 5 vezes 2?", answer="20").with_inputs('question'),
    dspy.Example(question="Qual o elemento químico com símbolo O?", answer="Oxigênio").with_inputs('question'),
    dspy.Example(question="Em que ano o homem pisou na lua?", answer="1969").with_inputs('question'),
]

# Conjunto de desenvolvimento para validar a métrica
dev_data = [
    dspy.Example(question="Qual a capital do Brasil?", answer="Brasília").with_inputs('question'),
    dspy.Example(question="Qual a cor do cavalo branco de Napoleão?", answer="Branco").with_inputs('question'),
]
