import dspy
from src.signatures.basic_qa import BasicQA

class ChainOfThoughtQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # O ChainOfThought adiciona automaticamente o passo de "Reasoning" (Raciocínio)
        # antes de gerar a resposta final, melhorando a lógica.
        self.prog = dspy.ChainOfThought(BasicQA)
    
    def forward(self, question):
        return self.prog(question=question)
