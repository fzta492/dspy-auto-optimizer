import dspy
import os
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot

# Importações locais
from src.modules.cot_module import ChainOfThoughtQA
from src.data.dummy_data import train_data

# 1. Configuração de Ambiente
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Por favor, configure a OPENAI_API_KEY no arquivo .env")

# Configura o modelo (Teacher e Student podem ser o mesmo para este exemplo)
lm = dspy.LM('openai/gpt-3.5-turbo', api_key=api_key)
dspy.configure(lm=lm)

def main():
    print("Iniciando Pipeline de Otimização DSPy...")

    # 2. Instancia o Módulo (Não otimizado)
    module = ChainOfThoughtQA()

    # 3. Define o Otimizador (Teleprompter)
    # BootstrapFewShot cria exemplos sintéticos e filtra os melhores
    teleprompter = BootstrapFewShot(
        metric=dspy.evaluate.answer_exact_match,
        max_bootstrapped_demos=4,  # Quantos exemplos criar
        max_labeled_demos=4        # Quantos exemplos reais usar
    )

    # 4. Compilação (O momento mágico)
    print("⚙️ Compilando prompts (Isso pode levar alguns segundos)...")
    compiled_module = teleprompter.compile(module, trainset=train_data)

    # 5. Salva o artefato
    output_path = "optimized_signature.json"
    compiled_module.save(output_path)
    print(f"Otimização concluída! Artefato salvo em: {output_path}")

    # 6. Teste Rápido
    print("\n Testando o módulo otimizado:")
    test_q = "Quem pintou a Mona Lisa?"
    pred = compiled_module(test_q)
    
    print(f"Q: {test_q}")
    # Exibe o raciocínio se disponível
    if hasattr(pred, 'reasoning'):
        print(f"Reasoning: {pred.reasoning}")
    print(f"A: {pred.answer}")

if __name__ == "__main__":
    main()
