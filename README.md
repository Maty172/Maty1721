# Maty1721
novela 1
pip install transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo GPT-2 y el tokenizador
model_name = "gpt2"  # Puedes usar también "EleutherAI/gpt-neo-1.3B" para GPT-Neo
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Definir el prompt inicial (sinopsis o introducción de la novela)
prompt = """
Título: La Conexión de Tristan
Género: Ciencia ficción e historia
Sinopsis: En la remota isla de Tristan da Cunha, un joven escritor llamado Alex descubre un secreto escondido en la historia de la isla...
"""

# Tokenizar el prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generar el texto
outputs = model.generate(inputs['input_ids'], max_length=1000, num_return_sequences=1, temperature=0.7, top_p=0.9)

# Decodificar el texto generado
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
