from collections import defaultdict
import math

# Datos de entrenamiento
data = [
    ("I loved the movie", "+"),
    ("I hated the movie", "-"),
    ("A great movie. Good movie.", "+"),
    ("Poor acting", "-"),
    ("Great acting. A good movie.", "+")
]

# Ajustes
def set_word(text):
    return text.lower().replace(".", "").split()

# Contadores
word_counts = {"+": defaultdict(int), "-": defaultdict(int)}
class_counts = {"+": 0, "-": 0}

# Contar palabras por clase
for text, label in data:
    class_counts[label] += 1
    for word in set_word(text):
        word_counts[label][word] += 1

# Vocabulario total
vocab = set()
for label in word_counts:
    vocab.update(word_counts[label].keys())
word_values = len(vocab)

# Totales por clase
total_words = {label: sum(word_counts[label].values()) for label in word_counts}

# Cálculo de P(palabra | clase)
def cond_prob(word, label):
    return (word_counts[label][word] + 1) / (total_words[label] + word_values)

# Predicción
def predict(text):
    tokens = set_word(text)
    probs = {}
    total_docs = sum(class_counts.values())

    for label in ["+", "-"]:
        # log(P(clase))
        log_prob = math.log(class_counts[label] / total_docs)
        # log(producto P(palabra|clase)) = suma de log(P)
        for word in tokens:
            log_prob += math.log(cond_prob(word, label))
        probs[label] = log_prob

    # Exponenciar (sin normalizar)
    exp_probs = {label: math.exp(log_prob) for label, log_prob in probs.items()}

    # Normalizar (para que sumen 1)
    total = sum(exp_probs.values())
    normalized = {label: exp_probs[label] / total for label in exp_probs}

    # Devolver ambos resultados
    return exp_probs, normalized

# Prueba con la frase
sentence = "I hated the poor acting"
exp_probs, result = predict(sentence)

# Mostrar resultados
print("Probabilidades SIN normalizar:")
for label, prob in exp_probs.items():
    print(f"{label}: {prob:.8f}")

print("\nProbabilidades normalizadas:")
for label, prob in result.items():
    print(f"{label}: {prob:.5f}")

print("\nProbabilidades en porcentaje:")
for label, prob in result.items():
    print(f"{label}: {prob*100:.3f}%")

print("\nPredicción:", max(result, key=result.get))
