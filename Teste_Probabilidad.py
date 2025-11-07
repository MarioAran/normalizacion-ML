from collections import defaultdict
import math

# Datos de entrenamiento
data = [
    ("I loved the movie", "+"),
    ("I hated the movie", "-"),
    ("A great movie. Good movie.", "+"),
    ("Poor acting", "-"),
    ("Great acting. A good movie.", "+"),
    ("Good acting", "-"),
    ("I love the acting", "+")
]

# Ajustes
def tokenize(text):
    return text.lower().replace(".", "").split()

# Contadores
word_counts = {
    #El defaultdict crea diccionarios sin tener que comprobar si una palabra ya existe antes de usarla
    "+": defaultdict(int), 
    "-": defaultdict(int)
}
class_counts = {"+": 0, "-": 0}

#Contar palabras por clase
for text, label in data:
    class_counts[label] += 1
    for word in tokenize(text):
        word_counts[label][word] += 1

#Vocabulario total 
vocab = set()
for label in word_counts:
    vocab.update(word_counts[label].keys())
V = len(vocab)

#Totales por clase
total_words = {
    label: sum(word_counts[label].values())
    for label in word_counts
}

#Calculo de P(palabra | clase) con Laplace
def cond_prob(word, label):
    return (word_counts[label][word] + 1) / (total_words[label] + V)

#Prediccion
def predict(text):
    tokens = tokenize(text)
    probs = {}
    total_docs = sum(class_counts.values())
    
    for label in ["+", "-"]:
        #log(P(clase))
        log_prob = math.log(class_counts[label] / total_docs)
        
        #log(producto P(palabra|clase)) = suma de log(P)
        for word in tokens:
            log_prob += math.log(cond_prob(word, label))
        
        probs[label] = log_prob
    
    # Convertir de log a valor normalizado
    exp_probs = {label: math.exp(log_prob) for label, log_prob in probs.items()}
    total = sum(exp_probs.values())
    normalized = {label: prob / total for label, prob in exp_probs.items()}
    
    return normalized

#Prueba con el registro 6
sentence = "I hated the poor acting"
result = predict(sentence)

print("Probabilidades normalizadas:")
for label, prob in result.items():
    print(f"{label}: {prob:.5f}")

print("\nPrediccion:", max(result, key=result.get))
