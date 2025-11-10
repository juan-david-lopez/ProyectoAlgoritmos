

# Algoritmos de Similitud de Textos: Guía Completa

**Autor:** Sistema de Análisis de Similitud
**Fecha:** 2025-10-27
**Versión:** 1.0

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Algoritmos Implementados](#algoritmos-implementados)
   - [Distancia de Levenshtein](#1-distancia-de-levenshtein)
   - [TF-IDF + Similitud del Coseno](#2-tf-idf--similitud-del-coseno)
   - [Índice de Jaccard](#3-índice-de-jaccard)
   - [Similitud de N-gramas](#4-similitud-de-n-gramas)
   - [Sentence-BERT (S-BERT)](#5-sentence-bert-s-bert)
   - [BERT](#6-bert)
3. [Tabla Comparativa](#tabla-comparativa)
4. [Casos de Uso Recomendados](#casos-de-uso-recomendados)
5. [Ejemplos de Código](#ejemplos-de-código)
6. [Referencias](#referencias)

---

## Introducción

La **similitud de textos** es fundamental en NLP para tareas como búsqueda semántica, detección de plagio, agrupación de documentos, y sistemas de recomendación. Este documento describe 6 algoritmos implementados, desde métodos clásicos hasta modelos de deep learning.

### Conceptos Clave

- **Similitud léxica**: Comparación basada en palabras exactas
- **Similitud semántica**: Comparación basada en significado
- **Embedding**: Representación vectorial de texto
- **Normalización**: Mapeo de resultados al rango $[0, 1]$

---

## Algoritmos Implementados

### 1. Distancia de Levenshtein

#### Descripción

La **distancia de Levenshtein** (o distancia de edición) mide el número mínimo de operaciones de edición necesarias para transformar una cadena en otra.

#### Matemática

**Operaciones permitidas:**
- Inserción de un carácter
- Eliminación de un carácter
- Sustitución de un carácter

**Definición recursiva:**

$$
\text{LD}(i, j) = \begin{cases}
\max(i, j) & \text{si } \min(i,j) = 0 \\
\min \begin{cases}
\text{LD}(i-1, j) + 1 & \text{(eliminación)} \\
\text{LD}(i, j-1) + 1 & \text{(inserción)} \\
\text{LD}(i-1, j-1) + \text{cost} & \text{(sustitución)}
\end{cases} & \text{si } \min(i,j) > 0
\end{cases}
$$

donde:

$$
\text{cost} = \begin{cases}
0 & \text{si } s_1[i] = s_2[j] \\
1 & \text{si } s_1[i] \neq s_2[j]
\end{cases}
$$

**Similitud normalizada:**

$$
\text{sim}(s_1, s_2) = 1 - \frac{\text{LD}(s_1, s_2)}{\max(|s_1|, |s_2|)}
$$

#### Ejemplo Paso a Paso

Comparar **"kitten"** vs **"sitting"**:

| | ε | s | i | t | t | i | n | g |
|---|---|---|---|---|---|---|---|---|
| **ε** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| **k** | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| **i** | 2 | 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| **t** | 3 | 3 | 2 | 1 | 2 | 3 | 4 | 5 |
| **t** | 4 | 4 | 3 | 2 | 1 | 2 | 3 | 4 |
| **e** | 5 | 5 | 4 | 3 | 2 | 2 | 3 | 4 |
| **n** | 6 | 6 | 5 | 4 | 3 | 3 | 2 | 3 |

- **Distancia:** 3
- **Similitud:** $1 - \frac{3}{7} \approx 0.571$

**Operaciones:** k→s, e→i, insertar g

#### Complejidad

- **Tiempo:** $O(m \times n)$
- **Espacio:** $O(m \times n)$ (optimizable a $O(\min(m,n))$)

#### Pros y Contras

✅ **Ventajas:**
- Simple e intuitivo
- No requiere preprocesamiento
- Útil para detección de errores tipográficos

❌ **Desventajas:**
- Muy lento para textos largos ($O(n^2)$)
- No captura semántica
- Sensible al orden de caracteres

#### Caso de Uso

- Corrección ortográfica
- Detección de errores tipográficos
- Comparación de nombres
- Textos muy cortos

---

### 2. TF-IDF + Similitud del Coseno

#### Descripción

**TF-IDF** (Term Frequency-Inverse Document Frequency) pondera términos por su importancia, y la **similitud del coseno** mide el ángulo entre vectores de documentos.

#### Matemática

**1. Term Frequency (TF):**

$$
\text{TF}(t, d) = \frac{\text{freq}(t, d)}{\sum_{t' \in d} \text{freq}(t', d)}
$$

Variante log-normalizada:

$$
\text{TF}(t, d) = \begin{cases}
1 + \log(\text{freq}(t, d)) & \text{si freq}(t, d) > 0 \\
0 & \text{si freq}(t, d) = 0
\end{cases}
$$

**2. Inverse Document Frequency (IDF):**

$$
\text{IDF}(t, D) = \log \frac{N}{\text{df}(t)}
$$

donde:
- $N$ = número total de documentos
- $\text{df}(t)$ = número de documentos que contienen $t$

Variante suavizada:

$$
\text{IDF}(t, D) = \log \frac{N + 1}{\text{df}(t) + 1} + 1
$$

**3. TF-IDF:**

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

**4. Similitud del Coseno:**

$$
\text{sim}(\mathbf{v}_1, \mathbf{v}_2) = \cos(\theta) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \times \|\mathbf{v}_2\|} = \frac{\sum_{i=1}^{n} v_{1i} \times v_{2i}}{\sqrt{\sum_{i=1}^{n} v_{1i}^2} \times \sqrt{\sum_{i=1}^{n} v_{2i}^2}}
$$

Rango: $[0, 1]$ para vectores no negativos.

#### Ejemplo Paso a Paso

**Documentos:**
- $D_1$: "cat sat mat"
- $D_2$: "cat sat hat"

**Paso 1: Vocabulario**

$V = \{\text{cat}, \text{sat}, \text{mat}, \text{hat}\}$

**Paso 2: TF**

| | cat | sat | mat | hat |
|---|---|---|---|---|
| $D_1$ | 0.33 | 0.33 | 0.33 | 0 |
| $D_2$ | 0.33 | 0.33 | 0 | 0.33 |

**Paso 3: IDF** ($N=2$)

- cat: $\log(\frac{2}{2}) = 0$
- sat: $\log(\frac{2}{2}) = 0$
- mat: $\log(\frac{2}{1}) = 0.693$
- hat: $\log(\frac{2}{1}) = 0.693$

**Paso 4: TF-IDF**

| | cat | sat | mat | hat |
|---|---|---|---|---|
| $D_1$ | 0 | 0 | 0.231 | 0 |
| $D_2$ | 0 | 0 | 0 | 0.231 |

**Paso 5: Similitud del Coseno**

$$
\text{sim}(D_1, D_2) = \frac{0 \times 0 + 0 \times 0 + 0.231 \times 0 + 0 \times 0.231}{\sqrt{0.231^2} \times \sqrt{0.231^2}} = 0
$$

#### Complejidad

- **Tiempo:** $O(n \times m)$ donde $n$ = documentos, $m$ = términos
- **Espacio:** $O(n \times m)$ (matriz dispersa)

#### Pros y Contras

✅ **Ventajas:**
- Rápido y escalable
- Independiente de longitud del documento
- Considera importancia de términos

❌ **Desventajas:**
- No captura semántica
- Ignora orden de palabras
- Requiere corpus representativo

#### Caso de Uso

- Búsqueda de documentos
- Recuperación de información
- Clasificación de textos
- Sistemas de recomendación

---

### 3. Índice de Jaccard

#### Descripción

El **Índice de Jaccard** mide la similitud entre dos conjuntos finitos como la razón entre la intersección y la unión.

#### Matemática

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

**Propiedades:**
- $J(A, A) = 1$ (reflexividad)
- $J(A, B) = J(B, A)$ (simetría)
- $0 \leq J(A, B) \leq 1$
- $J(A, \emptyset) = 0$

**Distancia de Jaccard:**

$$
d(A, B) = 1 - J(A, B)
$$

#### Ejemplo Paso a Paso

**Textos:**
- $T_1$: "the cat sat on mat"
- $T_2$: "the dog sat on log"

**Paso 1: Tokenización**

- $A = \{\text{the}, \text{cat}, \text{sat}, \text{on}, \text{mat}\}$
- $B = \{\text{the}, \text{dog}, \text{sat}, \text{on}, \text{log}\}$

**Paso 2: Intersección y Unión**

- $A \cap B = \{\text{the}, \text{sat}, \text{on}\}$ → $|A \cap B| = 3$
- $A \cup B = \{\text{the}, \text{cat}, \text{sat}, \text{on}, \text{mat}, \text{dog}, \text{log}\}$ → $|A \cup B| = 7$

**Paso 3: Similitud**

$$
J(A, B) = \frac{3}{7} \approx 0.429
$$

#### Complejidad

- **Tiempo:** $O(n + m)$ usando conjuntos hash
- **Espacio:** $O(n + m)$

#### Pros y Contras

✅ **Ventajas:**
- Simple e intuitivo
- Muy rápido
- Útil para conjuntos

❌ **Desventajas:**
- No considera frecuencias
- Ignora importancia de términos
- No captura semántica

#### Caso de Uso

- Comparación de etiquetas
- Palabras clave
- Categorías
- Conjuntos de hashtags

---

### 4. Similitud de N-gramas

#### Descripción

Un **n-grama** es una secuencia contigua de $n$ caracteres. La similitud se calcula comparando conjuntos de n-gramas.

#### Matemática

**Extracción de n-gramas:**

Para un texto $s$ de longitud $L$:

$$
\text{N-gramas}(s, n) = \{s[i:i+n] \mid i = 0, 1, \ldots, L-n\}
$$

**Similitud de Jaccard:**

$$
\text{sim}_J(s_1, s_2) = \frac{|G_1 \cap G_2|}{|G_1 \cup G_2|}
$$

**Coeficiente de Dice:**

$$
\text{sim}_D(s_1, s_2) = \frac{2 \times |G_1 \cap G_2|}{|G_1| + |G_2|}
$$

**Relación:** $\text{sim}_D \geq \text{sim}_J$ siempre.

#### Ejemplo Paso a Paso

**Bigramas** ($n=2$):

- $s_1 = \text{"abc"}$
- $s_2 = \text{"bcd"}$

**Paso 1: Extracción**

- $G_1 = \{\text{ab}, \text{bc}\}$
- $G_2 = \{\text{bc}, \text{cd}\}$

**Paso 2: Intersección y Unión**

- $G_1 \cap G_2 = \{\text{bc}\}$ → $|G_1 \cap G_2| = 1$
- $G_1 \cup G_2 = \{\text{ab}, \text{bc}, \text{cd}\}$ → $|G_1 \cup G_2| = 3$

**Paso 3: Similitud**

- **Jaccard:** $\frac{1}{3} \approx 0.333$
- **Dice:** $\frac{2 \times 1}{2 + 2} = 0.500$

#### Complejidad

- **Tiempo:** $O(m + n)$
- **Espacio:** $O(m + n)$

#### Pros y Contras

✅ **Ventajas:**
- Robusto a errores tipográficos
- Independiente del idioma
- Captura similitud de subcadenas

❌ **Desventajas:**
- Sensible a la elección de $n$
- No captura semántica
- Puede ser lento para $n$ grande

#### Caso de Uso

- Detección de plagio
- Comparación multilingüe
- Textos con errores
- Matching difuso

---

### 5. Sentence-BERT (S-BERT)

#### Descripción

**S-BERT** es una modificación de BERT que genera embeddings de oraciones semánticamente significativos mediante una arquitectura siamesa.

#### Arquitectura

**1. Transformer Encoder:**

Self-attention multi-cabeza:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

donde:
- $Q, K, V$ = matrices de query, key, value
- $d_k$ = dimensión de las claves

**2. Mean Pooling:**

$$
\mathbf{e}_{\text{sentence}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{e}_i
$$

donde $\mathbf{e}_i$ son los embeddings de tokens.

**3. Similitud del Coseno:**

$$
\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \times \|\mathbf{v}\|}
$$

Mapeo a $[0, 1]$:

$$
\text{sim}_{\text{norm}} = \frac{\text{sim} + 1}{2}
$$

#### Entrenamiento (Siamese Network)

**Función de pérdida contrastiva:**

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \|\mathbf{u}_i - \mathbf{v}_i\|^2 + (1 - y_i) \max(0, m - \|\mathbf{u}_i - \mathbf{v}_i\|)^2 \right]
$$

donde:
- $y_i \in \{0, 1\}$ = etiqueta de similitud
- $m$ = margen de separación

**Triplet Loss:**

$$
\mathcal{L} = \max(0, \|\mathbf{a} - \mathbf{p}\|^2 - \|\mathbf{a} - \mathbf{n}\|^2 + \alpha)
$$

donde:
- $\mathbf{a}$ = anchor
- $\mathbf{p}$ = positive
- $\mathbf{n}$ = negative
- $\alpha$ = margen

#### Complejidad

- **Encoding:** $O(n \times d^2)$ donde $n$ = tokens, $d$ = dimensión
- **Comparación:** $O(d)$ con embeddings precalculados
- **Memoria:** ~500MB

#### Pros y Contras

✅ **Ventajas:**
- Captura semántica profunda
- 2000x más rápido que BERT para comparaciones
- Multilingüe (según modelo)
- Embeddings comparables directamente

❌ **Desventajas:**
- Requiere GPU (recomendado)
- Mayor uso de memoria que métodos clásicos
- Modelo grande (~100-500MB)

#### Caso de Uso

- Búsqueda semántica
- Agrupación de documentos
- Sistemas de recomendación
- Detección de duplicados semánticos

---

### 6. BERT

#### Descripción

**BERT** (Bidirectional Encoder Representations from Transformers) es un modelo transformer bidireccional pre-entrenado para comprensión profunda del lenguaje.

#### Arquitectura

**1. Bidireccionalidad:**

A diferencia de modelos unidireccionales, BERT procesa contexto en ambas direcciones:

$$
\mathbf{h}_i = f(\text{left\_context}_i, \text{right\_context}_i)
$$

**2. Embeddings:**

$$
\mathbf{E}_{\text{token}} = \mathbf{E}_{\text{word}} + \mathbf{E}_{\text{position}} + \mathbf{E}_{\text{segment}}
$$

**3. Multi-Head Self-Attention:**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

donde:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**4. Feed-Forward Network:**

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2
$$

#### Pre-entrenamiento

**1. Masked Language Model (MLM):**

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(w_i \mid \text{context})
$$

**2. Next Sentence Prediction (NSP):**

$$
\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} \mid \text{sent}_A, \text{sent}_B)
$$

**Pérdida total:**

$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

#### Similitud de Oraciones

**Mean Pooling:**

$$
\mathbf{h}_{\text{sent}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i
$$

**Similitud:**

$$
\text{sim}(s_1, s_2) = \text{cosine}(\mathbf{h}_1, \mathbf{h}_2)
$$

#### Complejidad

- **Tiempo:** $O(n^2 \times d)$ (self-attention cuadrática)
- **Memoria:** ~500MB-1GB
- **Parámetros:** 110M (base), 340M (large)

#### Pros y Contras

✅ **Ventajas:**
- Máxima precisión semántica
- Estado del arte en NLP
- Contexto bidireccional completo

❌ **Desventajas:**
- Muy lento (especialmente para comparaciones por pares)
- Alto uso de memoria
- Requiere GPU para uso práctico

#### Caso de Uso

- Análisis profundo de similitud
- Cuando precisión es crítica
- Datasets pequeños
- Investigación y benchmarking

---

## Tabla Comparativa

| Algoritmo | Tipo | Velocidad | Precisión Semántica | Uso Memoria | Mejor Para |
|-----------|------|-----------|---------------------|-------------|------------|
| **Levenshtein** | Edición | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | Textos cortos, errores tipográficos |
| **TF-IDF + Coseno** | Léxico | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Búsqueda de documentos, gran escala |
| **Jaccard** | Conjunto | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | Etiquetas, palabras clave |
| **N-grama** | Subcadena | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Plagio, multilingüe |
| **S-BERT** | Semántico | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Búsqueda semántica, balance |
| **BERT** | Semántico | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Máxima precisión, datasets pequeños |

**Leyenda:**
- ⭐⭐⭐⭐⭐ = Excelente
- ⭐⭐⭐⭐ = Muy bueno
- ⭐⭐⭐ = Bueno
- ⭐⭐ = Regular
- ⭐ = Limitado

---

## Casos de Uso Recomendados

### Comparación de Abstracts Científicos

**Recomendación Principal: S-BERT**

**Razones:**
1. Captura similitud semántica entre conceptos
2. Robusto a diferentes formulaciones
3. Buen balance velocidad/precisión
4. Entrenado en textos científicos

**Alternativas:**
- **BERT:** Si precisión es absolutamente crítica y dataset es pequeño
- **TF-IDF:** Si velocidad es crítica o dataset es muy grande
- **N-grama:** Si hay múltiples idiomas o errores

### Detección de Plagio

**Recomendación:** N-grama + Levenshtein

**Pipeline:**
1. Filtrar con N-grama (rápido, primera pasada)
2. Verificar con Levenshtein (detectar paráfrasis leves)
3. Confirmar con S-BERT (similitud semántica)

### Búsqueda en Tiempo Real

**Recomendación:** TF-IDF o Jaccard

**Razones:**
- Muy rápidos (<1ms por comparación)
- Escalables a millones de documentos
- Bajo uso de memoria

### Análisis de Sentimientos Similares

**Recomendación:** S-BERT o BERT

**Razones:**
- Capturan matices semánticos
- Reconocen sinónimos y contexto
- Mejor para significado que léxico

### Corrección Ortográfica

**Recomendación:** Levenshtein

**Razones:**
- Diseñado para distancia de edición
- Simple y efectivo
- No requiere corpus

---

## Ejemplos de Código

### Ejemplo 1: Comparación Básica

```python
from src.algorithms.levenshtein import LevenshteinComparator

comparator = LevenshteinComparator()
sim = comparator.similarity("hello", "hallo")
print(f"Similitud: {sim:.3f}")
```

### Ejemplo 2: Comparación Múltiple

```python
from src.algorithms.tfidf_cosine import TFIDFCosineComparator

comparator = TFIDFCosineComparator()
texts = [
    "machine learning algorithms",
    "deep learning networks",
    "cooking recipes"
]

matrix = comparator.compare_multiple(texts)
print(matrix)
```

### Ejemplo 3: Búsqueda Semántica

```python
from src.algorithms.sbert import SBERTComparator

comparator = SBERTComparator()
query = "natural language processing"
candidates = [
    "NLP and text analysis",
    "cooking Italian food",
    "understanding human language"
]

results = comparator.find_most_similar(query, candidates, top_k=2)
for idx, sim in results:
    print(f"{candidates[idx]}: {sim:.3f}")
```

### Ejemplo 4: Pipeline Completo

```python
from src.algorithms.similarity_comparator import SimilarityComparator

# Cargar datos y comparar
comparator = SimilarityComparator('data/unified_articles.json')
selected = comparator.select_articles(['article_1', 'article_2', 'article_3'])
abstracts = [art['abstract'] for art in selected]

# Ejecutar todos los algoritmos
results = comparator.compare_all_algorithms(abstracts)

# Generar visualizaciones
comparator.visualize_results(results, 'output/visualizations')

# Generar reporte
comparator.generate_detailed_report(results, 'output/report.md', selected)
```

---

## Referencias

### Papers Fundamentales

1. **Levenshtein, V. I.** (1966). *Binary codes capable of correcting deletions, insertions, and reversals.* Soviet Physics Doklady, 10(8), 707-710.

2. **Salton, G., & McGill, M. J.** (1983). *Introduction to modern information retrieval.* McGraw-Hill.

3. **Jaccard, P.** (1901). *Étude comparative de la distribution florale dans une portion des Alpes et des Jura.* Bulletin de la Société Vaudoise des Sciences Naturelles, 37, 547-579.

4. **Kondrak, G.** (2005). *N-Gram Similarity and Distance.* String Processing and Information Retrieval, 115-126.

5. **Reimers, N., & Gurevych, I.** (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.

6. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.

### Recursos Adicionales

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

## Apéndice: Fórmulas Clave

### Similitud del Coseno

$$
\text{cosine}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}
$$

### Distancia Euclidiana

$$
d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^n (A_i - B_i)^2}
$$

### Softmax

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

### Layer Normalization

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mu}{\sigma} + \beta
$$

donde $\mu$ y $\sigma$ son la media y desviación estándar de $\mathbf{x}$.

---

**Fin del documento**
