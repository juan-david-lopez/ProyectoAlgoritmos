# Metodología de Clustering Jerárquico
## Documentación Matemática Detallada

---

## 1. Introducción

Este documento presenta la fundamentación matemática del sistema de clustering jerárquico implementado para análisis bibliométrico. El sistema utiliza técnicas de agrupamiento basadas en distancias para identificar clusters temáticos en colecciones de documentos científicos.

---

## 2. Representación Vectorial

### 2.1 TF-IDF (Term Frequency-Inverse Document Frequency)

Dado un conjunto de documentos $D = \{d_1, d_2, \ldots, d_n\}$ y un vocabulario $V = \{t_1, t_2, \ldots, t_m\}$, el peso TF-IDF del término $t_j$ en el documento $d_i$ se calcula como:

$$\text{TF-IDF}(t_j, d_i) = \text{TF}(t_j, d_i) \times \text{IDF}(t_j)$$

donde:

**Term Frequency (TF):**
$$\text{TF}(t_j, d_i) = \frac{f_{i,j}}{\max_k f_{i,k}}$$

- $f_{i,j}$ = frecuencia del término $t_j$ en el documento $d_i$
- El denominador normaliza por el término más frecuente

**Inverse Document Frequency (IDF):**
$$\text{IDF}(t_j) = \log \frac{|D|}{|\{d \in D : t_j \in d\}|}$$

- $|D|$ = número total de documentos
- El denominador cuenta documentos que contienen $t_j$

**Matriz TF-IDF resultante:**
$$\mathbf{X} = \begin{bmatrix}
\text{TF-IDF}(t_1, d_1) & \cdots & \text{TF-IDF}(t_m, d_1) \\
\vdots & \ddots & \vdots \\
\text{TF-IDF}(t_1, d_n) & \cdots & \text{TF-IDF}(t_m, d_n)
\end{bmatrix}_{n \times m}$$

### 2.2 SBERT (Sentence-BERT)

SBERT genera embeddings densos utilizando redes neuronales pre-entrenadas:

$$\mathbf{v}_i = \text{SBERT}(d_i) \in \mathbb{R}^{h}$$

donde $h$ es la dimensionalidad del embedding (típicamente 384 o 768).

**Normalización L2:**

$$\hat{\mathbf{v}}_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|_2} = \frac{\mathbf{v}_i}{\sqrt{\sum_{k=1}^{h} v_{i,k}^2}}$$

Esto convierte la distancia euclidiana en similitud coseno:

$$\|\hat{\mathbf{v}}_i - \hat{\mathbf{v}}_j\|_2^2 = 2(1 - \hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j) = 2(1 - \cos(\theta_{i,j}))$$

---

## 3. Métricas de Distancia

### 3.1 Distancia Coseno

Mide el ángulo entre vectores:

$$d_{\text{cosine}}(\mathbf{x}_i, \mathbf{x}_j) = 1 - \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|}$$

**Propiedades:**
- Rango: $[0, 2]$ (para vectores normalizados: $[0, 1]$)
- Insensible a magnitud, solo considera dirección
- Ideal para comparar documentos de diferente longitud

**Similitud coseno:**
$$\cos(\theta) = \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|} \in [-1, 1]$$

### 3.2 Distancia Euclidiana

Distancia $L_2$ en el espacio vectorial:

$$d_{\text{euclidean}}(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2 = \sqrt{\sum_{k=1}^{m} (x_{i,k} - x_{j,k})^2}$$

**Propiedades:**
- Rango: $[0, \infty)$
- Sensible a magnitud y dirección
- Satisface desigualdad triangular: $d(x, z) \leq d(x, y) + d(y, z)$

### 3.3 Distancia Manhattan

Distancia $L_1$ (city-block):

$$d_{\text{manhattan}}(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_1 = \sum_{k=1}^{m} |x_{i,k} - x_{j,k}|$$

**Propiedades:**
- Más robusta a outliers que euclidiana
- Útil en espacios de alta dimensionalidad

### 3.4 Distancia de Correlación

Basada en el coeficiente de correlación de Pearson:

$$d_{\text{correlation}}(\mathbf{x}_i, \mathbf{x}_j) = 1 - \rho(\mathbf{x}_i, \mathbf{x}_j)$$

donde:

$$\rho(\mathbf{x}_i, \mathbf{x}_j) = \frac{\text{cov}(\mathbf{x}_i, \mathbf{x}_j)}{\sigma_i \sigma_j} = \frac{\sum_k (x_{i,k} - \bar{x}_i)(x_{j,k} - \bar{x}_j)}{\sqrt{\sum_k (x_{i,k} - \bar{x}_i)^2} \sqrt{\sum_k (x_{j,k} - \bar{x}_j)^2}}$$

**Propiedades:**
- Rango: $[0, 2]$
- Invariante a transformaciones afines: $a\mathbf{x} + b$
- Mide co-variación lineal

---

## 4. Clustering Jerárquico Aglomerativo

### 4.1 Algoritmo General

**Entrada:**
- Matriz de distancias $\mathbf{D} \in \mathbb{R}^{n \times n}$, donde $D_{ij} = d(\mathbf{x}_i, \mathbf{x}_j)$

**Inicialización:**
- Cada documento es un cluster: $C_1, C_2, \ldots, C_n$

**Iteración (repetir $n-1$ veces):**
1. Encontrar los dos clusters más cercanos:
   $$C_i^*, C_j^* = \arg\min_{i < j} d(C_i, C_j)$$

2. Fusionar clusters:
   $$C_{\text{new}} = C_i^* \cup C_j^*$$

3. Actualizar distancias según método de linkage

**Salida:**
- Dendrograma (matriz de linkage $\mathbf{Z} \in \mathbb{R}^{(n-1) \times 4}$)

### 4.2 Métodos de Linkage

#### 4.2.1 Single Linkage (Vecino más Cercano)

$$d_{\text{single}}(C_i, C_j) = \min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

**Propiedades:**
- Tiende a formar clusters largos y encadenados (chaining)
- Sensible a outliers y ruido
- Complejidad: $O(n^2)$

**Actualización Lance-Williams:**
$$d(C_i \cup C_j, C_k) = \min(d(C_i, C_k), d(C_j, C_k))$$

#### 4.2.2 Complete Linkage (Vecino más Lejano)

$$d_{\text{complete}}(C_i, C_j) = \max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

**Propiedades:**
- Produce clusters compactos y bien separados
- Más robusto a outliers que single
- Sensible a clusters de tamaño variable

**Actualización Lance-Williams:**
$$d(C_i \cup C_j, C_k) = \max(d(C_i, C_k), d(C_j, C_k))$$

#### 4.2.3 Average Linkage (UPGMA)

$$d_{\text{average}}(C_i, C_j) = \frac{1}{|C_i| |C_j|} \sum_{\mathbf{x} \in C_i} \sum_{\mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

**Propiedades:**
- Balance entre single y complete
- Menos sensible a outliers
- Produce clusters homogéneos

**Actualización Lance-Williams:**
$$d(C_i \cup C_j, C_k) = \frac{|C_i| d(C_i, C_k) + |C_j| d(C_j, C_k)}{|C_i| + |C_j|}$$

#### 4.2.4 Ward's Method

Minimiza la varianza intra-cluster (suma de cuadrados dentro del cluster):

$$d_{\text{ward}}(C_i, C_j) = \frac{|C_i| |C_j|}{|C_i| + |C_j|} \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2^2$$

donde $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x} \in C_k} \mathbf{x}$ es el centroide del cluster.

**Criterio de fusión (minimizar):**
$$\text{SSE}(C) = \sum_{\mathbf{x} \in C} \|\mathbf{x} - \boldsymbol{\mu}_C\|_2^2$$

**Propiedades:**
- Produce clusters de tamaño similar
- Minimiza varianza total intra-cluster
- Requiere distancia euclidiana

**Actualización Lance-Williams:**
$$d(C_i \cup C_j, C_k) = \frac{(|C_i| + |C_k|)d(C_i, C_k) + (|C_j| + |C_k|)d(C_j, C_k) - |C_k|d(C_i, C_j)}{|C_i| + |C_j| + |C_k|}$$

---

## 5. Métricas de Evaluación

### 5.1 Cophenetic Correlation Coefficient (CPCC)

Mide qué tan bien el dendrograma preserva las distancias originales.

**Definición:**

$$\text{CPCC} = \frac{\sum_{i<j} (d_{ij} - \bar{d})(c_{ij} - \bar{c})}{\sqrt{\sum_{i<j} (d_{ij} - \bar{d})^2} \sqrt{\sum_{i<j} (c_{ij} - \bar{c})^2}}$$

donde:
- $d_{ij}$ = distancia original entre documentos $i$ y $j$
- $c_{ij}$ = distancia cofenética (altura del dendrograma donde $i$ y $j$ se fusionan)
- $\bar{d}$, $\bar{c}$ = medias respectivas

**Interpretación:**
- $\text{CPCC} \approx 1$: Excelente preservación de distancias
- $0.80 < \text{CPCC} < 0.90$: Buena representación
- $0.70 < \text{CPCC} < 0.80$: Representación aceptable
- $\text{CPCC} < 0.70$: Representación pobre

### 5.2 Silhouette Score

Evalúa la calidad del clustering midiendo cohesión y separación.

**Para cada documento $i$:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

donde:

**Cohesión intra-cluster:**
$$a(i) = \frac{1}{|C_i| - 1} \sum_{\mathbf{x}_j \in C_i, j \neq i} d(\mathbf{x}_i, \mathbf{x}_j)$$

**Separación inter-cluster:**
$$b(i) = \min_{C_k \neq C_i} \frac{1}{|C_k|} \sum_{\mathbf{x}_j \in C_k} d(\mathbf{x}_i, \mathbf{x}_j)$$

**Silhouette promedio:**
$$\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

**Interpretación:**
- $s(i) \approx 1$: Documento bien clasificado, lejos de otros clusters
- $s(i) \approx 0$: Documento en la frontera entre clusters
- $s(i) < 0$: Documento probablemente mal clasificado

**Rangos generales:**
- $0.71 \leq \bar{s} \leq 1.0$: Estructura fuerte
- $0.51 \leq \bar{s} \leq 0.70$: Estructura razonable
- $0.26 \leq \bar{s} \leq 0.50$: Estructura débil
- $\bar{s} < 0.25$: Sin estructura sustancial

### 5.3 Davies-Bouldin Index (DBI)

Mide el ratio promedio de dispersión intra-cluster a separación inter-cluster.

**Definición:**

$$\text{DBI} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} R_{ij}$$

donde:

$$R_{ij} = \frac{S_i + S_j}{M_{ij}}$$

- $S_i$ = dispersión promedio del cluster $C_i$:
  $$S_i = \frac{1}{|C_i|} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|_2$$

- $M_{ij}$ = separación entre centroides:
  $$M_{ij} = \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2$$

**Propiedades:**
- Rango: $[0, \infty)$
- **Menor es mejor**
- No requiere etiquetas verdaderas
- Sensible a forma de clusters

### 5.4 Calinski-Harabasz Score (Variance Ratio Criterion)

Mide el ratio de dispersión inter-cluster a intra-cluster.

**Definición:**

$$\text{CH} = \frac{\text{SSB} / (k - 1)}{\text{SSW} / (n - k)}$$

donde:

**Between-cluster sum of squares (SSB):**
$$\text{SSB} = \sum_{i=1}^{k} |C_i| \|\boldsymbol{\mu}_i - \boldsymbol{\mu}\|_2^2$$

**Within-cluster sum of squares (SSW):**
$$\text{SSW} = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|_2^2$$

- $\boldsymbol{\mu}$ = centroide global: $\boldsymbol{\mu} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i$
- $k$ = número de clusters
- $n$ = número de documentos

**Propiedades:**
- Rango: $[0, \infty)$
- **Mayor es mejor**
- Relacionado con F-statistic de ANOVA
- Favorece clusters bien separados y compactos

---

## 6. Determinación del Número Óptimo de Clusters

### 6.1 Método del Codo (Elbow Method)

Busca el punto donde la mejora marginal en la métrica decrece significativamente.

**Para distancia de fusión:**

Analizar la secuencia de distancias $\{d_1, d_2, \ldots, d_{n-1}\}$ de la matriz de linkage.

El "codo" ocurre donde:
$$\Delta^2 d_i = (d_{i+1} - d_i) - (d_i - d_{i-1})$$
es máximo (mayor aceleración).

### 6.2 Maximización de Silhouette

$$k^* = \arg\max_{k \in K} \bar{s}(k)$$

donde $K$ es un conjunto de candidatos (típicamente $\{2, 3, \ldots, 10\}$).

### 6.3 Minimización de Davies-Bouldin

$$k^* = \arg\min_{k \in K} \text{DBI}(k)$$

### 6.4 Maximización de Calinski-Harabasz

$$k^* = \arg\max_{k \in K} \text{CH}(k)$$

### 6.5 Criterio Combinado

Dado un conjunto de métricas normalizadas:

$$k^* = \arg\max_{k \in K} \left[ w_1 \cdot \bar{s}_{\text{norm}}(k) + w_2 \cdot \text{CH}_{\text{norm}}(k) - w_3 \cdot \text{DBI}_{\text{norm}}(k) \right]$$

con pesos $w_1 + w_2 + w_3 = 1$.

---

## 7. Complejidad Computacional

### 7.1 Cálculo de Distancias

**Matriz completa:**
$$O(n^2 m)$$
- $n$ documentos
- $m$ dimensiones

**Con paralelización ($p$ procesadores):**
$$O\left(\frac{n^2 m}{p}\right)$$

### 7.2 Clustering Jerárquico

**Sin optimización:**
$$O(n^3)$$

**Con algoritmos optimizados (NN-chain, SLINK):**
- Single linkage: $O(n^2)$
- Complete/Average linkage: $O(n^2 \log n)$
- Ward: $O(n^2 \log n)$

### 7.3 Evaluación de Métricas

- **CPCC:** $O(n^2)$
- **Silhouette:** $O(n^2)$ por valor de $k$
- **Davies-Bouldin:** $O(nk)$ por valor de $k$
- **Calinski-Harabasz:** $O(nm)$ por valor de $k$

---

## 8. Sistema de Checkpoints

Para datasets grandes ($n > 500$), el sistema implementa checkpoints para evitar recalcular operaciones costosas.

**Estados guardados:**

1. **Vectores:** $\mathbf{X} \in \mathbb{R}^{n \times m}$
   - Formato: `.npy` (NumPy)
   - Tamaño: $O(nm \cdot 8$ bytes$)$ para float64

2. **Matriz de distancias:** $\mathbf{D} \in \mathbb{R}^{n \times n}$
   - Formato: `.npy`
   - Tamaño: $O(n^2 \cdot 8$ bytes$)$

3. **Matrices de linkage:** $\mathbf{Z} \in \mathbb{R}^{(n-1) \times 4}$
   - Formato: `.npy`
   - Tamaño: $O(n \cdot 32$ bytes$)$

4. **Resultados de evaluación:** diccionarios con métricas
   - Formato: `.pkl` (pickle)
   - Tamaño variable

**Hash para invalidación:**
$$h = \text{SHA256}(\text{contenido} + \text{parámetros})$$

Usa los primeros 16 caracteres hexadecimales como clave de caché.

---

## 9. Análisis de Temas (Topic Extraction)

### 9.1 TF-IDF para Clusters

Para un cluster $C$, calcular TF-IDF agregado:

$$\text{TF-IDF}_C(t_j) = \frac{1}{|C|} \sum_{d_i \in C} \text{TF-IDF}(t_j, d_i)$$

Los términos más representativos son:
$$T_C = \{t_j : \text{TF-IDF}_C(t_j) \text{ en top-}k\}$$

### 9.2 RAKE (Rapid Automatic Keyword Extraction)

**Score de palabra:**
$$S(w) = \frac{\deg(w)}{\text{freq}(w)}$$

donde:
- $\deg(w)$ = número de co-ocurrencias con otras palabras clave
- $\text{freq}(w)$ = frecuencia de la palabra

**Score de frase:**
$$S(\text{frase}) = \sum_{w \in \text{frase}} S(w)$$

### 9.3 LDA (Latent Dirichlet Allocation)

Modelo generativo probabilístico:

**Para cada documento $d$:**
1. Elegir distribución de tópicos: $\boldsymbol{\theta}_d \sim \text{Dir}(\boldsymbol{\alpha})$
2. Para cada palabra $w$ en $d$:
   - Elegir tópico: $z \sim \text{Multinomial}(\boldsymbol{\theta}_d)$
   - Elegir palabra: $w \sim \text{Multinomial}(\boldsymbol{\phi}_z)$

donde:
- $\boldsymbol{\theta}_d \in \Delta^{k-1}$ = distribución de tópicos del documento
- $\boldsymbol{\phi}_z \in \Delta^{|V|-1}$ = distribución de palabras del tópico $z$
- $\boldsymbol{\alpha}$, $\boldsymbol{\beta}$ = hiperparámetros de Dirichlet

**Inferencia:** Variational Bayes o Gibbs Sampling

---

## 10. Visualización: Reducción de Dimensionalidad

### 10.1 t-SNE (t-distributed Stochastic Neighbor Embedding)

**Objetivo:** Preservar similitudes locales en espacio 2D.

**Similitud en espacio de alta dimensión:**
$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**Similitud en espacio 2D (distribución t):**
$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**Función de costo (KL divergence):**
$$C = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Optimización:** Gradient descent sobre $\mathbf{Y} = \{\mathbf{y}_1, \ldots, \mathbf{y}_n\}$

### 10.2 UMAP (Uniform Manifold Approximation and Projection)

**Objetivo:** Preservar tanto estructura local como global.

**Basado en teoría de variedades Riemannianas.**

**Ventajas sobre t-SNE:**
- Más rápido ($O(n^{1.14})$ vs $O(n^2)$)
- Mejor preservación de estructura global
- Determinístico con semilla fija

---

## 11. Paralelización

### 11.1 Cálculo de Distancias por Bloques

Dividir matriz en $p$ bloques de filas:

$$\mathbf{D} = \begin{bmatrix}
\mathbf{D}_1 \\
\mathbf{D}_2 \\
\vdots \\
\mathbf{D}_p
\end{bmatrix}$$

donde cada worker $i$ calcula:
$$\mathbf{D}_i[j, :] = [d(\mathbf{x}_j, \mathbf{x}_1), \ldots, d(\mathbf{x}_j, \mathbf{x}_n)]$$

para $j \in \text{chunk}_i$.

**Speedup ideal:** $S = \frac{T_{\text{seq}}}{T_{\text{par}}} \approx p$ (con $p$ procesadores)

**Eficiencia:** $E = \frac{S}{p}$

### 11.2 Silhouette Score Paralelo

Dividir cálculo por clusters:

Para cada cluster $C_i$ (en paralelo):
$$s_i = \{s(j) : j \in C_i\}$$

Combinar resultados:
$$\bar{s} = \frac{1}{n} \sum_{i=1}^{k} \sum_{j \in C_i} s(j)$$

---

## 12. Referencias Matemáticas

### 12.1 Propiedades de Métricas

Una función $d: X \times X \to \mathbb{R}$ es una métrica si satisface:

1. **No negatividad:** $d(\mathbf{x}, \mathbf{y}) \geq 0$
2. **Identidad:** $d(\mathbf{x}, \mathbf{y}) = 0 \iff \mathbf{x} = \mathbf{y}$
3. **Simetría:** $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$
4. **Desigualdad triangular:** $d(\mathbf{x}, \mathbf{z}) \leq d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$

**Nota:** La distancia coseno **no** satisface la desigualdad triangular, técnicamente es una *pseudo-métrica*.

### 12.2 Descomposición de Varianza (ANOVA)

$$\text{SST} = \text{SSB} + \text{SSW}$$

donde:
- $\text{SST}$ = total sum of squares
- $\text{SSB}$ = between-cluster sum of squares
- $\text{SSW}$ = within-cluster sum of squares

**Ratio de varianza explicada:**
$$R^2 = \frac{\text{SSB}}{\text{SST}} = 1 - \frac{\text{SSW}}{\text{SST}}$$

### 12.3 Índice de Rand Ajustado (para validación externa)

Si tenemos etiquetas verdaderas $\mathcal{T}$ y predichas $\mathcal{P}$:

$$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}$$

donde:
- $n_{ij}$ = número de objetos en cluster $i$ de $\mathcal{T}$ y cluster $j$ de $\mathcal{P}$
- $a_i = \sum_j n_{ij}$, $b_j = \sum_i n_{ij}$

**Rango:** $[-1, 1]$, donde 1 = acuerdo perfecto, 0 = acuerdo aleatorio

---

## 13. Consideraciones Prácticas

### 13.1 Elección de Métrica de Distancia

| Métrica | Ventajas | Desventajas | Caso de uso |
|---------|----------|-------------|-------------|
| **Coseno** | Insensible a longitud, rápida | No métrica, ignora magnitud | Documentos de texto, TF-IDF |
| **Euclidiana** | Métrica, intuitiva | Sensible a escala y dimensionalidad | SBERT normalizado, datos numéricos |
| **Manhattan** | Robusta a outliers | No considera correlaciones | Datos dispersos, alta dimensionalidad |
| **Correlación** | Invariante a escala lineal | Costosa computacionalmente | Series temporales, perfiles de expresión |

### 13.2 Elección de Método de Linkage

| Método | Ventajas | Desventajas | Recomendación |
|--------|----------|-------------|---------------|
| **Single** | Rápido, detecta clusters alargados | Chaining, sensible a ruido | Clusters no esféricos, exploración inicial |
| **Complete** | Clusters compactos, robusto | Sensible a outliers, clusters de tamaño desigual | Clusters bien separados |
| **Average** | Balance, robusto | Computacionalmente costoso | **Uso general (recomendado)** |
| **Ward** | Minimiza varianza, clusters balanceados | Solo con distancia euclidiana | Clusters de tamaño similar, datos numéricos |

### 13.3 Umbrales de Calidad

| Métrica | Excelente | Bueno | Aceptable | Pobre |
|---------|-----------|-------|-----------|-------|
| **CPCC** | > 0.90 | 0.80-0.90 | 0.70-0.80 | < 0.70 |
| **Silhouette** | > 0.70 | 0.50-0.70 | 0.25-0.50 | < 0.25 |
| **Davies-Bouldin** | < 0.5 | 0.5-1.0 | 1.0-2.0 | > 2.0 |
| **Calinski-Harabasz** | > 100 | 50-100 | 20-50 | < 20 |

*(Valores orientativos, dependen del dominio)*

---

## 14. Conclusiones

Este sistema de clustering jerárquico proporciona:

1. **Múltiples representaciones vectoriales** (TF-IDF, SBERT)
2. **Diversas métricas de distancia** adaptadas a diferentes tipos de datos
3. **Cuatro métodos de linkage** para explorar diferentes estructuras de clustering
4. **Evaluación exhaustiva** con métricas complementarias
5. **Determinación automática** del número óptimo de clusters
6. **Análisis temático** de los clusters resultantes
7. **Visualización intuitiva** mediante dendrogramas y proyecciones 2D
8. **Optimización computacional** con paralelización y checkpoints

La metodología matemática rigurosa presentada garantiza resultados reproducibles y interpretables para análisis bibliométrico de colecciones de documentos científicos.

---

**Documento generado para:** ProyectoAlgoritmos - Sistema de Clustering Jerárquico
**Versión:** 1.0
**Fecha:** 2025-10-28
