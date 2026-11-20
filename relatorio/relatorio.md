# Relatório — Reconstrução de matrizes com entradas ocultas

**Disciplina:** CIN0138 — Álgebra Vetorial e Linear para Computação
**Equipe:** Byanca Hayally Candido de Souza, Jaianny Karla de Oliveira Souza e Layse Maryane Feitoza Gomes

## 1. O problema

Recebemos uma matriz `observada` (m × n, até 40 × 40) em que parte das
entradas está oculta, e uma `mascara` que indica quais são conhecidas. A
função `principal` devolve a matriz completa. O erro é o NRMSE nas posições
ocultas: o RMSE dividido pelo desvio-padrão dos valores ocultos verdadeiros.
Assim, preencher tudo com a média dos ocultos daria NRMSE ≈ 1.

## 2. Ideia central: posto baixo

Analisando o lote público, vimos que a maioria das matrizes é
(aproximadamente) de **posto baixo**: poucos valores singulares concentram
quase toda a "energia" da matriz, e o resto é ruído pequeno. Se X tem posto k,
ela se fatora como

  X = U Vᵀ,  com U (m × k) e V (n × k),

isto é, cada linha de X é uma combinação linear das k linhas de Vᵀ. São só
k(m + n) parâmetros, bem menos que as m·n entradas. Com cobertura suficiente
das linhas e colunas, as entradas conhecidas podem permitir estimar U e V. O
produto U Vᵀ fornece as estimativas para as posições ocultas.

## 3. Como estimamos U e V: mínimos quadrados alternados (ALS)

Procuramos U e V que minimizam o erro **só nas entradas conhecidas**, com um
termo de regularização (ridge) que evita valores exagerados:

  soma sobre (i, j) conhecidas de (x_ij − u_i · v_j)² + λ (‖U‖² + ‖V‖²).

Com V fixo, cada linha u_i é a solução de um problema de mínimos quadrados
pequeno, dado pelas equações normais:

  (V_iᵀ V_i + λ I) u_i = V_iᵀ x_i,

em que V_i contém só as linhas de V das colunas conhecidas na linha i. Depois
fixamos U e resolvemos cada v_j do mesmo jeito, e repetimos. A cada passo, o
objetivo regularizado que estamos minimizando não aumenta. É a mesma ideia de
"rodadas" da estratégia de efeitos de linha e coluna do laboratório,
generalizada para k dimensões.

Detalhes que se mostraram importantes nos testes:

- **Ponto de partida:** começamos da SVD truncada da matriz com as ocultas
  preenchidas pela média. Partir de zeros fazia o ALS parar em mínimos locais
  ruins quando a matriz tinha um deslocamento grande (por exemplo, valores em
  torno de 16 com pequenas variações).
- **Escala da regularização:** usamos λ = c · s_k, proporcional ao k-ésimo
  valor singular, para não depender da escala dos números da matriz.

## 4. Escolha do posto k e de λ: validação cruzada

O posto certo é decisivo: errar k por 1 multiplicava o erro por 10 em alguns casos. Para escolher sem conhecer o gabarito, escondemos de propósito cerca de
10% das entradas **conhecidas**, ajustamos o modelo com o resto e medimos o
erro nessas entradas. Repetimos com grupos diferentes e somamos os erros.

- Os grupos são formados de modo **determinístico** (o avaliador proíbe fixar
  semente), ordenando as posições pela sequência da razão áurea.
- Em cada grupo, tudo o que o modelo usa (inclusive a escala da
  regularização, s_k) é calculado só com as entradas de treino do grupo, para
  que as entradas retiradas não vazem para o ajuste.
- Um grupo nunca deixa uma linha ou coluna com menos de 3 conhecidas, para a
  validação se parecer com a situação real.
- Testamos k = 1, 2, … até o erro parar de cair. Cada k parte da solução do
  k − 1 ("continuação"), o que deixa o ALS mais estável. Depois testamos
  alguns valores de c para o k escolhido.
- Também competem dois modelos simples: a média global e os efeitos de linha
  e coluna (x_ij ≈ μ + a_i + b_j). Quando a matriz não tem estrutura de posto
  baixo, a validação escolhe um deles.
- Por último, competem polinômios de grau 1 a 4 nas coordenadas (linha,
  coluna), com todos os termos xᵃyᵇ (a + b ≤ grau), ajustados por mínimos
  quadrados. Eles servem para matrizes "suaves" na posição, como a tabuada
  (x_ij = i·j), e acertam até uma coluna inteira oculta, que a fatoração não
  consegue prever. Como em matrizes de puro ruído algum grau sempre parece um
  pouco melhor por acaso, o polinômio só entra na disputa se o seu erro de
  validação for menor que 80% do erro da média e dos efeitos.

## 5. Proteções

- Linha ou coluna sem nenhuma entrada conhecida: se o vencedor foi a
  fatoração, usamos o modelo de efeitos ali, porque a fatoração não tem
  informação para ela.
- Qualquer falha inesperada cai para os modelos simples (uma exceção
  desclassificaria a equipe), e a saída nunca tem NaN nem infinito.
- Nas execuções locais feitas para esta versão, o maior tempo observado foi
  cerca de 1,1 s por matriz; os tempos variam conforme o ambiente. O limite
  oficial é 2 s por matriz.

## 6. Resultados no lote público

Resultados do avaliador oficial da monitoria (Docker): sucesso em todas as
matrizes, NRMSE médio agregado 0,262521, tempo total 1,4853 s.

| Caso | Tamanho | Média global | Nosso algoritmo |
|---|---|---|---|
| 1 | 8 × 8 | 1,234 | 1,234 |
| 2 | 9 × 6 | 1,051 | 0,035 |
| 3 | 6 × 9 | 1,099 | 0,071 |
| 4 | 14 × 5 | 1,039 | 0,039 |
| 5 | 14 × 7 | 1,001 | 0,113 |
| 6 | 16 × 16 | 1,031 | 0,043 |
| 7 | 18 × 22 | 1,001 | 0,089 |
| 8 | 17 × 17 | 1,003 | 1,003 |
| 9 | 24 × 19 | 1,011 | 0,0002 |
| 10 | 28 × 31 | 1,001 | 0,0003 |
| **Média** | | **≈ 1,05** | **0,263** |

**Interpretação:**
- Os casos 9 e 10 têm posto exato (4 e 6) e são recuperados praticamente sem
  erro.
- Nos casos 2–7 há ruído pequeno somado a uma matriz de posto 1 a 4. O erro
  que sobra vem desse ruído, que não é previsível.
- O caso 1 é ruído em torno de uma constante, e o caso 8 tem valores
  singulares decaindo devagar (posto cheio). Neles não há estrutura a
  explorar, e o algoritmo corretamente recua para a média.

Para não ajustar o algoritmo só a esses 10 casos, também o testamos com os
mesmos gabaritos sob máscaras novas, com matrizes sintéticas parecidas (até
40 × 40), com as 12 matrizes do laboratório da disciplina e com casos extremos
(valores ocultos fora da faixa observada, deslocamento de 10 000, colunas em
escalas de 1 a 1000, 70% de entradas ocultas, linhas e colunas inteiras
ocultas). Nas 153 matrizes desses bancos, a saída foi válida. Nas execuções
locais realizadas, o maior tempo observado ficou abaixo de 1,1 s por matriz.
O avaliador local sinalizou quatro casos em que a solução ficou claramente
pior que pelo menos uma das referências simples; esses avisos não significam
falha de formato ou execução.

## Referências

- G. Strang, MIT 18.06, aula 29 (decomposição em valores singulares).
- Google, *Recommendation Systems: Matrix Factorization*.
- Documentação do NumPy: `numpy.linalg.svd`, `numpy.linalg.solve`.
