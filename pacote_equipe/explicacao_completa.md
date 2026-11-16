# Explicacao completa de `solucao.py`

Este documento descreve o funcionamento de [solucao.py](./solucao.py). O arquivo contém uma unica função de nivel superior, `principal(observada, máscara)`. As funcoes auxiliares são definidas dentro dela para manter a estrutura pedida pelo avaliador.

## 1. Objetivo e dados de entrada

`observada` é uma matriz retangular. Nas posicoes conhecidas, contém o valor observado; nas ocultas, o conteudo não deve ser usado para estimar a resposta.

`máscara` tem as mesmas dimensoes. Um valor verdadeiro indica uma celula conhecida; um valor falso indica uma celula a preencher.

A função devolve uma lista de listas numérica com as mesmas dimensoes. Ela copia os valores conhecidos para a saida e preenche apenas as posicoes ocultas.

## 2. Preparação e casos pequenos

No início, a função:

1. Verifica se a matriz tem linhas. Se estiver vazia, devolve uma lista vazia.
2. Obtém o numero de colunas pela primeira linha.
3. Cria `resultado`, inicialmente com `0.0` nas células ocultas.
4. Cria `conhecidos`, uma lista de triplas `(linha, coluna, valor)` para as células marcadas como conhecidas. Os valores são convertidos para `float`.

Os casos sem informacao suficiente são tratados diretamenté:

- **Nenhum valor conhecido:** retorna a matriz de zeros criada para a saida, pois não existe informacao para inferir outro valor.
- **Menos de tres valores conhecidos:** preenche cada buraco com a média dos conhecidos e retorna.

`calcular_média` calcula essa média percorrendo a lista de triplas.

## 3. Solucao dos sistemas lineares: `resolver`

Alguns modelos precisam ajustar coeficientes. A função auxiliar `resolver(sistema, valores)` resolve o sistema linear representado por uma matriz de coeficientes e um vetor de termos independentes.

Ela usa eliminacao de Gauss com pivôtamento parcial:

1. Acrescenta o vetor de valores como ultima coluna da matriz.
2. Em cada coluna, escolhe como pivô a linha com maior valor absoluto naquela coluna.
3. Troca as linhas, normaliza a linha do pivô e elimina essa coluna das outras linhas.
4. Devolve a ultima coluna da matriz transformada como vetor de solucao.

O pivôtamento ajuda a reduzir problemas numericos. Se encontrar um pivô muito pequeno, a função pula a normalizacao daquela coluna. Os modelos também adicionam regularizacao diagonal para reduzir a chance de sistemas mal condicionados.

## 4. Modelo polinomial: `ajustar_polinomio`

O modelo polinomial estima uma superficie em função das coordenadas da celula:

```text
valor estimado = função(x, y)
```

As coordenadas das linhas e colunas são normalizadas aproximadamente para o intervalo `[-1, 1]`. Isso reduz a escala dos valores usados no ajuste.

Para um grau `d`, são usados todos os termos `x**a * y**b` que satisfazem `a + b <= d`. Por exemplo, no grau 2, os termos sao:

```text
1, x, y, x**2, x*y, y**2
```

Com as células conhecidas, a função monta as equacoes normais de mínimos quadrados e adiciona `1e-7` a diagonal do sistema. Em seguida, usa `resolver` para encontrar os coeficientes. O ajuste retorna os termos e os coeficientes, que seráo usados para calcular as previsões.

O grau não é escolhido antecipadamenté: os graus de 1 a 4 são candidatos, desde que a quantidade de termos caiba nos dados disponiveis para o treino.

## 5. Modelo aditivo: `ajustar_aditivo`

O modelo aditivo estima cada celula como a soma de tres componentes:

```text
intercepto + efeito da linha + efeito da coluna
```

Por exemplo, ele pode representar uma matriz em que certas linhas tendem a ter valores maiores e certas colunas tendem a ter valores menores.

A função monta um sistema linear usando uma variavel para o intercepto e variaveis de efeito para cada linha e coluna. Adiciona `0.05` a diagonal dos efeitos para regularizar o ajuste e `1e-8` ao intercepto. Os coeficientes resultantes são resolvidos por `resolver`.

Esse modelo so e adicionado aos candidatos quando o conjunto de treino tem pelo menos tantas observacoes quanto parametros do modelo.

## 6. Geracao de previsões: `prever`

`prever(modelo, treino, consultas)` recebe um modelo, uma lista de valores usados para ajusta-lo e uma lista de coordenadas a estimar. O primeiro elemento de `modelo` identifica o tipo:

- **`média`:** retorna a média dos valores de treino para cada consulta.
- **`polinomio`:** ajusta os coeficientes com `ajustar_polinomio` e avalia a expressão em cada coordenada.
- **`aditivo`:** ajusta os efeitos de linha e coluna e soma os efeitos correspondentes a cada consulta.
- **`kernel`:** combina os valores conhecidos com pesos maiores para células proximas.

### Media local ponderada (`kernel`)

Para uma consulta `(i, j)` e uma observacao `(linha, coluna, valor)`, a distância quadrática é:

```text
(i - linha)**2 + (j - coluna)**2
```

O peso diminui com essa distância segundo uma curva gaussiana:

```text
peso = exp(-distância_quadrática / (2 * largura**2))
```

A previsão e a média dos valores de treino ponderada por esses pesos. Se a soma dos pesos for praticamente zero, a função recorre a média global do treino.

As larguras avaliadas são `0.65`, `0.9`, `1.2`, `1.6`, `2.2`, `3.0`, `4.0` e `6.0`. Como as coordenadas são inteiras, os pesos são precomputados para cada distância quadrática possivel e guardados no proprio modelo. Esse cache evita recalcular potencias durante cada previsão e reduz o tempo de execucao.

## 7. Validacao cruzada e divisão dos dados

Para comparar os modelos sem consultar as células realmente ocultas, a função separa temporariamente parte das células conhecidas:

1. Define ate cinco grupos, reduzindo esse numero se houver menos de cinco observacoes.
2. Calcula uma chave deterministica a partir da coordenada de cada celula conhecida.
3. Ordena as coordenadas por essa chave e distribui as células em rodizio entre os grupos.
4. Para cada grupo, treina os modelos nos demais grupos e calcula previsões para o grupo separado.

O uso de uma chave baseada nas coordenadas torna a divisão repetivel, sem depender de geracao aleatoria ou de semente.

## 8. Modelos candidatos e selecao

A lista de candidatos inclui:

- média global;
- modelo aditivo, se houver observacoes suficientes;
- modelos polinomiais de graus 1 a 4, limitados pela quantidade de termos que o menor conjunto de treino pode sustentar;
- modelos de kernel para cada largura da lista acima.

Para cada modelo, a função acumula o erro quadrático das previsões nos valores conhecidos que ficaram fora do treino. Divide a soma pelo numero total de valores validados para obter o erro quadrático medio.

Primeiro, escolhe o candidato com menor erro. Depois, considera também candidatos cujo erro não exceda em mais de 2% o menor erro. Entre esses, prefere o candidato com menor classificação de complexidadé:

- média: classificação 0;
- polinomio: classificação igual ao grau;
- aditivo: classificação 3;
- kernel: classificação 10.

Essa regra tende a evitar um modelo mais complexo quando sua melhora estimada e muito pequena. A validação e apenas uma estimativa com os valores conhecidos; não garante qual modelo tera o menor erro nas células ocultas ou em um lote futuro.

## 9. Ajuste final e montagem da resposta

Depois de selecionar o modelo, a função:

1. Lista as coordenadas onde `máscara` é falsa.
2. Ajusta o modelo escolhido usando **todos** os valores conhecidos.
3. Calcula as previsões para as células ocultas.
4. Grava essas previsões como `float` nas posicoes correspondentes de `resultado`.
5. Devolve a matriz completa.

As posicoes conhecidas não são reestimadas: seus valores originais, convertidos para `float`, permanecem na saida.

## 10. Compatibilidade e desempenho

O codigo não importa bibliotecas, não define classes e não usa geradores de números aleatorios. As funcoes auxiliares são locais a `principal`. A solucao foi compilada e executada sob RestrictedPython, e testada nos dez exemplos publicos quanto a dimensoes, valores finitos e preservacao das células conhecidas.

Na medição realizada com o ambiente restrito e o lote publico, o maior tempo por matriz foi aproximadamente **1,28 s**, e o total foi aproximadamente **2,99 s**. Esses tempos dependem do hardware e da carga do ambiente; não constituem garantia para outros lotes ou maquinas.
