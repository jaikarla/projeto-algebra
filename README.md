# Projeto de Álgebra Linear — Torneio de reconstrução de matrizes (CIN0138)

Função `principal(observada, mascara)` que estima as entradas ocultas de uma
matriz a partir das conhecidas, para o torneio da disciplina CIN0138
(Álgebra Vetorial e Linear para Computação, CIn/UFPE).

## Estrutura

| Caminho | O que é |
|---|---|
| `solucao/solucao.py` | **o arquivo que vai na entrega** (função `principal`) |
| `relatorio/relatorio.md` | rascunho do relatório curto exigido na entrega |
| `ferramentas/avaliar_local.py` | avaliador local (RestrictedPython, NRMSE, tempo, validade da saída) |
| `ferramentas/bancos.py` | bancos de teste: público, remascarado, sintético, laboratório, casos extremos |
| `pacote_equipe/` | pacote oficial da monitoria (avaliador em Docker + lote público) |

## Como testar

Avaliador local (rápido, sem Docker):

```bash
pip install -r ferramentas/requisitos.txt
```

```bash
python ferramentas/avaliar_local.py
```

```bash
python ferramentas/avaliar_local.py --bancos publico,remascarado,sintetico,laboratorio,extremos --detalhes
```

Ele executa o código com RestrictedPython na configuração mais estrita (e
para com erro se o RestrictedPython não estiver instalado; `--sem-restricted`
roda com Python normal, sem testar o ambiente restrito), procura nomes
proibidos pelo guia, valida a saída (dimensões, NaN, infinito), mede o tempo
de cada matriz e compara o NRMSE com dois modelos simples (média global e
efeitos de linha e coluna), avisando quando a solução fica pior que eles.

Com `--depurar`, os `except Exception:` do arquivo da solução são trocados por
`except ():`. Assim, exceções internas deixam de ser capturadas pelos fallbacks
e aparecem como falhas no avaliador:

```bash
python ferramentas/avaliar_local.py --depurar --bancos publico,remascarado,sintetico,laboratorio,extremos
```

Antes de entregar, confira também no avaliador oficial (`pacote_equipe/`,
ver o README de lá): é o resultado dele que vale.

## Resultado atual (lote público)

**Avaliador oficial (Docker da monitoria):** sucesso em todas as matrizes,
NRMSE médio agregado **0,262521**, tempo total **1,4853 s** (limite: 2 s por matriz
e 25 s por lote). O avaliador local dá os mesmos valores por matriz. Para
comparar: preencher com a média global daria ≈ 1,05.

| Caso | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| NRMSE | 1,2342 | 0,0349 | 0,0706 | 0,0385 | 0,1127 | 0,0425 | 0,0887 | 1,0026 | 0,0002 | 0,0003 |

Os casos 1 e 8 não têm estrutura aproveitável (ruído em torno de uma
constante e posto cheio); nenhum método consegue muito melhor que a média ali.

## Regras que o código precisa respeitar

Do guia oficial e de testes com o RestrictedPython:

- Só `numpy`, `scipy`, `math` e `random`; **nada de semente** (`seed`,
  `default_rng`, `RandomState`, `Random` não podem nem aparecer no código).
- Sem `class`; funções auxiliares **dentro** de `principal`.
- Limite de **2 s por matriz** e **25 s por lote**; o código não pode usar o
  módulo `time` para se controlar, então tem de ser rápido por construção.
- Desclassifica: dimensão errada, NaN, infinito, timeout ou exceção.
- No RestrictedPython padrão **não compila**: `A[i] += x` (atribuição
  aumentada em índice; use `A[i] = A[i] + x`) e nomes que começam com `_`
  (`_aux`; o `_` sozinho passa).
- A `mascara` do lote público vem como **inteiro 0/1** (não booleana) e as
  ocultas vêm como **NaN** em `observada`: converta com `.astype(bool)`
  antes de usar `~mascara`.
