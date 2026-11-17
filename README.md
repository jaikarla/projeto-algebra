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

Ele executa o código com RestrictedPython na configuração mais estrita,
procura nomes proibidos pelo guia, valida a saída (dimensões, NaN, infinito) e
mede o tempo de cada matriz. Antes de entregar, confira também no avaliador
oficial (`pacote_equipe/`, ver o README de lá): é o resultado dele que vale.

## Resultado atual (lote público)

NRMSE médio **0,263** (o mesmo que preencher com a média daria ≈ 1,05), com a
matriz mais lenta em ~0,2 s (limite: 2 s por matriz e 25 s por lote).

| Caso | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| NRMSE | 1,234 | 0,035 | 0,071 | 0,039 | 0,113 | 0,043 | 0,089 | 1,003 | 0,000 | 0,000 |

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

## Entrega (Etapa 1) — até 20/11, 23:59, pelo Classroom

- [ ] `GrupoN.zip` (N = número da equipe na planilha) com `solucao.py` e o relatório
- [ ] uma só pessoa entrega e escreve os nomes da equipe num comentário particular
- [ ] código final testado no avaliador oficial (Docker) sem falhas
- [ ] todos os integrantes conseguem explicar o algoritmo (Etapa 2, em 03/12)
