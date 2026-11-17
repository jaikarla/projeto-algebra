"""
Bancos de matrizes para testar a solução localmente.

Cada banco é uma lista de tuplas (nome, gabarito, mascara), com mascara booleana
(True = conhecida). A observada é montada pelo avaliador: gabarito nas
conhecidas e NaN nas ocultas, como no lote oficial.

  publico      os 10 casos de pacote_equipe/banco_publico (máscaras originais)
  remascarado  os mesmos 10 gabaritos com máscaras novas (mesma fração oculta)
  sintetico    matrizes novas imitando os 10 tipos do lote público, mais 40x40
  laboratorio  as 12 matrizes do laboratório da disciplina (matriz-oculta.html)
  extremos     casos de borda: 1 linha, coluna inteira oculta, constante, etc.
"""
import os

import numpy as np

PASTA_PUBLICO = os.path.join(os.path.dirname(__file__), '..', 'pacote_equipe', 'banco_publico')


def mascara_aleatoria(rng, m, n, frac_oculta, min_lin=3, min_col=3):
    """Esconde ~frac_oculta das entradas, deixando pelo menos min_lin conhecidas
    por linha e min_col por coluna (quando possível)."""
    M = np.ones((m, n), bool)
    alvo = int(round(frac_oculta * m * n))
    feitos = 0
    for p in rng.permutation(m * n):
        if feitos >= alvo:
            break
        i, j = divmod(int(p), n)
        if M[i].sum() <= min_lin or M[:, j].sum() <= min_col:
            continue
        M[i, j] = False
        feitos += 1
    return M


def publico():
    out = []
    for i in range(1, 11):
        g = np.load(os.path.join(PASTA_PUBLICO, f'gabarito_{i:02d}.npy'))
        m = np.load(os.path.join(PASTA_PUBLICO, f'mascara_{i:02d}.npy')).astype(bool)
        out.append((f'pub{i:02d}', g, m))
    return out


def remascarado(rep=6, semente=1):
    rng = np.random.default_rng(semente)
    out = []
    for nome, g, m in publico():
        frac = 1 - m.mean()
        for r in range(rep):
            out.append((f'{nome}m{r}', g, mascara_aleatoria(rng, *g.shape, frac)))
    return out


def sintetico(rep=4, semente=2):
    """Réplicas dos tipos observados no lote público (tamanho, posto, ruído e
    fração oculta parecidos), com sorteios novos, mais três casos maiores."""
    rng = np.random.default_rng(semente)

    def baixo_posto(m, n, r, ruido, deslocamento=0.0, escala=1.0):
        U = rng.normal(size=(m, r))
        V = rng.normal(size=(n, r))
        return deslocamento + escala * (U @ V.T) + ruido * rng.normal(size=(m, n))

    def espectro_linear(t):
        Q1 = np.linalg.qr(rng.normal(size=(t, t)))[0]
        Q2 = np.linalg.qr(rng.normal(size=(t, t)))[0]
        return Q1 @ np.diag(np.linspace(50, 3, t)) @ Q2.T

    tipos = [
        ('t01', lambda: 50 + 1.8 * rng.normal(size=(8, 8)), 0.16),          # constante + ruído
        ('t02', lambda: np.outer(rng.uniform(1, 10, 9), rng.uniform(1, 10, 6)) + 0.6 * rng.normal(size=(9, 6)), 0.13),
        ('t03', lambda: np.outer(rng.uniform(1, 7, 6), rng.uniform(1, 7, 9)) + 0.6 * rng.normal(size=(6, 9)), 0.26),
        ('t04', lambda: baixo_posto(14, 5, 2, 0.1, escala=2.5), 0.13),
        ('t05', lambda: baixo_posto(14, 7, 3, 0.12, escala=2.5), 0.17),
        ('t06', lambda: baixo_posto(16, 16, 3, 0.03, escala=0.6), 0.27),
        ('t07', lambda: baixo_posto(18, 22, 4, 0.06, escala=0.5), 0.26),
        ('t08', lambda: espectro_linear(17), 0.34),                           # posto cheio, sem estrutura
        ('t09', lambda: baixo_posto(24, 19, 3, 0.0, deslocamento=16, escala=0.8), 0.35),
        ('t10', lambda: baixo_posto(28, 31, 5, 0.0, deslocamento=12, escala=0.6), 0.41),
        ('t11', lambda: baixo_posto(40, 40, 5, 0.05, deslocamento=3), 0.45),  # tamanho máximo
        ('t12', lambda: baixo_posto(40, 33, 8, 0.1), 0.40),
        ('t13', lambda: baixo_posto(12, 40, 2, 0.2, deslocamento=5), 0.30),
    ]
    out = []
    for nome, gerar, frac in tipos:
        for r in range(rep):
            g = gerar()
            out.append((f'{nome}s{r}', g, mascara_aleatoria(rng, *g.shape, frac)))
    return out


def laboratorio(frac_oculta=0.3, semente=3):
    """As 12 matrizes do laboratório da disciplina (mesmas regras de geração,
    sorteios feitos aqui em Python)."""
    rng = np.random.default_rng(semente)
    I8, J8 = np.mgrid[0:8, 0:8]
    nivel = rng.uniform(-0.9, 0.9, 8)
    qualidade = rng.uniform(-1.1, 1.1, 6)
    u = rng.uniform(-1, 1, (8, 2))
    v = rng.uniform(-1, 1, (6, 2))
    filmes = np.clip(np.round(3 + nivel[:, None] + qualidade[None, :] + 1.2 * u @ v.T), 1, 5)
    tabuada = (I8 + 1) * (J8 + 1.0)
    soma = rng.integers(0, 6, 8)[:, None] + rng.integers(0, 6, 8)[None, :] + 0.0
    xadrez = np.where((I8 + J8) % 2 == 0, 1.0, -1.0)
    listras = np.repeat(rng.integers(1, 10, 8)[:, None], 10, axis=1) + 0.0
    colunas = np.repeat(rng.integers(1, 10, 10)[None, :], 8, axis=0) + 0.0
    hora = 2 * np.arange(12)
    sensores = np.round((rng.uniform(21, 27, 6)[:, None] + rng.uniform(2, 6, 6)[:, None]
                         * np.sin(2 * np.pi * (hora - 9) / 24)[None, :] + 0.2 * rng.normal(size=(6, 12))) * 10) / 10
    I14, J14 = np.mgrid[0:14, 0:14]
    imagem = ((((J14 >= 3) & (J14 <= 5)) & ((I14 >= 2) & (I14 <= 11)))
              | (((I14 >= 9) & (I14 <= 11)) & ((J14 >= 3) & (J14 <= 11)))).astype(float)
    G = np.array([[8, 2, 5], [2, 8, 5], [6, 6, 1]], float)
    grupos = G[np.arange(9)[:, None] // 3, np.arange(9)[None, :] // 3]
    ruido = np.round((rng.normal(size=(10, 2)) @ rng.normal(size=(10, 2)).T + 0.15 * rng.normal(size=(10, 10))) * 10) / 10
    aleatoria = np.round(rng.normal(size=(10, 10)) * 10) / 10

    mats = [('filmes', filmes), ('tabuada', tabuada), ('soma', soma), ('xadrez', xadrez),
            ('listras', listras), ('colunas', colunas), ('sensores', sensores), ('imagem', imagem),
            ('grupos', grupos), ('ruido', ruido), ('aleatoria', aleatoria)]
    out = [(f'lab_{nome}', g, mascara_aleatoria(rng, *g.shape, frac_oculta, 1, 1)) for nome, g in mats]
    M = mascara_aleatoria(rng, 8, 8, frac_oculta - 1 / 8, 1, 1)
    M[:, 5] = False  # coluna inteira oculta
    out.append(('lab_coluna_oculta', tabuada, M))
    return out


def extremos(semente=4):
    """Casos de borda: o objetivo é não quebrar (shape certo, sem NaN/inf, rápido)."""
    rng = np.random.default_rng(semente)
    out = []
    g = rng.normal(size=(1, 12))
    out.append(('ext_1_linha', g, mascara_aleatoria(rng, 1, 12, 0.3, 0, 0)))
    g = rng.normal(size=(15, 1))
    out.append(('ext_1_coluna', g, mascara_aleatoria(rng, 15, 1, 0.3, 0, 0)))
    g = rng.normal(size=(2, 2))
    out.append(('ext_2x2', g, np.array([[True, True], [True, False]])))
    g = np.full((10, 10), 7.0)
    out.append(('ext_constante', g, mascara_aleatoria(rng, 10, 10, 0.3)))
    g = rng.normal(size=(12, 9))
    M = mascara_aleatoria(rng, 12, 9, 0.2)
    M[4, :] = False
    out.append(('ext_linha_oculta', g, M))
    g = 1e6 * (rng.normal(size=(20, 3)) @ rng.normal(size=(3, 20)))
    out.append(('ext_valores_enormes', g, mascara_aleatoria(rng, 20, 20, 0.4)))
    g = 1e-6 * (rng.normal(size=(20, 3)) @ rng.normal(size=(3, 20)))
    out.append(('ext_valores_minusculos', g, mascara_aleatoria(rng, 20, 20, 0.4)))
    g = rng.normal(size=(40, 40)) @ rng.normal(size=(40, 40))
    out.append(('ext_40x40_80pct_oculto', g, mascara_aleatoria(rng, 40, 40, 0.8, 1, 1)))
    g = rng.normal(size=(40, 15)) @ rng.normal(size=(15, 40))
    out.append(('ext_40x40_posto15', g, mascara_aleatoria(rng, 40, 40, 0.3)))
    g = rng.normal(size=(40, 40))
    out.append(('ext_40x40_ruido', g, mascara_aleatoria(rng, 40, 40, 0.3)))
    g = rng.normal(size=(6, 6))
    out.append(('ext_quase_nada_conhecido', g, mascara_aleatoria(rng, 6, 6, 0.9, 0, 0)))
    g = rng.normal(size=(9, 7))
    out.append(('ext_nada_oculto', g, np.ones((9, 7), bool)))
    return out


BANCOS = {
    'publico': publico,
    'remascarado': remascarado,
    'sintetico': sintetico,
    'laboratorio': laboratorio,
    'extremos': extremos,
}
