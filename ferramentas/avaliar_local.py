"""
Avaliador local: roda a função `principal` nos bancos de teste, do jeito mais
parecido possível com o avaliador oficial da monitoria.

  - Compila e executa o arquivo com RestrictedPython (se estiver instalado),
    usando a configuração mais estrita (escrita em índices só em list/dict).
  - Procura no código os nomes proibidos pelo guia (seed, default_rng,
    RandomState, Random e definição de class).
  - Confere a saída (dimensões, só números, sem NaN/inf) e o tempo
    (limite oficial: 2 s por matriz e 25 s por lote).
  - Calcula o NRMSE nas posições ocultas: RMSE / desvio-padrão dos valores
    ocultos verdadeiros.

Uso (a partir da raiz do repositório):
    python ferramentas/avaliar_local.py
    python ferramentas/avaliar_local.py --bancos publico,remascarado,sintetico,laboratorio,extremos
    python ferramentas/avaliar_local.py solucao/solucao.py --detalhes
    python ferramentas/avaliar_local.py --depurar   # mostra falhas escondidas pela rede de segurança
"""
import argparse
import operator
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bancos  # noqa: E402

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIMITE_MATRIZ = 2.0
LIMITE_LOTE = 25.0
MODULOS_PERMITIDOS = ('numpy', 'scipy', 'math', 'random')
PROIBIDOS = [
    (r'\bseed\b', 'uso de seed'),
    (r'\bdefault_rng\b', 'uso de default_rng'),
    (r'\bRandomState\b', 'uso de RandomState'),
    (r'\bRandom\b', 'uso de Random'),
    (r'^\s*class\s', 'definição de class'),
]


def checar_proibidos(codigo):
    problemas = []
    for n, linha in enumerate(codigo.splitlines(), 1):
        sem_comentario = linha.split('#', 1)[0]
        for padrao, descricao in PROIBIDOS:
            if re.search(padrao, sem_comentario):
                problemas.append(f'linha {n}: {descricao}: {linha.strip()}')
    return problemas


def carregar_restrito(codigo, caminho):
    from RestrictedPython import compile_restricted_exec, safe_builtins
    from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
    from RestrictedPython.Guards import (full_write_guard, guarded_iter_unpack_sequence,
                                         guarded_unpack_sequence, safer_getattr)

    resultado = compile_restricted_exec(codigo, filename=caminho)
    if resultado.errors:
        raise SyntaxError('RestrictedPython recusou o código:\n  ' + '\n  '.join(resultado.errors))

    def importar(nome, globais=None, locais=None, fromlist=(), nivel=0):
        if nome.split('.')[0] not in MODULOS_PERMITIDOS:
            raise ImportError(f'import não permitido: {nome}')
        return __import__(nome, globais, locais, fromlist, nivel)

    operadores = {'+=': operator.iadd, '-=': operator.isub, '*=': operator.imul,
                  '/=': operator.itruediv, '//=': operator.ifloordiv, '%=': operator.imod,
                  '**=': operator.ipow, '<<=': operator.ilshift, '>>=': operator.irshift,
                  '&=': operator.iand, '^=': operator.ixor, '|=': operator.ior,
                  '@=': operator.imatmul}

    embutidos = dict(safe_builtins)
    embutidos['__import__'] = importar
    # funções que o guia oficial lista como permitidas (e algumas afins) que
    # não vêm em safe_builtins do RestrictedPython
    for nome in ('enumerate', 'zip', 'map', 'filter', 'min', 'max', 'sum', 'sorted',
                 'any', 'all', 'list', 'tuple', 'dict', 'set', 'frozenset', 'reversed',
                 'iter', 'next'):
        embutidos[nome] = __builtins__[nome] if isinstance(__builtins__, dict) else getattr(__builtins__, nome)
    globais = {
        '__builtins__': embutidos,
        '__name__': 'solucao',
        '__metaclass__': type,
        '_getattr_': safer_getattr,
        '_getitem_': default_guarded_getitem,
        '_getiter_': default_guarded_getiter,
        '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
        '_unpack_sequence_': guarded_unpack_sequence,
        '_write_': full_write_guard,
        '_inplacevar_': lambda op, x, y: operadores[op](x, y),
    }
    exec(resultado.code, globais)
    return globais['principal']


def carregar_normal(codigo, caminho):
    globais = {'__name__': 'solucao'}
    exec(compile(codigo, caminho, 'exec'), globais)
    return globais['principal']


def nrmse(X, g, m):
    h = ~m
    if not h.any():
        return None
    d = float(np.std(g[h]))
    if d == 0:
        d = float(g.max() - g.min()) or 1.0
    return float(np.sqrt(np.mean((X[h] - g[h]) ** 2)) / d)


def validar(saida, forma):
    try:
        X = np.array(saida, dtype=float)
    except (TypeError, ValueError) as e:
        return None, f'saída não numérica ({e})'
    if X.shape != forma:
        return None, f'dimensões erradas: {X.shape} (esperado {forma})'
    if np.isnan(X).any():
        return None, 'saída com NaN'
    if np.isinf(X).any():
        return None, 'saída com infinito'
    return X, None


def referencias(g, m):
    """NRMSE de dois modelos simples, para comparar com a solução:
    média global e efeitos de linha e coluna (mu + a_i + b_j, mínimos
    quadrados em rodadas alternadas)."""
    Oz = np.where(m, g, 0.0)
    mu = Oz[m].mean()
    media = np.full(g.shape, mu)
    a = np.zeros(g.shape[0])
    b = np.zeros(g.shape[1])
    R = np.where(m, Oz - mu, 0.0)
    cl = np.maximum(m.sum(1), 1)
    cc = np.maximum(m.sum(0), 1)
    for _ in range(50):
        a = np.where(m, R - b[None, :], 0.0).sum(1) / cl
        b = np.where(m, R - a[:, None], 0.0).sum(0) / cc
    efeitos = mu + a[:, None] + b[None, :]
    return nrmse(media, g, m), nrmse(efeitos, g, m)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('arquivo', nargs='?', default=os.path.join(RAIZ, 'solucao', 'solucao.py'))
    p.add_argument('--bancos', default='publico', help='lista separada por vírgulas: ' + ','.join(bancos.BANCOS))
    p.add_argument('--sem-restricted', action='store_true',
                   help='executa com Python normal (NÃO garante compatibilidade com o avaliador oficial)')
    p.add_argument('--depurar', action='store_true',
                   help='desliga a "rede de segurança" da solução: qualquer falha interna aparece como FALHA')
    p.add_argument('--detalhes', action='store_true', help='mostra cada matriz')
    args = p.parse_args()

    codigo = open(args.arquivo, encoding='utf-8').read()
    problemas = checar_proibidos(codigo)
    for pr in problemas:
        print('PROIBIDO PELO GUIA ->', pr)

    if args.depurar:
        codigo, n_redes = re.subn(r'except Exception\s*:', 'except ():', codigo)
        # "except ():" não captura nenhuma exceção: o erro original sobe
        print(f'Modo depuração: {n_redes} "rede(s) de segurança" desligada(s)')

    if args.sem_restricted:
        principal = carregar_normal(codigo, args.arquivo)
        print('Execução: Python normal (sem RestrictedPython) -- não garante compatibilidade com o oficial')
    else:
        try:
            principal = carregar_restrito(codigo, args.arquivo)
        except ImportError:
            print('ERRO: RestrictedPython não está instalado, então não dá para conferir a compatibilidade')
            print('com o avaliador oficial. Instale com: pip install -r ferramentas/requisitos.txt')
            print('(ou rode com --sem-restricted, sabendo que isso não testa o ambiente restrito)')
            return 2
        print('Execução: RestrictedPython (configuração estrita)')

    falhas = len(problemas)
    atencoes = 0
    for nome_banco in args.bancos.split(','):
        dados = bancos.BANCOS[nome_banco.strip()]()
        erros, ref_media, ref_efeitos = [], [], []
        t_lote, t_max, lento = 0.0, 0.0, ''
        piores = []
        for nome, g, m in dados:
            observada = np.where(m, g, np.nan)
            mascara = m.astype(np.int64)  # o lote público vem com máscara 0/1 inteira
            t0 = time.perf_counter()
            try:
                saida = principal(observada.copy(), mascara.copy())
                erro_exec = None
            except Exception as e:  # noqa: BLE001
                saida, erro_exec = None, f'{type(e).__name__}: {e}'
            dt = time.perf_counter() - t0
            t_lote += dt
            if dt > t_max:
                t_max, lento = dt, nome
            if erro_exec is None:
                X, erro_exec = validar(saida, g.shape)
            if erro_exec is not None:
                falhas += 1
                print(f'  FALHA {nome}: {erro_exec}')
                continue
            if dt > LIMITE_MATRIZ:
                falhas += 1
                print(f'  FALHA {nome}: {dt:.2f}s (limite {LIMITE_MATRIZ}s por matriz)')
            e = nrmse(X, g, m)
            if e is None:
                if args.detalhes:
                    print(f'  {nome:28s} {str(g.shape):9s} (sem posições ocultas)  {dt:.2f}s')
                continue
            rm, re_ = referencias(g, m)
            erros.append(e)
            ref_media.append(rm)
            ref_efeitos.append(re_)
            # qualidade: pior que o melhor modelo simples (com margem) merece atenção
            if e > 1.05 * min(rm, re_) + 0.01:
                piores.append((nome, e, min(rm, re_)))
            if args.detalhes:
                print(f'  {nome:28s} {str(g.shape):9s} NRMSE {e:7.4f}  '
                      f'(média {rm:.3f}, efeitos {re_:.3f})  {dt:.2f}s')
        aviso = ''
        if nome_banco.strip() == 'publico' and t_lote > LIMITE_LOTE:
            falhas += 1
            aviso = f'  <- passou do limite de {LIMITE_LOTE}s por lote'
        if erros:
            print(f'[{nome_banco}] {len(dados)} matrizes | NRMSE médio {np.mean(erros):.4f} '
                  f'(média global {np.mean(ref_media):.4f}, efeitos {np.mean(ref_efeitos):.4f}) | '
                  f'tempo total {t_lote:.2f}s | mais lenta {t_max:.2f}s ({lento}){aviso}')
        else:
            print(f'[{nome_banco}] {len(dados)} matrizes | sem NRMSE calculável | tempo total {t_lote:.2f}s{aviso}')
        for nome, e, ref in piores:
            atencoes += 1
            print(f'  ATENÇÃO {nome}: NRMSE {e:.4f} pior que o melhor modelo simples ({ref:.4f})')

    if falhas:
        print(f'RESULTADO: {falhas} problema(s) encontrado(s)')
    elif atencoes:
        print(f'RESULTADO: saída válida em todas, mas {atencoes} matriz(es) pior(es) que um modelo simples')
    else:
        print('RESULTADO: OK (saída válida e nunca pior que os modelos simples)')
    return 0 if falhas == 0 else 1


if __name__ == '__main__':
    sys.exit(main())