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
    for nome in ('enumerate', 'map', 'filter', 'min', 'max', 'sum', 'any', 'all',
                 'list', 'dict', 'set', 'frozenset', 'reversed', 'iter', 'next'):
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('arquivo', nargs='?', default=os.path.join(RAIZ, 'solucao', 'solucao.py'))
    p.add_argument('--bancos', default='publico', help='lista separada por vírgulas: ' + ','.join(bancos.BANCOS))
    p.add_argument('--sem-restricted', action='store_true', help='executa com Python normal')
    p.add_argument('--detalhes', action='store_true', help='mostra cada matriz')
    args = p.parse_args()

    codigo = open(args.arquivo, encoding='utf-8').read()
    problemas = checar_proibidos(codigo)
    for pr in problemas:
        print('PROIBIDO PELO GUIA ->', pr)

    if args.sem_restricted:
        principal = carregar_normal(codigo, args.arquivo)
        print('Execução: Python normal (sem RestrictedPython)')
    else:
        try:
            principal = carregar_restrito(codigo, args.arquivo)
            print('Execução: RestrictedPython (configuração estrita)')
        except ImportError:
            print('AVISO: RestrictedPython não instalado (pip install RestrictedPython); usando Python normal.')
            principal = carregar_normal(codigo, args.arquivo)

    falhas = len(problemas)
    for nome_banco in args.bancos.split(','):
        dados = bancos.BANCOS[nome_banco.strip()]()
        erros, t_lote, t_max, lento = [], 0.0, 0.0, ''
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
            if e is not None:
                erros.append(e)
            if args.detalhes:
                print(f'  {nome:28s} {str(g.shape):9s} NRMSE {"-" if e is None else f"{e:.4f}":>7s}  {dt:.2f}s')
        aviso = ''
        if nome_banco.strip() == 'publico' and t_lote > LIMITE_LOTE:
            falhas += 1
            aviso = f'  <- passou do limite de {LIMITE_LOTE}s por lote'
        media = f'{np.mean(erros):.4f}' if erros else '-'
        print(f'[{nome_banco}] {len(dados)} matrizes | NRMSE médio {media} | '
              f'tempo total {t_lote:.2f}s | mais lenta {t_max:.2f}s ({lento}){aviso}')

    print('RESULTADO:', 'OK' if falhas == 0 else f'{falhas} problema(s) encontrado(s)')
    return 0 if falhas == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
