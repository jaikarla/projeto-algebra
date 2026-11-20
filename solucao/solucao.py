"""
Torneio de Álgebra Linear (CIN0138) — reconstrução de matrizes com entradas ocultas.

Ideia geral
-----------
Muitas matrizes reais são (aproximadamente) de POSTO BAIXO: cada linha é uma
combinação de poucas "linhas-base". Se a matriz completa X tem posto k, dá para
escrevê-la como X = U V^T, com U (m x k) e V (n x k). As entradas conhecidas
bastam para estimar U e V, e o produto U V^T preenche as ocultas.

O algoritmo:
  1. Monta candidatos: média global, efeitos de linha/coluna e, para cada posto
     k, uma fatoração U V^T ajustada por mínimos quadrados alternados (ALS)
     com regularização (ridge).
  2. Escolhe o melhor candidato por VALIDAÇÃO CRUZADA: esconde de propósito
     parte das entradas conhecidas, tenta prevê-las e mede o erro.
  3. Ajusta o candidato escolhido com todas as entradas conhecidas e devolve
     a matriz completa (as entradas conhecidas não mudam).

Regras do avaliador respeitadas: só NumPy; nenhuma semente aleatória (o
algoritmo é determinístico); nenhuma classe; funções auxiliares dentro de
`principal`; nenhuma atribuição aumentada em índice (A[i] += x), que o
RestrictedPython proíbe.
"""
import numpy as np


def principal(observada, mascara):
    O = np.array(observada, dtype=float)
    m, n = O.shape
    M = np.array(mascara).astype(bool)
    if M.shape != O.shape:
        M = np.isfinite(O)
    M = M & np.isfinite(O)  # nunca confiar numa posição "conhecida" que veio NaN/inf

    total = int(M.sum())
    if total == 0:
        return np.zeros((m, n))
    Oz = np.where(M, O, 0.0)  # conhecidas com o valor real, ocultas com 0
    mu_total = float(Oz[M].mean())

    # ------------------------------------------------------------------
    # Modelos de referência
    # ------------------------------------------------------------------
    def media(Mk):
        """Todas as ocultas recebem a média das conhecidas."""
        return np.full((m, n), Oz[Mk].mean())

    def efeitos(Mk, rodadas=40):
        """x_ij ~ mu + a_i + b_j (efeito da linha + efeito da coluna),
        ajustado por mínimos quadrados em rodadas alternadas."""
        mu = Oz[Mk].mean()
        a = np.zeros(m)
        b = np.zeros(n)
        R = np.where(Mk, Oz - mu, 0.0)
        cont_l = np.maximum(Mk.sum(1), 1)
        cont_c = np.maximum(Mk.sum(0), 1)
        for r in range(rodadas):
            a = np.where(Mk, R - b[None, :], 0.0).sum(1) / cont_l
            b = np.where(Mk, R - a[:, None], 0.0).sum(0) / cont_c
        return mu + a[:, None] + b[None, :]

    # coordenadas de cada célula normalizadas para [-1, 1]
    if m > 1:
        cx = 2.0 * np.arange(m) / (m - 1) - 1.0
    else:
        cx = np.zeros(1)
    if n > 1:
        cy = 2.0 * np.arange(n) / (n - 1) - 1.0
    else:
        cy = np.zeros(1)
    X_coord = np.repeat(cx[:, None], n, axis=1)
    Y_coord = np.repeat(cy[None, :], m, axis=0)

    def polinomio(Mk, grau):
        """x_ij ~ polinômio de grau `grau` nas coordenadas (linha, coluna),
        com todos os termos x^a y^b, a + b <= grau, ajustado por mínimos
        quadrados (equações normais com uma regularização mínima).
        Captura matrizes "suaves" na posição, como a tabuada (x_ij = i*j)."""
        termos = [(a, grau_y) for a in range(grau + 1) for grau_y in range(grau + 1 - a)]
        F = np.stack([X_coord ** t[0] * Y_coord ** t[1] for t in termos], axis=-1)
        A = F[Mk]
        G = A.T @ A + 1e-7 * np.eye(len(termos))
        coef = np.linalg.solve(G, A.T @ Oz[Mk])
        return F @ coef

    # ------------------------------------------------------------------
    # Fatoração de posto k por mínimos quadrados alternados (ALS)
    # ------------------------------------------------------------------
    def als(Mk, k, lam, preenchida, iters=30, tol=1e-5):
        """Minimiza  soma_{(i,j) conhecidas} (x_ij - u_i . v_j)^2 + lam (|U|^2 + |V|^2).

        Com V fixo, cada linha u_i é a solução de um pequeno sistema de
        mínimos quadrados (equações normais com ridge):
            (V_i^T V_i + lam I) u_i = V_i^T x_i
        onde V_i tem só as linhas de V das colunas conhecidas na linha i.
        Depois troca os papéis (U fixo, resolve cada v_j) e repete.
        Todas as linhas são resolvidas de uma vez (np.linalg.solve em lote).

        O ponto de partida vem da SVD truncada de `preenchida` (a matriz com as
        ocultas preenchidas por um palpite), o que evita mínimos locais ruins.
        """
        W = Mk.astype(float)
        R = np.where(Mk, Oz, 0.0)
        Uu, s, Vt = np.linalg.svd(preenchida, full_matrices=False)
        k = min(k, len(s))
        U = Uu[:, :k] * np.sqrt(s[:k])
        V = Vt[:k].T * np.sqrt(s[:k])
        I = (lam + 1e-12) * np.eye(k)
        anterior = None
        for it in range(iters):
            # matrizes V_i^T V_i de todas as linhas: soma_j W_ij v_j v_j^T
            VV = (V[:, :, None] * V[:, None, :]).reshape(n, k * k)
            A = (W @ VV).reshape(m, k, k) + I
            U = np.linalg.solve(A, (R @ V)[:, :, None])[:, :, 0]
            UU = (U[:, :, None] * U[:, None, :]).reshape(m, k * k)
            A = (W.T @ UU).reshape(n, k, k) + I
            V = np.linalg.solve(A, (R.T @ U)[:, :, None])[:, :, 0]
            if it % 5 == 4:  # parada antecipada quando a solução estabiliza
                L = U @ V.T
                if anterior is not None:
                    mudanca = np.sqrt(np.sum((L - anterior) ** 2))
                    if mudanca < tol * (np.sqrt(np.sum(L * L)) + 1e-12):
                        break
                anterior = L
        return U @ V.T

    def estimar():
        # --------------------------------------------------------------
        # Grupos da validação cruzada (determinísticos, sem semente):
        # as posições conhecidas são "embaralhadas" pela razão áurea e
        # divididas em K grupos. Cada grupo retira no máximo ~1/K das
        # conhecidas de cada linha e coluna e nunca deixa uma linha ou
        # coluna com menos de 3 conhecidas, para imitar a situação real.
        # --------------------------------------------------------------
        idx = np.argwhere(M)
        q = len(idx)
        ordem = np.argsort((np.arange(q) * 0.6180339887498949) % 1.0)
        K = 10
        n_grupos = int(min(K, max(4, np.ceil(200.0 / max(q / K, 1.0)))))
        cont_l = M.sum(1)
        cont_c = M.sum(0)
        lim_l = np.maximum(1, np.ceil(cont_l / K)).astype(int)
        lim_c = np.maximum(1, np.ceil(cont_c / K)).astype(int)
        grupos = []
        for f in range(n_grupos):
            tira_l = [0] * m  # listas Python: escrever nelas é sempre permitido
            tira_c = [0] * n
            escolhidos = []
            for t in range(f, q, K):
                i = int(idx[ordem[t], 0])
                j = int(idx[ordem[t], 1])
                if cont_l[i] - tira_l[i] <= 3 or cont_c[j] - tira_c[j] <= 3:
                    continue
                if tira_l[i] >= lim_l[i] or tira_c[j] >= lim_c[j]:
                    continue
                tira_l[i] = tira_l[i] + 1
                tira_c[j] = tira_c[j] + 1
                escolhidos.append((i, j))
            if len(escolhidos) == 0:
                continue
            te = np.array(escolhidos)
            # máscara do grupo sem escrever em posições do array:
            # bincount marca as posições retiradas
            retiradas = np.bincount(te[:, 0] * n + te[:, 1], minlength=m * n).reshape(m, n) > 0
            Mk = M & ~retiradas
            grupos.append((te, Mk))

        if len(grupos) == 0:  # matriz pequena demais para validar
            return efeitos(M), 'efeitos'

        def erro(X, te):
            d = X[te[:, 0], te[:, 1]] - O[te[:, 0], te[:, 1]]
            e = float(np.sum(d * d))
            if not np.isfinite(e):
                return np.inf
            return e

        # Valores singulares da matriz preenchida pela média: dão a escala da
        # regularização (lam = c * s_k, proporcional à k-ésima componente).
        # Cada grupo da validação usa os SEUS valores singulares, calculados só
        # com as entradas de treino do grupo, para que as entradas retiradas
        # não influenciem o ajuste (sem vazamento de informação).
        sv = np.linalg.svd(np.where(M, Oz, mu_total), compute_uv=False)
        sv_grupo = [np.linalg.svd(np.where(Mk, Oz, Oz[Mk].mean()), compute_uv=False)
                    for te, Mk in grupos]

        erros = {}
        erros['media'] = sum(erro(media(Mk), te) for te, Mk in grupos)
        erros['efeitos'] = sum(erro(efeitos(Mk), te) for te, Mk in grupos)
        referencia = min(erros['media'], erros['efeitos'])

        # polinômios na posição, graus 1 a 4 (só se houver dados para os termos)
        menor_treino = min(int(Mk.sum()) for te, Mk in grupos)
        # Ele só entra na disputa se ganhar com folga da média e dos efeitos:
        # em matrizes de puro ruído, algum grau sempre parece um pouco melhor
        # por acaso (erro ~0,85-0,97 da referência), e escolhê-lo piora o resultado.
        for grau in range(1, 5):
            if (grau + 1) * (grau + 2) // 2 < menor_treino:
                e = sum(erro(polinomio(Mk, grau), te) for te, Mk in grupos)
                if e < 0.8 * referencia:
                    erros[('poli', grau)] = e

        # --------------------------------------------------------------
        # 1) Escolha do posto k. "Continuação": o ajuste do posto k parte
        #    do ajuste do posto k-1, o que deixa o ALS mais estável.
        # --------------------------------------------------------------
        kmax = int(max(1, min(min(m, n) - 1, 15)))
        c_posto = 1e-3
        preench = [np.where(Mk, Oz, Oz[Mk].mean()) for te, Mk in grupos]
        partida = {}
        melhor_k = 1
        melhor_e = np.inf
        for k in range(1, kmax + 1):
            e = 0.0
            novos = []
            for f in range(len(grupos)):
                te, Mk = grupos[f]
                L = als(Mk, k, c_posto * sv_grupo[f][k - 1], preench[f])
                e = e + erro(L, te)
                novos.append(np.where(Mk, Oz, L))
            partida[k] = preench
            preench = novos
            erros[('als', k, c_posto)] = e
            if e < melhor_e:
                melhor_e = e
                melhor_k = k
            elif k >= melhor_k + 4 and k >= 8:
                break  # o erro parou de cair há 4 postos
            if k >= 4 and melhor_e > 1.25 * referencia:
                break  # nenhum posto baixo ajuda: matriz sem estrutura

        # --------------------------------------------------------------
        # 2) Intensidade da regularização no posto escolhido
        # --------------------------------------------------------------
        if melhor_e <= 1.25 * referencia:
            for c in (1e-4, 3e-3, 1e-2, 3e-2):
                e = 0.0
                for f in range(len(grupos)):
                    te, Mk = grupos[f]
                    lam = c * sv_grupo[f][melhor_k - 1]
                    e = e + erro(als(Mk, melhor_k, lam, partida[melhor_k][f]), te)
                erros[('als', melhor_k, c)] = e

        # --------------------------------------------------------------
        # 3) Ajuste final do vencedor com todas as entradas conhecidas
        # --------------------------------------------------------------
        escolha = min(erros, key=erros.get)
        if escolha == 'media':
            return media(M), 'media'
        if escolha == 'efeitos':
            return efeitos(M), 'efeitos'
        if escolha[0] == 'poli':
            return polinomio(M, escolha[1]), 'poli'
        k_final = escolha[1]
        lam_final = escolha[2] * sv[k_final - 1]
        # dois pontos de partida (continuação pelos postos e média simples);
        # fica o que melhor se ajusta às entradas conhecidas
        P = np.where(M, Oz, mu_total)
        for k in range(1, k_final):
            P = np.where(M, Oz, als(M, k, c_posto * sv[k - 1], P))
        opcoes = [als(M, k_final, lam_final, P, iters=300),
                  als(M, k_final, lam_final, np.where(M, Oz, mu_total), iters=300)]
        custos = [float(np.sum((L - Oz)[M] ** 2)) for L in opcoes]
        return opcoes[int(np.argmin(custos))], 'als'

    # Rede de segurança: se algo inesperado falhar, cai para modelos simples,
    # porque um erro de execução desclassifica a equipe. As linhas marcadas
    # com "rede de segurança" são desligadas pelo avaliador local na opção
    # --depurar, para mostrar qualquer falha escondida pelo recuo.
    try:
        X, modelo = estimar()
    except Exception:  # rede de segurança
        modelo = 'efeitos'
        try:
            X = efeitos(M)
        except Exception:  # rede de segurança
            X = np.full((m, n), mu_total)

    # Linha ou coluna sem NENHUMA entrada conhecida: a fatoração não tem como
    # estimá-la (preveria ~0). Ali o melhor palpite é o modelo de efeitos
    # (média + efeito da linha ou da coluna que ainda tem dados).
    # O polinômio e os modelos simples já preveem essas posições sozinhos.
    vazia_l = M.sum(1) == 0
    vazia_c = M.sum(0) == 0
    if modelo == 'als' and (vazia_l.any() or vazia_c.any()):
        try:
            X = np.where(vazia_l[:, None] | vazia_c[None, :], efeitos(M), X)
        except Exception:  # rede de segurança
            X = np.where(vazia_l[:, None] | vazia_c[None, :], mu_total, X)

    # só corta valores absurdamente fora da faixa observada
    lo = float(Oz[M].min())
    hi = float(Oz[M].max())
    folga = 0.5 * (hi - lo) + 1e-9
    X = np.clip(X, lo - folga, hi + folga)
    X = np.where(np.isfinite(X), X, mu_total)
    return np.where(M, O, X)
