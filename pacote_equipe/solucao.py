def principal(observada, mascara):
    numero_linhas = len(observada)
    if numero_linhas == 0:
        return []

    numero_colunas = len(observada[0])
    resultado = []
    conhecidos = []

    for i in range(numero_linhas):
        linha = []
        for j in range(numero_colunas):
            if mascara[i][j]:
                valor = float(observada[i][j])
                linha.append(valor)
                conhecidos.append((i, j, valor))
            else:
                linha.append(0.0)
        resultado.append(linha)

    if not conhecidos:
        return resultado

    def calcular_media(valores):
        soma = 0.0
        for item in valores:
            soma = soma + item[2]
        return soma / len(valores)

    media_global = calcular_media(conhecidos)
    if len(conhecidos) < 3:
        for i in range(numero_linhas):
            for j in range(numero_colunas):
                if not mascara[i][j]:
                    resultado[i][j] = media_global
        return resultado

    def resolver(sistema, valores):
        tamanho = len(valores)
        aumentado = []
        for i in range(tamanho):
            linha = sistema[i][:]
            linha.append(valores[i])
            aumentado.append(linha)

        for coluna in range(tamanho):
            pivo = coluna
            for candidata in range(coluna + 1, tamanho):
                if abs(aumentado[candidata][coluna]) > abs(aumentado[pivo][coluna]):
                    pivo = candidata

            if abs(aumentado[pivo][coluna]) < 1e-12:
                continue

            aumentado[coluna], aumentado[pivo] = aumentado[pivo], aumentado[coluna]
            valor_pivo = aumentado[coluna][coluna]
            for j in range(coluna, tamanho + 1):
                aumentado[coluna][j] = aumentado[coluna][j] / valor_pivo

            for i in range(tamanho):
                if i != coluna:
                    fator = aumentado[i][coluna]
                    for j in range(coluna, tamanho + 1):
                        aumentado[i][j] = aumentado[i][j] - fator * aumentado[coluna][j]

        return [aumentado[i][tamanho] for i in range(tamanho)]

    def ajustar_polinomio(treino, grau):
        termos = []
        for a in range(grau + 1):
            for b in range(grau + 1 - a):
                termos.append((a, b))

        tamanho = len(termos)
        sistema = [[0.0] * tamanho for _ in range(tamanho)]
        valores = [0.0] * tamanho

        for i, j, valor in treino:
            x = 0.0 if numero_linhas < 2 else 2.0 * i / (numero_linhas - 1) - 1.0
            y = 0.0 if numero_colunas < 2 else 2.0 * j / (numero_colunas - 1) - 1.0
            caracteristicas = [x ** a * y ** b for a, b in termos]
            for a in range(tamanho):
                valores[a] = valores[a] + caracteristicas[a] * valor
                for b in range(tamanho):
                    sistema[a][b] = sistema[a][b] + caracteristicas[a] * caracteristicas[b]

        for i in range(tamanho):
            sistema[i][i] = sistema[i][i] + 1e-7

        coeficientes = resolver(sistema, valores)
        return termos, coeficientes

    def ajustar_aditivo(treino):
        tamanho = 1 + numero_linhas + numero_colunas
        sistema = [[0.0] * tamanho for _ in range(tamanho)]
        valores = [0.0] * tamanho

        for i, j, valor in treino:
            indices = (0, 1 + i, 1 + numero_linhas + j)
            for a in indices:
                valores[a] = valores[a] + valor
                for b in indices:
                    sistema[a][b] = sistema[a][b] + 1.0

        sistema[0][0] = sistema[0][0] + 1e-8
        for i in range(1, tamanho):
            sistema[i][i] = sistema[i][i] + 0.05

        return resolver(sistema, valores)

    def prever(modelo, treino, consultas):
        tipo = modelo[0]
        parametro = modelo[1]

        if tipo == "media":
            media = calcular_media(treino)
            return [media] * len(consultas)

        if tipo == "polinomio":
            termos, coeficientes = ajustar_polinomio(treino, parametro)
            previsoes = []
            for i, j in consultas:
                x = 0.0 if numero_linhas < 2 else 2.0 * i / (numero_linhas - 1) - 1.0
                y = 0.0 if numero_colunas < 2 else 2.0 * j / (numero_colunas - 1) - 1.0
                previsao = 0.0
                for k in range(len(termos)):
                    a, b = termos[k]
                    previsao = previsao + coeficientes[k] * x ** a * y ** b
                previsoes.append(previsao)
            return previsoes

        if tipo == "aditivo":
            coeficientes = ajustar_aditivo(treino)
            return [
                coeficientes[0] + coeficientes[1 + i] + coeficientes[1 + numero_linhas + j]
                for i, j in consultas
            ]

        largura = parametro
        pesos_por_distancia = modelo[3]
        previsoes = []
        for i, j in consultas:
            soma_pesos = 0.0
            soma_valores = 0.0
            for linha, coluna, valor in treino:
                distancia = (i - linha) ** 2 + (j - coluna) ** 2
                peso = pesos_por_distancia[distancia]
                soma_pesos = soma_pesos + peso
                soma_valores = soma_valores + peso * valor
            if soma_pesos > 1e-300:
                previsoes.append(soma_valores / soma_pesos)
            else:
                previsoes.append(calcular_media(treino))
        return previsoes

    numero_folds = 5
    if len(conhecidos) < numero_folds:
        numero_folds = len(conhecidos)
    coordenadas = [(item[0], item[1]) for item in conhecidos]

    def chave_coordenada(posicao):
        i, j = posicao
        return (
            ((i + 1) * 73856093)
            ^ ((j + 1) * 19349663)
            ^ ((i + 1) * (j + 1) * 83492791)
        ) % 1000000007

    coordenadas = sorted(coordenadas, key=chave_coordenada)
    fold_por_celula = [[-1] * numero_colunas for _ in range(numero_linhas)]
    for indice in range(len(coordenadas)):
        i, j = coordenadas[indice]
        fold_por_celula[i][j] = indice % numero_folds

    menor_treino = len(conhecidos) - (len(conhecidos) + numero_folds - 1) // numero_folds
    modelos = [("media", 0, 0)]
    if 1 + numero_linhas + numero_colunas <= menor_treino:
        modelos.append(("aditivo", 0, 3))
    for grau in range(1, 5):
        quantidade_termos = (grau + 1) * (grau + 2) // 2
        if quantidade_termos <= menor_treino:
            modelos.append(("polinomio", grau, grau))
    distancia_maxima = (numero_linhas - 1) ** 2 + (numero_colunas - 1) ** 2
    for largura in (0.65, 0.9, 1.2, 1.6, 2.2, 3.0, 4.0, 6.0):
        pesos_por_distancia = []
        for distancia in range(distancia_maxima + 1):
            peso = 2.718281828459045 ** (-distancia / (2.0 * largura * largura))
            pesos_por_distancia.append(peso)
        modelos.append(("kernel", largura, 10, pesos_por_distancia))

    erros = [0.0] * len(modelos)
    total_validacao = 0
    for fold in range(numero_folds):
        treino = []
        validacao = []
        for i, j, valor in conhecidos:
            if fold_por_celula[i][j] == fold:
                validacao.append((i, j, valor))
            else:
                treino.append((i, j, valor))

        consultas = [(item[0], item[1]) for item in validacao]
        total_validacao = total_validacao + len(validacao)
        for indice in range(len(modelos)):
            modelo = modelos[indice]
            previsoes = prever(modelo, treino, consultas)
            for previsao, item in zip(previsoes, validacao):
                diferenca = previsao - item[2]
                erros[indice] = erros[indice] + diferenca * diferenca

    melhor_indice = 0
    melhor_erro = erros[0] / total_validacao
    for indice in range(1, len(modelos)):
        erro = erros[indice] / total_validacao
        if erro < melhor_erro:
            melhor_indice = indice
            melhor_erro = erro

    limite = melhor_erro * 1.02
    for indice in range(len(modelos)):
        modelo = modelos[indice]
        erro = erros[indice] / total_validacao
        if erro <= limite and modelo[2] < modelos[melhor_indice][2]:
            melhor_indice = indice

    consultas_finais = []
    for i in range(numero_linhas):
        for j in range(numero_colunas):
            if not mascara[i][j]:
                consultas_finais.append((i, j))

    previsoes_finais = prever(
        modelos[melhor_indice],
        conhecidos,
        consultas_finais
    )
    for (i, j), valor in zip(consultas_finais, previsoes_finais):
        resultado[i][j] = float(valor)

    return resultado