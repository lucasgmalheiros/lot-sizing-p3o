import ast
import numpy as np

def read_instance(path):
    with open(path, "r") as file:
        dados = file.readlines()
        numero_ligas = ast.literal_eval(dados[0])  # Ligas (K)
        numero_pedidos = ast.literal_eval(dados[1])  # Pedidos (I)
        numero_itens = ast.literal_eval(dados[2])  # Itens (J)
        numero_periodos = ast.literal_eval(dados[3])  # Periodos (T)
        numero_subp = ast.literal_eval(dados[4])  # Subperiodos (n)
        numero_capacidade = ast.literal_eval(dados[5])  # Capacidade (cap)
        matriz_custo_atraso = ast.literal_eval(dados[6])  # Custo de atraso de pedidos (BO_it)
        vetor_custo_estoque = ast.literal_eval(dados[7])  # Custo de estoque por item no último período (H_jT)
        vetor_peso_itens = ast.literal_eval(dados[8])  # Peso dos itens (p_j)
        matriz_demanda = ast.literal_eval(dados[9])  # Demanda do item j no pedido i (a_ji)
        matriz_due_date = ast.literal_eval(dados[10])  # Due date (d_it)
        conjunto_itens_pedido = ast.literal_eval(dados[11])  # Itens que pertencem ao pedido i: Set S(i)
        conjunto_ordens_item = ast.literal_eval(dados[12])  # Ordens que contém o item j: Set A(j)
        conjunto_itens_liga = ast.literal_eval(dados[13])  # Itens a serem produzidos com a liga k: Set L(k)
        
    # Ajuste dos dados ao Pyomo
    # Conjuntos
    Ligas = np.arange(1, numero_ligas + 1)  # k
    Pedidos = np.arange(1, numero_pedidos + 1)  # i
    Itens = np.arange(1, numero_itens + 1)  # j
    Periodos = np.arange(1, numero_periodos + 1)  # t
    Subperiodos = np.arange(1, numero_subp * numero_periodos + 1)  # n

    # Subconjuntos
    sub_S = dict(zip(Pedidos, conjunto_itens_pedido))
    sub_A = dict(zip(Itens, conjunto_ordens_item))
    sub_L = dict(zip(Ligas, conjunto_itens_liga))

    # Parâmetros
    numero_subperiodos = numero_subp  # NS
    capacidade_fornalha = numero_capacidade  # cap
    penalidade_atraso = np.array(matriz_custo_atraso).ravel()  # bo_{it}
    beneficio_ultimo_estoque = np.array(vetor_custo_estoque)  # h_j
    peso_item = np.array(vetor_peso_itens)  # p_j
    demanda = np.array(matriz_demanda).ravel()  # a_{ji}
    due_date = np.array(matriz_due_date).ravel()  # d_{it}

    # Tuplas de índices
    i_t = np.array(np.meshgrid(Pedidos, Periodos)).T.reshape(-1, 2)
    i_t = [tuple(i_t[a]) for a in range(len(i_t))]
    j_i = np.array(np.meshgrid(Itens, Pedidos)).T.reshape(-1, 2)
    j_i = [tuple(j_i[a]) for a in range(len(j_i))]

    # Dicionários para leitura
    penalidade_atraso = dict(zip(i_t, penalidade_atraso))
    beneficio_ultimo_estoque = dict(zip(Itens, beneficio_ultimo_estoque))
    peso_item = dict(zip(Itens, peso_item))
    demanda = dict(zip(j_i, demanda))
    due_date = dict(zip(i_t, due_date))

    # Parâmetros do modelo indisponíveis nas instâncias
    penalidade_entrega_excesso = 0
    beneficio_entrega_adiantada = 0
    maximo_extra_delivery = 0

    return (Ligas, Pedidos, Itens, Periodos, Subperiodos, sub_S, sub_A, sub_L,
            numero_subperiodos, penalidade_atraso, penalidade_entrega_excesso,
            beneficio_entrega_adiantada, beneficio_ultimo_estoque, due_date,
            peso_item, capacidade_fornalha, maximo_extra_delivery, demanda)
