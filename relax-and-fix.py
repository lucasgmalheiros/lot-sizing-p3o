from pyomo.environ import *
from pyomo.opt import SolverFactory
import ast
import numpy as np
import os
import csv
import time

# Melhores resultados das instâncias
benchmark = [55.95, 55.95, 67.39, 81.43, 106.24, 120.35,
             165.93, 179.09, 174.5, 178.71, 165.14,
             48.69, 56.3, 67.65, 68.02, 104.4, 125.73, 84.85,
             167.64, 154.93, 143.35, 170.52, 36.96, 37.42, 
             43.36, 53.22, 67.61, 82.77, 85.24, 95.37, 124.49,
             130.7, 110.58]

# Arquivo csv para armazenar resultados
with open("results.csv", "w", encoding="UTF8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Instance", "Result", "LB", "Error", "Time"])

    # Loop das instâncias
    instancias = [i for i in os.listdir('INSTANCIAS 4P2') if i[-3:] == "dat"]
    for index, instancia in enumerate(instancias):
        path = f"INSTANCIAS 4P2/{instancias[index]}"
        print(f"Resolvendo {instancia.replace('.dat', '')}")

        # ------------------ Leitura dos dados ------------------ #
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

        # ------------------ Modelo matemático ------------------ #
        model = ConcreteModel()
        
        # Conjuntos
        model.K = Set(initialize=Ligas)  # Tipos de ligas (índice k)
        model.I = Set(initialize=Pedidos)  # Pedidos ou ordens (índice i)
        model.N = Set(initialize=Itens)  # Tipos de itens (índice j)
        model.T = Set(initialize=Periodos)  # Períodos de produção (índice t)
        model.L = Set(initialize=Subperiodos)  # Subperíodos da produção (índice n)
        # Subconjuntos
        model.S = Set(Pedidos, initialize=sub_S)  # S(i) itens contidos na ordem i
        model.A = Set(Itens, initialize=sub_A)  # A(j) conjunto de ordens que contém o item j
        model.P = Set(Ligas, initialize=sub_L)  # L(k) itens a serem produzidos com a liga k

        K, I, N, T, L, S, A, P = model.K, model.I, model.N, model.T, model.L, model.S, model.A, model.P
        
        # Parâmetros
        # Primeiro subperíodo no período t
        def first_init(model, t):
            if t == 1:
                return 1
            else:
                return 1 + sum(model.NS[r] for r in range(1, t))
            
        # Último subperíodo no período t
        def last_init(model, t):
            return sum(model.NS[r] for r in range(1, t + 1))
        
        # Número de subperíodos por período t
        model.NS = Param(T, initialize=numero_subperiodos, within=NonNegativeReals)
        # Primeiro subperíodo no período t
        model.f = Param(T, initialize=first_init, within=NonNegativeReals)
        # Último subperíodo no período t
        model.l = Param(T, initialize=last_init, within=NonNegativeReals)
        # Penalidade por atraso do pedido i no período t
        model.bo = Param(I * T, initialize=penalidade_atraso, within=NonNegativeReals)
        # Penalidade por entrega em excesso do pedido i no período t
        model.pe = Param(I * T, initialize=penalidade_entrega_excesso, within=NonNegativeReals)
        # Ganho por entrega adiantada do item j do pedido i no período t
        model.r = Param(N * I * T, initialize=beneficio_entrega_adiantada, within=NonNegativeReals)
        # Ganho por segurar o item j no fim do horizonte de planejamento (período T)
        model.h = Param(N, initialize=beneficio_ultimo_estoque, within=NonNegativeReals)
        # Due date: 1 se prazo de entrega está no período, zero caso contrário
        model.d = Param(I * T, initialize=due_date, within=Binary)
        # Peso do item j (kg)
        model.p = Param(N, initialize=peso_item, within=NonNegativeReals)
        # Capacidade de uma fornalha (kg)
        model.cap = Param(initialize=capacidade_fornalha, within=NonNegativeReals)
        # Número máximo de entregas extras da ordem i
        model.me = Param(I, initialize=maximo_extra_delivery, within=NonNegativeReals)
        # Unidades do item j pedidas na ordem i
        model.a = Param(N * I, initialize=demanda, within=NonNegativeReals)

        NS, f, l, bo, pe, r, h, d, p, cap, me, a = (model.NS, model.f, model.l, model.bo, model.pe, model.r,
                                                    model.h, model.d, model.p, model.cap, model.me,model.a)

        # Variáveis de decisão
        # Recebe 1 caso haja atraso da ordem i no período t
        model.BO = Var(I * T, within=Binary)
        # Recebe 1 em caso de entrega extra de i no período t
        model.E = Var(I * T, within=Binary)
        # Quantidade do item j da ordem i entregue no período t em uma entrega extra
        model.G = Var(N * I * T, within=NonNegativeReals)
        # Número de itens j estocados ao fim do período t
        model.Q = Var(N * T, within=NonNegativeReals)
        # Recebe 1 se a ordem i é concluída no período t
        model.XO = Var(I * T, within=Binary)
        # Número de itens j produzidos no subperíodo n
        model.X = Var(N * L, within=NonNegativeReals) # Relaxação
        # Recebe 1 caso a fornalha seja utilizada para produzir a liga k no subperíodo n
        model.Y = Var(K * L, within=NonNegativeReals, bounds=(0, 1)) # Relaxação
        # Quantidade de itens j do pedido i entregues no período t
        model.W = Var(N * I * T, within=NonNegativeIntegers)

        BO, E, G, Q, XO, X, Y, W = model.BO, model.E, model.G, model.Q, model.XO, model.X, model.Y, model.W

        # Objetivo
        # Minimizar custos
        model.obj = Objective(sense=minimize,
                            expr=sum(sum(bo[i, t] * BO[i, t] + pe[i, t] * E[i, t] for t in T) for i in I) -
                                sum(sum(r[j, i, t] * G[j, i, t] for i in I for j in S[i]) for t in T) -
                                sum(h[j] * Q[j, max(T)] for j in N))

        Z = model.obj
        
        # Restrições
        # Define quando a ordem é completada (2.2)
        model.r_complete = ConstraintList()
        for i in I:
            for t in T:
                if t == 1:
                    restr = XO[i, t] + BO[i, t] == d[i, t]
                    model.r_complete.add(expr=restr)
                else:
                    restr = XO[i, t] + BO[i, t] == d[i, t] + BO[i, t - 1]
                    model.r_complete.add(expr=restr)
                    
        # Balanço de estoques (2.3)
        model.r_balanco = ConstraintList()
        for j in N:
            for t in T:
                if t == 1:
                    restr = (sum(X[j, n] for n in range(f[t], l[t] + 1)) ==
                            Q[j, t] + sum(W[j, i, t] for i in A[j]))
                    model.r_balanco.add(expr=restr)
                else:
                    restr = (Q[j, t - 1] + sum(X[j, n] for n in range(f[t], l[t] + 1)) == 
                            Q[j, t] + sum(W[j, i, t] for i in A[j]))
                    model.r_balanco.add(expr=restr)
                    
        # Uso da fornalha no período (2.4)
        model.r_uso_fornalha = ConstraintList()
        for n in L:
            restr = sum(Y[k, n] for k in K) <= 1
            model.r_uso_fornalha.add(expr=restr)
            
        # Capacidade da fornalha (2.5)
        model.r_capacidade_fornalha = ConstraintList()
        for k in K:
            for n in L:
                restr = sum(p[j] * X[j, n] for j in P[k]) <= cap * Y[k, n]
                model.r_capacidade_fornalha.add(expr=restr)
                
        # Produção limitada à demanda (2.6)
        model.r_limite_producao_demanda = ConstraintList()
        for j in N:
            restr = sum(X[j, n] for n in L) <= sum(a[j, i] for i in I)
            model.r_limite_producao_demanda.add(expr=restr)
            
        # Entrega só ocorre quando a ordem é concluída ou extra delivery (2.7)
        model.r_entregas = ConstraintList()
        for i in I:
            for t in T:
                for j in S[i]:
                    restr = W[j, i , t] <= a[j, i] * (XO[i, t] + E[i, t])
                    model.r_entregas.add(expr=restr)
                    
        # Entregas limitadas à demanda (2.8)
        model.r_limite_entrega_demanda = ConstraintList()
        for j in N:
            for i in A[j]:
                restr = sum(W[j, i, t] for t in T) <= a[j, i]
                model.r_limite_entrega_demanda.add(expr=restr)
                
        # Entregas extras apenas de ordens em aberto no período (2.9)
        model.r_entrega_extra = ConstraintList()
        for i in I:
            for t in T:
                restr = XO[i, t] + E[i, t] <= 1
                model.r_entrega_extra.add(expr=restr)
                
        # Conclusão da ordem quando todos os itens foram entregues (2.10)
        model.r_conclusao_ordem = ConstraintList()
        for t in T:
            for i in I:
                for j in S[i]:
                    restr = sum(W[j, i, l] for l in range(1, t + 1)) >= a[j, i] * XO[i, t]
                    model.r_conclusao_ordem.add(expr=restr)
                    
        # Limite superior de entregas extras (2.11)
        model.r_limite_entrega_extra = ConstraintList()
        for i in I:
            restr = sum(E[i, t] for t in T) <= me[i]
            model.r_limite_entrega_extra.add(expr=restr)
            
        # Limite da quantidade de itens entregues com adiantamento (2.12)
        model.r_limite_entrega_adiantada_1 = ConstraintList()
        model.r_limite_entrega_adiantada_2 = ConstraintList()
        for t in T:
            for i in I:
                for j in S[i]:
                    restr1 = G[j, i, t] <= W[j, i, t]
                    restr2 = G[j, i, t] <= a[j, i] * (1 - XO[i, t])
                    model.r_limite_entrega_adiantada_1.add(expr=restr1)
                    model.r_limite_entrega_adiantada_2.add(expr=restr2)

        # Remoçao de soluções simétricas (2.17)
        model.remove_simetria = ConstraintList()
        for t in T:
            for n in range(f[t], l[t]):
                restr = sum(Y[k, n] for k in K) >= sum(Y[k, n + 1] for k in K)
                model.remove_simetria.add(expr=restr)
                
        
        # ------------------ Heurística Relax-and-Fix ------------------ #
        solver = SolverFactory("gurobi")
        solver.options['TimeLimit'] = 3600
        start = time.time()
        for t in T:
            for n in range(f[t], l[t] + 1): # Subperíodos pertencentes ao período t
                # Variáveis X tornam-se inteiras
                for j in N:
                    X[j, n].domain = NonNegativeIntegers
                # Variáveis Y tornam-se inteiras
                for k in K:
                    Y[k, n].domain = NonNegativeIntegers
            # Resolução inteira no período t
            solver.solve(model, tee=False)
            # Fixar variáveis para próxima iteração
            for n in range(f[t], l[t] + 1):
                for j in N:
                    X[j, n].fix()
                for k in K:
                    Y[k, n].fix()
        end = time.time()
        
        # Escreve linha no arquivo csv
        writer.writerow([instancia.replace(".dat", ""), f"{value(Z) :.2f}", f"{benchmark[index] :.2f}",
                        f"{((value(Z) - benchmark[index]) / benchmark[index]) :.2f}",
                        f"{end - start :.2f}"])
