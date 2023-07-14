from pyomo.environ import *
    
def model_p3o(dados):
    model = ConcreteModel()

    # Conjuntos
    model.K = Set(initialize=dados[0])  # Tipos de ligas (índice k)
    model.I = Set(initialize=dados[1])  # Pedidos ou ordens (índice i)
    model.N = Set(initialize=dados[2])  # Tipos de itens (índice j)
    model.T = Set(initialize=dados[3])  # Períodos de produção (índice t)
    model.L = Set(initialize=dados[4])  # Subperíodos da produção (índice n)
    # Subconjuntos
    model.S = Set(model.I, initialize=dados[5])  # S(i) itens contidos na ordem i
    model.A = Set(model.N, initialize=dados[6])  # A(j) conjunto de ordens que contém o item j
    model.P = Set(model.K, initialize=dados[7])  # L(k) itens a serem produzidos com a liga k

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
    model.NS = Param(T, initialize=dados[8], within=NonNegativeReals)
    # Primeiro subperíodo no período t
    model.f = Param(T, initialize=first_init, within=NonNegativeReals)
    # Último subperíodo no período t
    model.l = Param(T, initialize=last_init, within=NonNegativeReals)
    # Penalidade por atraso do pedido i no período t
    model.bo = Param(I * T, initialize=dados[9], within=NonNegativeReals)
    # Penalidade por entrega em excesso do pedido i no período t
    model.pe = Param(I * T, initialize=dados[10], within=NonNegativeReals)
    # Ganho por entrega adiantada do item j do pedido i no período t
    model.r = Param(N * I * T, initialize=dados[11], within=NonNegativeReals)
    # Ganho por segurar o item j no fim do horizonte de planejamento (período T)
    model.h = Param(N, initialize=dados[12], within=NonNegativeReals)
    # Due date: 1 se prazo de entrega está no período, zero caso contrário
    model.d = Param(I * T, initialize=dados[13], within=Binary)
    # Peso do item j (kg)
    model.p = Param(N, initialize=dados[14], within=NonNegativeReals)
    # Capacidade de uma fornalha (kg)
    model.cap = Param(initialize=dados[15], within=NonNegativeReals)
    # Número máximo de entregas extras da ordem i
    model.me = Param(I, initialize=dados[16], within=NonNegativeReals)
    # Unidades do item j pedidas na ordem i
    model.a = Param(N * I, initialize=dados[17], within=NonNegativeReals)

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
            
    return model
