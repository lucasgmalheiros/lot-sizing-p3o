from pyomo.environ import *
from pyomo.opt import SolverFactory
from read_data import read_instance
from linear_models import model_p3o
import os
import csv
import time

# Escolha do solver
solver = SolverFactory("gurobi")
solver.options["TimeLimit"] = 3600

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
    instancias = [i for i in os.listdir("INSTANCIAS 4P2") if i[-4:] == ".dat"]
    for index, instancia in enumerate(instancias):
        path = f"INSTANCIAS 4P2/{instancias[index]}"
        print(f"Resolvendo {instancia.replace('.dat', '')}")

        # ------------------ Leitura dos dados da instância ------------------ #
        dados = read_instance(path)

        # ------------------ Modelo matemático ------------------ #
        model = model_p3o(dados)

        # ------------------ Heurística Relax-and-Fix ------------------ #
        start = time.time()
        for t in model.T:  # Para cada período da instância
            for n in range(model.f[t], model.l[t] + 1):  # Subperíodos pertencentes ao período t
                # Variáveis X tornam-se inteiras
                for j in model.N:
                    model.X[j, n].domain = NonNegativeIntegers
                # Variáveis Y tornam-se inteiras
                for k in model.K:
                    model.Y[k, n].domain = NonNegativeIntegers
            # Resolução inteira no período t
            solver.solve(model, tee=False)
            # Fixar variáveis para próxima iteração
            for n in range(model.f[t], model.l[t] + 1):
                for j in model.N:
                    model.X[j, n].fix()
                for k in model.K:
                    model.Y[k, n].fix()
        end = time.time()

        # Escreve linha no arquivo csv
        writer.writerow([instancia.replace(".dat", ""),
                         f"{value(model.obj) :.2f}",
                         f"{benchmark[index] :.2f}",
                         f"{((value(model.obj) - benchmark[index]) / benchmark[index]) :.2f}",
                         f"{end - start :.2f}"])
