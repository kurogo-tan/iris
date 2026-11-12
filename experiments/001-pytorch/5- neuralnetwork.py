import torch
from torch import nn

# Cria uma camada com 1 característica de entrada e 1 saída.
# O PyTorch inicializa automaticamente o peso (w) e o bias (b).
# Fórmula: y_previsto = w*x + b
modelo = nn.Linear(1, 1)

# Cada lista interna representa um exemplo com 1 característica.
# Formato (3, 1): 3 exemplos, 1 característica por exemplo.
# Usar [2.0, 4.0, 6.0] indicaria 3 características na última dimensão.
x = torch.tensor([[2.0], [4.0], [6.0]])

# Respostas corretas para a relação y = 2x + 3.
y = torch.tensor([[7.0], [11.0], [15.0]])

optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01)
criterio = nn.MSELoss()
for epoch in range(10000):

    

    previsao = modelo(x)
    # Compara as previsões com as respostas corretas.
    loss = criterio(previsao, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

print(f"weight:{modelo.weight} bias:{modelo.bias}")

novo_x = torch.tensor([[10.0]])

with torch.no_grad():
    previsao = modelo(novo_x)

print(previsao.item())