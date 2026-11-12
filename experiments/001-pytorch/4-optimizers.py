import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0])

w = torch.tensor(0.5, requires_grad=True)

# Criamos o otimizador responsável por atualizar w.
# optimizer = torch.optim.SGD([w], lr=0.01)
# o Adam é melhor porque ele guarda informacoes do gradiente e consegue ajustar melhor o peso
optimizer = torch.optim.Adam([w], lr=0.01)

for epoch in range(100):

    # Fazemos a previsão.
    previsao = w * x

    # Calculamos a perda.
    loss = ((previsao - y) ** 2).mean()

    # Limpamos os gradientes anteriores.
    optimizer.zero_grad()

    # Calculamos o gradiente da loss em relação a w.
    loss.backward()

    # Atualizamos w utilizando o gradiente calculado.
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Peso: {w.item():.4f} | Loss: {loss.item():.4f}")