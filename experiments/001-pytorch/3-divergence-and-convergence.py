import torch

# Dados de treinamento: queremos aprender a relação y = 2x.
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0])

# Começamos com um peso que ainda não representa a relação correta.
w = torch.tensor(0.5, requires_grad=True)

# Intenção: controlar o tamanho dos ajustes feitos no peso.
#
# O tamanho do ajuste depende de:
# learning_rate * gradiente
#
# Learning rate pequena:
# - Atualizações menores.
# - Pode tornar o treinamento mais lento.
#
# Learning rate grande:
# - Atualizações maiores.
# - Pode ultrapassar o mínimo da loss.
# - Se os passos forem excessivos, pode causar divergência.
learning_rate = 0.01

for epoch in range(100):

    # 1. Calculamos a previsão.
    previsao = w * x

    # 2. Calculamos a loss para medir o erro do modelo.
    loss = ((previsao - y) ** 2).mean()

    # 3. Calculamos o gradiente.
    # Intenção: descobrir como a loss varia em relação ao peso.
    loss.backward()

    # 4. Atualizamos o peso na direção contrária ao gradiente.
    # Intenção: diminuir a loss.
    with torch.no_grad():
        if w.grad is not None:
            w -= learning_rate * w.grad

    # 5. Zeramos o gradiente para evitar acumulação.
    if w.grad is not None:
        w.grad.zero_()

    # 6. Acompanhamos o treinamento.
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Peso: {w.item():.4f} | Loss: {loss.item():.4f}")


# CONVERGÊNCIA:
# O peso se aproxima de um valor que minimiza a loss.
# A loss tende a diminuir ao longo do treinamento.
#
# DIVERGÊNCIA:
# Os ajustes afastam o peso do valor ideal.
# A loss pode crescer cada vez mais.
#
# IMPORTANTE:
# Inverter o gradiente escolhe a direção de descida local.
# Mas não garante convergência se o passo for grande demais.
#
# Intenção geral:
# Encontrar uma learning_rate que permita reduzir a loss
# com velocidade e estabilidade.