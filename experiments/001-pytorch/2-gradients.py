import torch

# O objetivo do treinamento é ajustar os pesos para que
# o modelo aprenda uma determinada relação.

# Neste exemplo, a relação é y = 2x.
# Ou seja, a saída deve ser o dobro da entrada.

# O modelo começa tentando prever y = wx,
# onde w é um peso ajustável.

# No aprendizado supervisionado, precisamos de exemplos
# de entrada (x) e saída correta (y) para calcular o erro.

# O gradiente é a derivada da função de erro em relação ao peso.
# Ele indica como o erro varia quando modificamos esse peso.

# Com o gradiente e a taxa de aprendizado (learning rate),
# conseguimos ajustar o peso para tentar diminuir o erro.


# Dados de treinamento: y = 2x
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0])


# Peso inicial do modelo
w = torch.tensor(0.5, requires_grad=True)


# Taxa de aprendizado
learning_rate = 0.1

for epoch in range(100):

    previsao = w*x
    
    # O VERDADEIRO ERRO ESTA NESSE CALCULO
    erro = previsao - y 
    
    # Aqui utilizamos uma tecnica pra deixar o erro ao quadrado 
    # ( tipo um modulo, pq pra gente o interessante é pontuar o quanto a ia errou )
    # Nesse caso o quadrado tambem penaliza erros maiores
    # , depois fazemos a media desses erros
    loss = ((erro)**2).mean()

    # 3. Calculamos o gradiente
    # Intenção: calcular matematicamente o que acontece com a loss
    # se aumentarmos um pouquinho o peso (w).
    #
    # A derivada nos informa como a loss varia em relação ao peso:
    # - Gradiente negativo: aumentar o peso tende a diminuir a loss.
    # - Gradiente positivo: diminuir o peso tende a diminuir a loss.
    # - Gradiente zero: não indica uma direção de descida naquele ponto.
    #
    # O backward() apenas calcula o gradiente e o armazena em w.grad.
    # Depois, usamos essa informação para decidir como ajustar o peso,
    # com o objetivo de reduzir a loss.
    loss.backward()

    # Intenção: atualizar o peso na direção que diminui a loss.
    #
    # O gradiente indica como a loss varia ao aumentar um pouco o peso.
    # Subtraímos o gradiente porque queremos reduzir a loss:
    #
    # - Gradiente positivo: diminuímos o peso.
    # - Gradiente negativo: aumentamos o peso.
    #
    # A learning_rate controla o tamanho do ajuste.
    # Quanto menor o valor, menor será o passo.
    #
    # Fórmula: novo_peso = peso_atual - (learning_rate * gradiente)
    # O objetivo é aproximar o peso de um valor que minimize a loss.
    with torch.no_grad():
        if w.grad is not None:
            w -= learning_rate * w.grad

    # Intenção: limpar o gradiente utilizado na atualização anterior.
    #
    # O PyTorch acumula gradientes a cada loss.backward().
    # Se não zerarmos, o próximo gradiente será somado ao anterior.
    #
    # Usamos zero_() para que a próxima iteração calcule o novo
    # gradiente sem acumular o resultado da iteração passada.
    if w.grad is not None:
        w.grad.zero_()

    

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | "
            f"Peso: {w.item():.4f} | "
            f"Loss: {loss.item():.4f}"
        )

print(f"\nPeso aprendido: {w.item():.4f}")
print(f"Previsão para x=5: {(w * 5).item():.4f}")