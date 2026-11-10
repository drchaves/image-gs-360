import numpy as np
import matplotlib.pyplot as plt
import os

# Parametros de exemplo
mu = 10.2
scale = 1.5
symbol = 10
lower = symbol - 0.5
upper = symbol + 0.5

# Gerando eixo X
x = np.linspace(4, 16, 500)

# PDF (Função de Densidade de Probabilidade) - A Curva de Sino
# PDF logistica = exp(-(x-mu)/s) / (s * (1 + exp(-(x-mu)/s))^2)
z = (x - mu) / scale
pdf = np.exp(-z) / (scale * (1 + np.exp(-z))**2)

# CDF (Função de Distribuição Acumulada) - A Curva S (Sigmoide)
cdf = 1 / (1 + np.exp(-z))

# Valores exatos nos limites
cdf_upper = 1 / (1 + np.exp(-(upper - mu) / scale))
cdf_lower = 1 / (1 + np.exp(-(lower - mu) / scale))
prob = cdf_upper - cdf_lower

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# --- PLOT 1: A Curva PDF (Sino) ---
ax1.plot(x, pdf, color='black', lw=2, label='PDF (Curva Logística Contínua)')
ax1.axvline(mu, color='red', linestyle='--', label=f'mu (Pico da Previsão = {mu})')

# Preenchendo a área (A "Fatia de Bolo")
x_fill = np.linspace(lower, upper, 100)
z_fill = (x_fill - mu) / scale
pdf_fill = np.exp(-z_fill) / (scale * (1 + np.exp(-z_fill))**2)
ax1.fill_between(x_fill, pdf_fill, color='blue', alpha=0.3, label=f'Área = Probabilidade do Símbolo {symbol} (Prob: {prob:.3f})')

ax1.axvline(lower, color='blue', linestyle=':', label=f'Lower Bound ({lower})')
ax1.axvline(upper, color='blue', linestyle=':', label=f'Upper Bound ({upper})')
ax1.set_title("PDF: Área sob a curva = Probabilidade")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- PLOT 2: A Curva CDF (Sigmóide) ---
ax2.plot(x, cdf, color='green', lw=2, label='CDF (Função Sigmóide Acumulada)')

# Marcando os pontos cdf_lower e cdf_upper
ax2.scatter([lower], [cdf_lower], color='blue', s=100, zorder=5, label=f'cdf_lower ({cdf_lower:.3f})')
ax2.scatter([upper], [cdf_upper], color='purple', s=100, zorder=5, label=f'cdf_upper ({cdf_upper:.3f})')

# Linhas de anotação
ax2.hlines(y=cdf_upper, xmin=4, xmax=upper, color='purple', linestyle='--', alpha=0.5)
ax2.hlines(y=cdf_lower, xmin=4, xmax=lower, color='blue', linestyle='--', alpha=0.5)
ax2.axvline(lower, color='blue', linestyle=':', alpha=0.5)
ax2.axvline(upper, color='purple', linestyle=':', alpha=0.5)

# A diferença
ax2.annotate(f'', xy=(14, cdf_lower), xytext=(14, cdf_upper), arrowprops=dict(arrowstyle='<->', color='black'))
ax2.text(14.2, (cdf_lower + cdf_upper)/2, f'Diferença = {prob:.3f}\n(É a exata Área Azul do Gráfico Acima!)', va='center')

ax2.set_title("CDF: A diferença entre os pontos no eixo Y dá a Área exata")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results/debug", exist_ok=True)
plt.savefig("results/debug/logistic_demo.png", dpi=150)
print("Saved to results/debug/logistic_demo.png")
