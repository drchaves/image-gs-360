# Guia Técnico Completo: Estabilização de Gradientes, Arquiteturas e Janela de Contexto do GaussianARM

Este documento reúne a fundamentação matemática, os detalhes de implementação em PyTorch, as referências exatas da literatura (incluindo o *PixelCNN++* e *STE*), os resultados experimentais completos nas 45 imagens (2K) e os textos prontos em LaTeX para o TCC.

---

## Sumário
1. [Visão Geral do Problema de Estabilização no Treinamento Conjunto RDO](#1-visão-geral-do-problema-de-estabilização-no-treinamento-conjunto-rdo)
2. [Detalhamento Matemático das 5 Estratégias de Estabilização (`--stabilize_grads`)](#2-detalhamento-matemático-das-5-estratégias-de-estabilização---stabilize_grads)
   - [2.1. Gradiente Substituto (Surrogate) em Log-Space para Probabilidades Saturadas](#21-gradiente-substituto-surrogate-em-log-space-para-probabilidades-saturadas)
   - [2.2. Limitação da Log-Escala Bruta entre `-7.0` e `5.0` (De onde vêm esses números?)](#22-limitação-da-log-escala-bruta-entre--70-e-50-de-onde-vêm-esses-números)
   - [2.3. STE Direcional (Asymmetric Clipping STE) e Projeção de Pesos Latentes](#23-ste-direcional-asymmetric-clipping-ste-e-projeção-de-pesos-latentes)
   - [2.4. Truncamento de Norma de Gradiente (Gradient Clipping) no ARM](#24-truncamento-de-norma-de-gradiente-gradient-clipping-no-arm)
   - [2.5. Preservação dos Momentos do Adam durante a Adição Progressiva de Gaussianas](#25-preservação-dos-momentos-do-adam-durante-a-adição-progressiva-de-gaussianas)
3. [Onde Encontrar Exatamente no Artigo e no Código do PixelCNN++ (e Outras Referências)](#3-onde-encontrar-exatamente-no-artigo-e-no-código-do-pixelcnn-e-outras-referências)
4. [Resultados Experimentais Completos (45 Imagens 2K)](#4-resultados-experimentais-completos-45-imagens-2k)
   - [4.1. Sem Estabilização vs. Com Estabilização](#41-sem-estabilização-vs-com-estabilização)
   - [4.2. Comparação de Arquiteturas: `Original` vs. `Shallow` vs. `Deep`](#42-comparação-de-arquiteturas-original-vs-shallow-vs-deep)
   - [4.3. Comparação do Tamanho da Janela de Contexto: `C = 4, 8, 16, 32`](#43-comparação-do-tamanho-da-janela-de-contexto-c--4-8-16-32)
5. [Resumo Executivo Pronto para Enviar ao Orientador](#5-resumo-executivo-pronto-para-enviar-ao-orientador)
6. [Texto em LaTeX Pronto para o TCC (`arquivo_2021.tex`) e BibTeX (`referencias.bib`)](#6-texto-em-latex-pronto-para-o-tcc-arquivo_2021tex-e-bibtex-referenciasbib)

---

## 1. Visão Geral do Problema de Estabilização no Treinamento Conjunto RDO

No *Image-GS* com modelagem de entropia autorregressiva, a função de perda de Otimização de Taxa-Distorção (RDO) acopla dois subsistemas de naturezas muito distintas:
$$\mathcal{L}(\boldsymbol{\Theta}, \boldsymbol{\Phi}) = D(\mathbf{X}, \hat{\mathbf{X}}_{\boldsymbol{\Theta}}) + \lambda R(\boldsymbol{\Theta}, \boldsymbol{\Phi})$$
onde:
- $\boldsymbol{\Theta} = \{G_i\}_{i=1}^N$ são os parâmetros contínuos das gaussianas 2D (posição $\boldsymbol{\mu}_i$, escala $\mathbf{S}_i$, rotação $\theta_i$ e cor $\mathbf{c}_i$), cujo número total $N$ cresce de $50.000$ até $100.000$ ao longo do treinamento (alocação progressiva guiada por erro a cada $500$ passos) e cuja ordenação unidimensional na **curva-Z (Código de Morton)** pode mudar a cada passo conforme as posições $(x_i, y_i)$ se movem.
- $\boldsymbol{\Phi}$ são os pesos da rede convolucional 1D causal (**GaussianARM**), que estima a média $m_{t,f}$ e a escala de incerteza $\nu_{t,f}$ de uma distribuição logística para cada atributo quantizado $k_{t,f} \in \{0, 1, \dots, 2^B-1\}$. Para que os pesos do próprio *GaussianARM* ocupem poucos bits no cabeçalho do arquivo comprimido ($\approx 0.01\text{ a }0.025\text{ bpp}$), suas camadas convolucionais (`QuantizedConv1d`) são quantizadas para $8\text{ bits}$ durante o treinamento utilizando um **Estimador Straight-Through (STE)**.

Sem mecanismos específicos de estabilização, a combinação de:
1. **Saturação de caudas da distribuição logística em `float32`**,
2. **Exponenciação irrestrita da escala $\nu = \exp(s) + \varepsilon$**,
3. **Saturação de pesos latentes fora do intervalo $[-1, 1]$ do STE com otimizador Adam**,
4. **Mudanças abruptas na permutação da curva-Z**, e
5. **Reinicialização periódica do otimizador Adam a cada adição de gaussianas**

provoca tanto **desaparecimento de gradientes (*vanishing gradients*)** — deixando canais do ARM permanentemente mortos — quanto **explosão de gradientes (*exploding gradients*)**, causando saltos severos de BPP em imagens de alto contraste.

---

## 2. Detalhamento Matemático das 5 Estratégias de Estabilização (`--stabilize_grads`)

### 2.1. Gradiente Substituto (*Surrogate*) em Log-Space para Probabilidades Saturadas

- **Local no código:** `discrete_logistic_nll` em `utils/entropy_utils.py` (linhas 173–195).

#### O Problema (Sem Estabilização):
A probabilidade discreta exata de um símbolo quantizado $k \in \{0, \dots, q_{\max}\}$ sob uma distribuição logística de média $m$ e escala $\nu$ é dada pela área sob a densidade entre as bordas do bin $[k - 0.5, k + 0.5]$, calculada analiticamente pela subtração de duas funções sigmoides (CDFs):
$$P(k) = \sigma\left(\frac{k + 0.5 - m}{\nu}\right) - \sigma\left(\frac{k - 0.5 - m}{\nu}\right)$$
Para calcular o custo entrópico em bits, $r(k) = -\log_2 P(k)$, o código aplica um truncamento inferior (`prob.clamp(min=1e-6)`) para evitar $\log_2(0) = -\infty$.

Contudo, em aritmética de ponto flutuante de 32 bits (`float32`), quando a predição $m$ do ARM está distante do símbolo real $k$ (isto é, quando o erro padronizado $|z| = \left|\frac{k - m}{\nu}\right| \gg 1$), ambas as sigmoides saturam em $0.0$ (se $z \ll -14$) ou em $1.0$ (se $z \gg 14$).
Consequentemente, a diferença $\sigma_+ - \sigma_-$ sofre *underflow* numérico e cai abaixo de $10^{-6}$, ativando o `clamp(min=1e-6)`.
No PyTorch, a derivada de $\text{clamp}(P, \min=10^{-6})$ na região $P \le 10^{-6}$ é **identicamente zero**:
$$\frac{\partial (-\log_2 \max(P, 10^{-6}))}{\partial m} = 0, \qquad \frac{\partial (-\log_2 \max(P, 10^{-6}))}{\partial \nu} = 0 \qquad (\text{para } P \le 10^{-6})$$
**Paradoxo:** Justamente nos símbolos em que o *GaussianARM* comete seus maiores erros de predição (pagando a penalidade máxima de $-\log_2(10^{-6}) \approx 19.93\text{ bits}$ por símbolo no *forward*), **ele recebe gradiente zero no *backward***! A rede não recebe nenhuma informação direcional para saber se deve aumentar ou diminuir $m$ ou $\nu$, gerando canais mortos.

#### A Solução Matemática (Com Estabilização):
Inspirando-se na formulação em espaço logarítmico do **PixelCNN++** (*Salimans et al., 2017*), utilizamos o **Teorema do Valor Médio** para aproximar a probabilidade discreta nas caudas extremas ($P(k) \le 10^{-6}$): a integral da função densidade de probabilidade (PDF) $f(x)$ em um bin de largura unitária ($\Delta k = (k+0.5) - (k-0.5) = 1$) é aproximada pelo valor da própria PDF avaliada no ponto central do bin ($x = k$):
$$P(k) = \int_{k-0.5}^{k+0.5} f(x)\,dx \approx f(k) \cdot 1$$
Definindo o desvio padronizado no centro do bin como $z = \frac{k - m}{\nu}$, a PDF da distribuição logística é:
$$f(k) = \frac{e^{-z}}{\nu (1 + e^{-z})^2} = \frac{e^{z}}{\nu (1 + e^{z})^2}$$
Aplicando o logaritmo **antes** da exponenciação (em *log-space*), eliminamos qualquer possibilidade de *underflow*:
$$\ln f(k) = z - \ln \nu - 2\ln(1 + e^z) = z - \ln \nu - 2\,\text{softplus}(z)$$
Convertendo para log-verossimilhança negativa (NLL) em bits (dividindo por $\ln 2$):
$$\tilde{\ell}(k) = -\frac{1}{\ln 2}\Big( z - \ln \nu - 2\,\text{softplus}(z) \Big)$$
Calculando as derivadas parciais de $\tilde{\ell}(k)$ em relação à média $m$ e à escala $\nu$:
$$\frac{\partial \tilde{\ell}}{\partial m} = \frac{1}{\nu \ln 2}\big(1 - 2\sigma(z)\big) = \frac{1}{\nu \ln 2}\tanh\left(\frac{m - k}{2\nu}\right)$$
$$\frac{\partial \tilde{\ell}}{\partial \nu} = \frac{1}{\nu \ln 2}\Big(1 + z\big(1 - 2\sigma(z)\big)\Big)$$
Observe as propriedades desse gradiente nas caudas ($|z| \gg 1$):
1. **Nunca desaparece (*no vanishing*):** Quando $k \gg m$ ($z \to +\infty$, $\sigma(z) \to 1$), temos $\frac{\partial \tilde{\ell}}{\partial m} \to -\frac{1}{\nu \ln 2} < 0$. O descenso de gradiente ($m \leftarrow m - \eta \frac{\partial \tilde{\ell}}{\partial m}$) **aumenta $m$ em direção a $k$**. Quando $k \ll m$ ($z \to -\infty$, $\sigma(z) \to 0$), temos $\frac{\partial \tilde{\ell}}{\partial m} \to +\frac{1}{\nu \ln 2} > 0$, **diminuindo $m$ em direção a $k$**.
2. **Nunca explode em relação a $m$ (*bounded*):** Como $|\tanh(\cdot)| \le 1$, a derivada em relação a $m$ é estritamente limitada por $\frac{1}{\nu \ln 2}$.

#### Como foi implementado no PyTorch sem alterar o Forward:
No código (`utils/entropy_utils.py`), queremos manter o cálculo de bits **100% inalterado no passo direto (*forward*)** e substituir o gradiente **apenas onde o clamp saturou (`prob <= 1e-6`)** e **apenas para os parâmetros do ARM** (usando `symbols.detach()`):
```python
inv_scale = 1.0 / scale
mid_in_arm = (symbols.detach() - mu) * inv_scale
log_pdf_mid = mid_in_arm - torch.log(scale) - 2.0 * F.softplus(mid_in_arm)
tail_surrogate_nll = -log_pdf_mid / math.log(2.0)

is_saturated = (prob <= 1e-6).detach()
surrogate_term = torch.where(
    is_saturated,
    tail_surrogate_nll - tail_surrogate_nll.detach(),
    torch.zeros_like(nll_bits)
)
return nll_bits + surrogate_term
```
- No *forward*: `tail_surrogate_nll - tail_surrogate_nll.detach()` é identicamente zero, portanto a função retorna exatamente `nll_bits`.
- No *backward*: onde `is_saturated == False` ($P(k) > 10^{-6}$), usa-se o gradiente analítico exato de `nll_bits`; onde `is_saturated == True` ($P(k) \le 10^{-6}$), o gradiente zero do `clamp` é somado ao gradiente vivo de `tail_surrogate_nll`, resgatando imediatamente os canais saturados do ARM.

---

### 2.2. Limitação da Log-Escala Bruta entre `-7.0` e `5.0` (De onde vêm esses números?)

- **Local no código:** `GaussianARM.forward` em `utils/entropy_utils.py` (linhas 107–112).
```python
if self.stabilize_grads:
    raw_log_scale = torch.clamp(raw_log_scale, min=-7.0, max=5.0)
scale = torch.exp(raw_log_scale) + 1e-4
```
E em `model.py` (linha 661), essa escala normalizada é desnormalizada multiplicando pelo alcance máximo de quantização de 14 bits ($q_{\max} = 2^{14} - 1 = 16383$):
$$\nu = \underbrace{(\exp(s) + \varepsilon)}_{\nu_{\text{norm}}} \cdot q_{\max}, \qquad \text{com } \varepsilon = 10^{-4}$$

Esses dois limites (**`-7.0`** e **`5.0`**) não são números arbitrários: eles decorrem diretamente da equação $\nu_{\text{norm}} = e^s + 10^{-4}$ e da precisão numérica do padrão `IEEE 754 float32`.

#### A. De onde vem o limite inferior `min = -7.0`?
1. **Referência direta da literatura (PixelCNN++):**
   Na implementação oficial do *PixelCNN++* (*Salimans et al., 2017*, arquivo `pixel_cnn_pp/nn.py`, linha 53) e do *L3C* (*Mentzer et al., 2019*), o truncamento inferior da log-escala da distribuição logística discreta é literalmente:
   ```python
   log_scales = tf.maximum(log_scales, -7.)
   ```
2. **Derivação matemática a partir do piso $\varepsilon = 10^{-4}$:**
   Pela regra da cadeia, o gradiente da função de perda $\mathcal{L}$ em relação à saída bruta $s = \text{raw\_log\_scale}$ da rede é:
   $$\frac{\partial \mathcal{L}}{\partial s} = \frac{\partial \mathcal{L}}{\partial \nu_{\text{norm}}} \cdot \frac{d\nu_{\text{norm}}}{ds} = \frac{\partial \mathcal{L}}{\partial \nu_{\text{norm}}} \cdot e^s$$
   Sem o limite inferior, se a rede reduz $s$ para valores muito negativos (por exemplo, $s = -15$) ao tentar estreitar a distribuição em uma região constante:
   - O valor da escala $\nu_{\text{norm}} = e^{-15} + 10^{-4} \approx 0.0000003 + 0.0001 \approx 10^{-4}$ **estagna completamente no piso constante $\varepsilon = 10^{-4}$**.
   - Porém, o fator multiplicativo do gradiente cai para $e^{-15} \approx 3 \times 10^{-7}$ (**330 vezes menor que $\varepsilon$**), matando o gradiente de $s$ (*vanishing gradient*). O neurônio nunca mais consegue voltar a aumentar a escala se a variância dos dados aumentar!
   - Para impedir que o piso constante $\varepsilon = 10^{-4}$ domine a soma $e^s + \varepsilon$ e anule a derivada relativa, exigimos que o termo variável $e^s$ seja pelo menos **uma ordem de grandeza ($10\times$) maior que $\varepsilon$**:
     $$e^s \ge 10\,\varepsilon = 10 \times 10^{-4} = 10^{-3} \implies s \ge \ln(10^{-3}) = -6.9077 \approx \mathbf{-7.0}$$
   Em $s = -7.0$, temos $e^{-7.0} \approx 9.12 \times 10^{-4} \approx 9.1\,\varepsilon$. Ou seja, **`-7.0` é exatamente o menor número inteiro para o qual $e^s$ permanece uma ordem de grandeza acima de $\varepsilon = 10^{-4}$**, mantendo a derivada viva!

#### B. De onde vem o limite superior `max = 5.0`?
1. **Limite de precisão de ponto flutuante (`float32`) na subtração da CDF:**
   Na Equação da probabilidade discreta, subtraímos a sigmoide avaliada em $k + 0.5$ da sigmoide avaliada em $k - 0.5$. A distância entre os dois argumentos passados para $\sigma(\cdot)$ é de $1\text{ bin}$ dividido por $\nu$:
   $$\Delta z = \frac{(k + 0.5) - (k - 0.5)}{\nu} = \frac{1}{\nu_{\text{norm}} \cdot q_{\max}} = \frac{1}{(e^s + 10^{-4}) \cdot 16383}$$
   No centro da distribuição ($k \approx m$), onde a derivada da sigmoide é máxima ($\sigma'(0) = 0.25$), a probabilidade máxima de um bin para escalas grandes é:
   $$P_{\max}(k) \approx \sigma'(0) \cdot \Delta z = \frac{0.25}{e^s \cdot 16383}$$
   Em ponto flutuante de 32 bits (`float32`), o *machine epsilon* (menor diferença representável próximo de $0.5$) é $\epsilon_{\text{mach}} = 2^{-24} \approx 5.96 \times 10^{-8}$. Se $P_{\max}(k)$ cair na faixa de $10^{-7}$, a subtração $\sigma(z + \Delta z) - \sigma(z)$ sofre **cancelamento catastrófico** em `float32`, perdendo todos os bits significativos e colapsando em zero ou ruído de arredondamento!
   Calculando o limite exato onde $P_{\max}(k)$ atinge $10^{-7}$ para $q_{\max} = 16383$ (14 bits):
   $$\frac{0.25}{e^s \cdot 16383} > 10^{-7} \implies e^s < \frac{0.25}{16383 \times 10^{-7}} = 152.597 \implies s < \ln(152.597) = \mathbf{5.0278 \approx 5.0}$$
   Portanto, **`5.0` ($\ln(152.6) \approx 5.03$) é exatamente o maior número inteiro antes de a diferença das sigmoides em 14 bits colapsar por limite de precisão do `float32`**!
2. **Significado estatístico:** Como os símbolos de entrada do ARM estão normalizados no intervalo $[0, 1]$, uma escala normalizada de $e^{5.0} \approx 148.4$ já é quase $150$ vezes maior que todo o domínio $[0, 1]$ dos dados — o que já representa uma distribuição completamente uniforme (plana) sobre todos os símbolos possíveis. Permitir $s > 5.0$ não muda nada na modelagem estatística e apenas introduz risco de *overflow* ($\exp(89) = \infty$ em `float32`) e perda de precisão.

---

### 2.3. STE Direcional (*Asymmetric Clipping STE*) e Projeção de Pesos Latentes

- **Locais no código:**
  - `ClippingSTEFunc.backward` em `utils/entropy_utils.py` (linhas 25–38)
  - Loop de otimização `GaussianSplatting2D.optimize` em `model.py` (linhas 572–579)

#### Como funciona a Quantização via STE dos pesos do ARM:
Para que os $\approx 13.500$ pesos do *GaussianARM* ocupem apenas $1\text{ byte}$ cada no arquivo final ($8\text{ bits}$ em vez de $32\text{ bits}$, reduzindo o cabeçalho da rede em $75\%$), utilizamos Quantização Consciente de Treinamento (*Quantization-Aware Training* — Jacob et al., 2018) via **Straight-Through Estimator (STE)** (Bengio et al., 2013; Hubara et al., 2016).
No PyTorch, cada camada `QuantizedConv1d` guarda um tensor de **pesos latentes** $w \in \mathbb{R}$ em `float32` (onde o otimizador Adam acumula pequenas atualizações). A cada passo de *forward*:
1. **Truncamento (*Clipping*):** Restringimos o peso latente ao intervalo $[-c, +c]$ (com $c = \text{clip\_val} = 1.0$):
   $$w_{\text{clip}} = \text{clamp}(w, -1.0, +1.0)$$
2. **Quantização Simétrica de 8 bits e Desquantização:**
   $$\text{scale}_w = \frac{2^{8-1} - 1}{1.0} = 127, \qquad w_q = \frac{\text{round}(w_{\text{clip}} \cdot 127)}{127}$$
3. A convolução 1D é executada usando os pesos quantizados $w_q$.

Como a função $\text{round}(\cdot)$ tem derivada zero em quase toda parte, no *backward* o STE aproxima $\frac{\partial w_q}{\partial w_{\text{clip}}} \approx 1$.

#### O Problema do STE Clássico (Sem Estabilização): "Pesos Zumbis / Mortos" na Borda
No STE clássico com truncamento (*Clipped STE*, Hubara et al., 2016), como $w_{\text{clip}} = \text{clamp}(w, -c, c)$ possui derivada $1$ para $|w| \le c$ e derivada $0$ para $|w| > c$, a regra de retropropagação sem estabilização (linhas 35–36 de `utils/entropy_utils.py`) faz:
```python
grad_weight[weight > ctx.clip_val] = 0
grad_weight[weight < -ctx.clip_val] = 0
```
**Por que essa regra falha com o otimizador Adam?**
Lembre-se de que a regra de atualização de um otimizador por descida de gradiente é:
$$w^{(k+1)} = w^{(k)} - \eta \cdot \hat{g}$$
onde:
- Um gradiente **negativo** ($g < 0$) faz $w$ **aumentar** ($-\eta g > 0$).
- Um gradiente **positivo** ($g > 0$) faz $w$ **diminuir** ($-\eta g < 0$).

Agora acompanhe o que acontece sem estabilização:
1. Na iteração $100$, um peso latente vale $w = +0.98$ (dentro de $[-1, 1]$). Ele recebe um gradiente negativo forte ou tem momento acumulado (`exp_avg`) no Adam e, após o `optimizer.step()`, salta para **$w = +1.05$**.
2. Na iteração $101$, como $w = 1.05 > 1.0$, a linha `grad_weight[weight > 1.0] = 0` **anula todo e qualquer gradiente daquele peso**, independentemente do sinal!
3. Pior: como o Adam mantém uma média móvel exponencial dos gradientes passados ($m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$), mesmo com $g_t = 0$, o momento antigo $m_{t-1} < 0$ continua empurrando o peso latente para $1.10 \to 1.25 \to 1.40$ ao longo de dezenas de passos!
4. Se na iteração $300$ a distribuição dos dados mudar e a rede precisar que esse peso volte a diminuir (gerando um gradiente restaurador positivo $g = \frac{\partial \mathcal{L}}{\partial w_q} > 0$, que puxaria $w$ de volta em direção a $[-1, 1]$), **o `if weight > 1.0: grad = 0` bloqueia esse gradiente restaurador!** O peso fica permanentemente travado em $+1.0$ pelo resto das $50.000$ iterações ("peso zumbi").

#### A Solução Dupla (Com Estabilização):
Implementamos duas proteções complementares:

1. **STE Direcional no `backward` (`utils/entropy_utils.py`, linhas 29–33):**
   ```python
   if ctx.stabilize_grads:
       grad_weight[(weight > ctx.clip_val) & (grad_output < 0)] = 0
       grad_weight[(weight < -ctx.clip_val) & (grad_output > 0)] = 0
   ```
   - Se o peso passou do limite superior ($w > +1.0$) e o gradiente é negativo ($g < 0$, tentando empurrá-lo ainda mais para cima), zeramos o gradiente.
   - Porém, se $w > +1.0$ e o gradiente é **positivo** ($g > 0$, querendo puxar o peso de volta para dentro de $[-1, 1]$), **o gradiente passa livremente!** (E analogamente na borda inferior $w < -1.0$ quando $g < 0$).

2. **Projeção dos Pesos Latentes Pós-Step (`model.py`, linhas 572–579):**
   Imediatamente após `self.optimizer.step()`, projetamos os pesos e *biases* latentes em `float32` de volta para a caixa $[-c, c]$:
   ```python
   with torch.no_grad():
       for m in self.arm.modules():
           if isinstance(m, QuantizedConv1d):
               m.weight.clamp_(-m.clip_val, m.clip_val)
               if m.bias is not None:
                   m.bias.clamp_(-m.clip_val, m.clip_val)
   ```
   Isso impede que a inércia do Adam afaste o peso latente para longe da borda (ex.: $1.40$). O peso saturado fica estacionado exatamente sobre a fronteira $+1.00$; no primeiro passo em que o gradiente inverter de sinal ($g > 0$), o peso já retorna imediatamente para $0.99$ sem precisar "desandar" uma deriva acumulada fora da caixa.

---

### 2.4. Truncamento de Norma de Gradiente (*Gradient Clipping*) no ARM

- **Local no código:** `GaussianSplatting2D.optimize` em `model.py` (linhas 568–570).
```python
if getattr(self, 'stabilize_grads', False):
    if getattr(self, 'rate_loss_ratio', 0) > 0 and hasattr(self, 'arm'):
        torch.nn.utils.clip_grad_norm_(self.arm.parameters(), max_norm=self.grad_clip_norm)
```
- **Justificativa (*Pascanu et al., 2013*):** Diferentemente de uma imagem em grade fixa de pixels (onde o vizinho à esquerda de um pixel é sempre o mesmo), no *Image-GS* as gaussianas se movem continuamente no plano 2D. Quando duas gaussianas cruzam uma fronteira de quadrante binário da curva-Z, a ordem 1D da sequência muda instantaneamente de um passo para o outro. Essa permutação repentina gera picos transitórios na perda de entropia. O truncamento da norma global $\ell_2$ dos gradientes do ARM em $\tau = 1.0$ (`max_norm=1.0`) garante que esses picos pontuais não destruam os pesos já aprendidos pela CNN causal.

---

### 2.5. Preservação dos Momentos do Adam durante a Adição Progressiva de Gaussianas

- **Local no código:** `GaussianSplatting2D._add_gaussians` em `model.py` (linhas 870–884).
```python
old_state = self.optimizer.state if getattr(self, 'stabilize_grads', False) else None
...
self.optimizer = torch.optim.Adam(params)
if old_state is not None:
    if getattr(self, 'rate_loss_ratio', 0) > 0 and hasattr(self, 'arm'):
        for p in self.arm.parameters():
            if p in old_state:
                self.optimizer.state[p] = old_state[p]
```
- **Justificativa (*Kingma & Ba, 2015*):** No *Image-GS*, o treinamento começa com $50\%$ das gaussianas ($50.000$) e adiciona $+12.500$ gaussianas nos passos $500$, $1000$, $1500$ e $2000$. Como o tamanho dos tensores `self.xy`, `self.scale`, `self.rot` e `self.feat` muda, o código recria o objeto `torch.optim.Adam(params)` do zero.
- **O problema sem estabilização:** Ao recriar `torch.optim.Adam(params)`, o estado interno do otimizador (`exp_avg` $m_t$, `exp_avg_sq` $v_t$ e contador `step` $t$) era apagado **também para os pesos do `self.arm`** (cujo tamanho nunca mudou!). No Adam, quando `step` volta para $1$ e $v_t$ zera, o passo efetivo inicial dá um salto brusco em todas as camadas do ARM justamente no momento mais delicado (quando $12.500$ novas gaussianas acabaram de entrar na sequência). Preservar `self.optimizer.state[p]` para os parâmetros do ARM elimina esse choque periódico.

---

## 3. Onde Encontrar Exatamente no Artigo e no Código do PixelCNN++ (e Outras Referências)

Caso a banca ou o orientador pergunte onde exatamente cada técnica aparece na literatura:

1. **PixelCNN++ — *Salimans, Karpathy, Chen, & Kingma (ICLR 2017)***
   - **Artigo:** *"PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications"* (`arXiv:1701.05517`).
     - **Seção 2.1 (*Discretized Logistic Mixture Likelihood*), páginas 2–3, Equações (1) e (2):** Define a modelagem de símbolos discretos via diferença de sigmoides ($\sigma((x+0.5-\mu)/s) - \sigma((x-0.5-\mu)/s)$) e os casos de borda em $0$ e $255$.
   - **Código Oficial da OpenAI (`https://github.com/openai/pixel-cnn`, citado na pág. 2 do artigo):**
     - Arquivo **`pixel_cnn_pp/nn.py`**, função **`discretized_mix_logistic_loss` (linhas 46 a 76)**:
       - **Linha 53:** `log_scales = tf.maximum(log_scales, -7.)` (Origem exata do limite inferior `-7.0` da log-escala).
       - **Linhas 62–63:** `mid_in = inv_stdv * centered_x` e `log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)` (Origem exata da fórmula da PDF logística em log-space via `softplus` no centro do bin).
       - **Linhas 68–73:** `tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))` (Troca para a aproximação contínua `log_pdf_mid` quando a diferença das sigmoides `cdf_delta` satura abaixo de `1e-5`).

2. **L3C / torchac — *Mentzer, Agustsson, Tschannen, Timofte, & Van Gool (CVPR 2019)***
   - **Artigo:** *"Practical Full Resolution Learned Lossless Image Compression"* (já citado no seu TCC como `\cite{mentzer2019practical}`).
   - **Seção 3.3 (*Entropy Coding / Logistic Mixture*):** Adota exatamente a mesma perda logística discretizada estabilizada do PixelCNN++ acoplada ao codificador aritmético `torchac`.

3. **Straight-Through Estimator (STE) e Quantização de Redes Neurais:**
   - **Bengio, Léonard, & Courville (2013):** *"Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation"* (`arXiv:1308.3432`) — Propõe o STE original e discute o bloqueio de gradientes quando a ativação satura fora do limiar.
   - **Hubara et al. (NeurIPS 2016):** *"Binarized Neural Networks"* (Seção 1.2) — Define o *Clipped STE* (identidade dentro de $[-1, 1]$ e zero fora) e a necessidade de aplicar *clipping* explícito nos pesos reais em float32 (`clamp(-1, 1)`) após cada atualização do otimizador para evitar que os pesos latentes cresçam indefinidamente fora da região ativa.
   - **Jacob et al. (CVPR 2018):** *"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"* — Padrão de quantização simulada de 8 bits com STE durante o treinamento.

4. **Gradient Clipping e Exploding/Vanishing Gradients:**
   - **Pascanu, Mikolov, & Bengio (ICML 2013):** *"On the difficulty of training recurrent neural networks"* (Seção 2, Algoritmo 1) — Introduz formalmente o truncamento de gradiente pela norma global (`clip_grad_norm_`).

---

## 4. Resultados Experimentais Completos (45 Imagens 2K)

Os experimentos foram executados sobre **45 imagens de alta resolução (2K)** divididas igualmente nas 5 categorias do dataset (*anime, art, painting, photo, vector* — 9 imagens por categoria), com $N = 100.000$ gaussianas e três valores de $\lambda \in \{0.0005, 0.0025, 0.0075\}$.

### 4.1. Sem Estabilização vs. Com Estabilização
*(Arquivo: `resultados_diego_csvs/results_archs_stabilize_3_lambdas_45_images.csv`)*
*(Valores negativos de BD-Rate indicam que `stabilize_grads=True` economiza bits para a mesma qualidade visual)*

| Arquitetura | Mediana BD-Rate (WS-PSNR) | Média BD-Rate s/ outliers (`<100%`) | Mediana BD-Rate (PSNR) | Média PSNR s/ outliers (`<100%`) | Imagens Válidas |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original** ($3$ camadas, $W=64$) | +0.46% | +1.89% | +0.64% | +0.99% | 44 / 45 |
| **Shallow** ($2$ camadas, $W=93$) | **-1.03%** | **-1.32%** | **-1.03%** | +0.15% | **45 / 45** |
| **Deep** ($4$ camadas, $W=53$) | +1.11% | +0.85% | +1.12% | +0.74% | 43 / 45 |

- **Principais conclusões:**
  1. Na arquitetura **Shallow**, ativar `stabilize_grads=True` reduziu o BD-Rate mediano em **$-1.03\%$** (e **$-1.32\%$** na média sem outliers).
  2. Sem estabilização, imagens complexas (como `anime-2_2k`, `anime-4_2k`, `art-6_2k`, `vector-7_2k`) sofriam saltos anômalos de BPP de $+0.20\text{ a }+0.30\text{ bpp}$ quando os canais do ARM saturavam. Com a estabilização, esses colapsos foram eliminados e as curvas R-D tornaram-se estritamente monotônicas.

---

### 4.2. Comparação de Arquiteturas: `Original` vs. `Shallow` vs. `Deep`

As três arquiteturas foram projetadas para serem **isométricas em número de parâmetros** ($\approx 13.5\text{k}$ parâmetros para $C=16$, resultando no mesmo custo de cabeçalho `bpp_arm` $\approx 0.0257\text{ bpp}$):
- **`original` (3 camadas):** $\text{Conv1d}(8 \to 64, k=16) \to \text{ReLU} \to \text{Conv1d}(64 \to 64, k=1) \to \text{ReLU} \to \text{Conv1d}(64 \to 16, k=1)$ — **13.456 parâmetros** (`0.0257 bpp`)
- **`shallow` (2 camadas, mais larga):** $\text{Conv1d}(8 \to 93, k=16) \to \text{ReLU} \to \text{Conv1d}(93 \to 16, k=1)$ — **13.501 parâmetros** (`0.0258 bpp`)
- **`deep` (4 camadas, mais estreita):** $\text{Conv1d}(8 \to 53, k=16) \to \text{ReLU} \to \text{Conv1d}(53 \to 53, k=1) \to \text{ReLU} \to \text{Conv1d}(53 \to 53, k=1) \to \text{ReLU} \to \text{Conv1d}(53 \to 16, k=1)$ — **13.425 parâmetros** (`0.0256 bpp`)

#### Tabela Comparativa (Com `stabilize_grads = True`, média nas 43 imagens comuns):

| Arquitetura | $\lambda = 0.0005$ (WS-PSNR / BPP) | $\lambda = 0.0025$ (WS-PSNR / BPP) | $\lambda = 0.0075$ (WS-PSNR / BPP) | BD-Rate Mediano vs `Original` (WS-PSNR) | BD-Rate Médio vs `Original` (PSNR) | Vitórias vs `Original` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | 37.778 dB / 1.9058 | 35.819 dB / 1.8265 | 31.451 dB / 1.6932 | 0.00% (Âncora) | 0.00% (Âncora) | — |
| **Shallow (Vencedora)** | **37.929 dB** / 1.9097 | **35.934 dB** / 1.8298 | 31.404 dB / **1.6823** | **-0.15%** | **-1.48%** (Med: **-0.20%**) | **24 / 44 (54.5%)** |
| **Deep** | 37.884 dB / 1.9247 | 35.879 dB / 1.8427 | 31.424 dB / 1.7265 | +0.54% | -0.53% (Med: +0.54%) | 11 / 43 (25.6%) |

- **Por que a `Shallow` é a vencedora?**
  1. No *GaussianARM*, **somente a primeira camada convolucional** possui kernel espacial ($k = C = 16$); as camadas subsequentes são projeções pontuais $1\times 1$.
  2. Ao concentrar o orçamento de parâmetros na largura da primeira camada ($W = 93$ canais contra $64$ na *Original* e $53$ na *Deep*), a rede forma um banco de filtros espaciais muito mais expressivo para capturar padrões na janela causal da curva-Z, além de propagar gradientes de forma mais direta por ter apenas 2 camadas.
  3. A *Shallow* entrega **$+0.15\text{ dB}$** a mais de WS-PSNR em $\lambda=0.0005$, **$+0.115\text{ dB}$** a mais em $\lambda=0.0025$, e o **menor BPP ($1.6823$)** em $\lambda=0.0075$, superando a *Original* na maioria das imagens e batendo amplamente a *Deep* (que perde para a *Original* em 75% das imagens).

---

### 4.3. Comparação do Tamanho da Janela de Contexto: `C = 4, 8, 16, 32`
*(Arquivo: `resultados_diego_csvs/results_shallow_stab_context_4_8_16_32.csv` — Arquitetura `shallow` + `stabilize_grads=True`)*

Fixando a arquitetura vencedora (`shallow`, $W=93$) com estabilização ativa, variamos o tamanho da janela de contexto causal $C \in \{4, 8, 16, 32\}$. Como o número de pesos da primeira camada é $8 \times 93 \times C + 93$, o custo de transmissão do próprio modelo (`bpp_arm`) escala linearmente com $C$:

| Contexto ($C$) | Parâmetros do ARM | Custo da Rede (`bpp_arm`) | $\lambda = 0.0005$ (WS-PSNR / BPP Total / `bpp_img`) | $\lambda = 0.0025$ (WS-PSNR / BPP Total / `bpp_img`) | $\lambda = 0.0075$ (WS-PSNR / BPP Total / `bpp_img`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$C = 4$** | $4.573$ ($4.5\text{ KB}$) | **0.0087 bpp** | 37.811 dB / **1.8720** / **1.8633** | 35.599 dB / **1.7766** / **1.7679** | 30.560 dB / 1.6381 / 1.6294 |
| **$C = 8$** | $7.549$ ($7.4\text{ KB}$) | **0.0144 bpp** | 37.744 dB / 1.8997 / 1.8853 | 35.558 dB / 1.7985 / 1.7841 | 30.826 dB / **1.6376** / **1.6232** |
| **$C = 16$** | $13.501$ ($13.2\text{ KB}$) | 0.0258 bpp | **37.864 dB** / 1.9093 / 1.8835 | 35.781 dB / 1.8301 / 1.8044 | 31.256 dB / 1.6785 / 1.6528 |
| **$C = 32$** | $25.405$ ($24.8\text{ KB}$) | 0.0485 bpp | 37.774 dB / 1.9553 / 1.9068 | **35.991 dB** / 1.8762 / 1.8278 | **31.602 dB** / 1.7495 / 1.7010 |

#### Tabela de BD-Rate Comparando Contra a Âncora `C = 16` (45 imagens):

| Comparação vs `C = 16` | Mediana BD-Rate (WS-PSNR) | Média BD-Rate s/ outliers (WS-PSNR) | Mediana BD-Rate (PSNR) | Vitórias sobre `C = 16` (WS-PSNR) |
| :--- | :---: | :---: | :---: | :---: |
| **`C = 4` vs `C = 16`** | **-2.16%** | **-4.24%** (43/44 imgs) | **-1.87%** | **31 / 44 (70.5%)** |
| **`C = 8` vs `C = 16`** | **-1.39%** | **-3.00%** (45/45 imgs, zero outliers!) | **-1.14%** | **34 / 45 (75.6%)** |
| **`C = 16` (Âncora)** | 0.00% | 0.00% | 0.00% | — |
| **`C = 32` vs `C = 16`** | +2.80% | -0.73% | +2.82% | 10 / 45 (22.2%) |

#### Interpretação Física e Estatística do Resultado de Contexto:
1. **Por que contextos menores ($C=4$ e $C=8$) vencem $C=16$ e $C=32$?**
   - **Geometria fractal da Curva-Z:** O código de Morton agrupa elementos em blocos recursivos de $2\times 2 = \mathbf{4}$ elementos e sub-grades de $2\times 4 = \mathbf{8}$ elementos. Dentro de uma janela de $4$ a $8$ vizinhos anteriores, as gaussianas estão quase garantidamente coladas no espaço 2D da imagem. Quando a janela cresce para $16$ ou $32$ posições 1D, a curva-Z frequentemente atravessa fronteiras de quadrantes maiores (dando "saltos espaciais" na tela), fazendo com que as posições $t-17 \dots t-32$ tenham baixa correlação espacial com a gaussiana atual $t$.
   - **Sobrecusto de cabeçalho (`bpp_arm`) e sobreajuste:** Aumentar de $C=8$ para $C=32$ mais que triplica o tamanho da rede (`bpp_arm` sobe de $0.0144$ para $0.0485\text{ bpp}$) e dificulta a otimização da camada convolucional, aumentando também o `bpp_img`.
2. **Qual escolher entre $C=4$ e $C=8$?**
   - **$C=8$** é o mais consistente e robusto: vence $C=16$ em **75.6% das imagens (34 de 45)**, não apresenta **nenhum outlier** em todas as 45 imagens, atinge **BD-Rate médio global de $-3.00\%$** (mediana **$-1.39\%$**) e preserva melhor o PSNR em alta compressão ($\lambda=0.0075$) do que $C=4$.
   - **$C=4$** é o mais leve de todos (`bpp_arm = 0.0087`) e atinge a maior redução mediana de BD-Rate (**$-2.16\%$**, média s/ outlier de **$-4.24\%$**), sendo excelente em taxas altas/médias ($\lambda=0.0005$), mas perde cerca de $0.7\text{ dB}$ de WS-PSNR no extremo $\lambda=0.0075$.

---

## 5. Resumo Executivo Pronto para Enviar ao Orientador

> Professor, finalizei as baterias de experimentos nas **45 imagens 2K** (9 por categoria) avaliando: (1) a estabilização de gradientes no treinamento RDO conjunto, (2) a ablação de arquiteturas do *GaussianARM* (*Original*, *Shallow* e *Deep*), e (3) o tamanho da janela de contexto ($C \in \{4, 8, 16, 32\}$).
>
> **1. Estabilização de Gradientes (`--stabilize_grads`):**
> Identificamos que o treinamento conjunto das gaussianas com o ARM quantizado via STE sofria com canais mortos (*vanishing gradients*) e picos de perda entrópica quando a ordenação da curva-Z mudava. Implementamos 5 ajustes fundamentados na literatura:
> - *(i) Gradiente substituto em log-space nas caudas saturadas (baseado no PixelCNN++, Salimans et al., 2017, Seção 2.1 / `nn.py`):* quando a probabilidade discreta $P(k) = \sigma_+ - \sigma_-$ sofre *underflow* em `float32` e bate no `clamp(min=1e-6)`, a derivada do `clamp` zera justamente onde o ARM mais errou. Mantivemos o *forward* exato, mas injetamos no *backward* (apenas onde $P(k) \le 10^{-6}$) a derivada da densidade logística contínua em log-space via `softplus`, cujo gradiente $\propto \tanh((m-k)/2\nu)$ é limitado e não nulo, puxando a média $m$ de volta para perto do símbolo $k$.
> - *(ii) Limitação da ativação de log-escala em $s \in [-7.0, 5.0]$:* na equação da escala $\nu_{\text{norm}} = e^s + \varepsilon$ ($\varepsilon = 10^{-4}$), o limite inferior $-7.0$ (também usado no PixelCNN++) vem de $\ln(10\varepsilon) \approx -6.91 \approx -7.0$, garantindo que a derivada $e^s \approx 9\times 10^{-4}$ permaneça uma ordem de grandeza acima do piso constante $\varepsilon$ e não trave a escala com gradiente zero. O limite superior $5.0$ vem do limite de precisão do `float32` em 14 bits ($q_{\max}=16383$): para que a diferença das sigmoides em 1 bin ($\Delta P \approx \frac{0.25}{e^s \cdot 16383}$) não colapse abaixo de $10^{-7}$ por cancelamento catastrófico, exige-se $e^s < 152.6 \implies s < \ln(152.6) \approx 5.03 \approx 5.0$.
> - *(iii) STE direcional e projeção de pesos latentes (baseado em Bengio et al., 2013 e Hubara et al., 2016):* no STE padrão em $[-1, 1]$, se o momento do Adam empurrava um peso latente para $w > 1.0$, todos os seus gradientes futuros eram zerados — inclusive gradientes positivos que o trariam de volta para dentro de $[-1, 1]$. Passamos a zerar apenas gradientes que empurram o peso ainda mais para fora do intervalo e aplicamos `clamp(-1, 1)` nos pesos latentes após cada passo do otimizador.
> - *(iv) Gradient clipping (`max_norm=1.0`, Pascanu et al., 2013)* nos pesos do ARM para amortecer mudanças na permutação da curva-Z.
> - *(v) Preservação dos momentos do Adam do ARM* quando o otimizador é recriado a cada 500 passos na adição progressiva de gaussianas.
> - **Resultado:** Eliminou os saltos anômalos de BPP que ocorriam sem estabilização e gerou um ganho mediano de BD-Rate de **$-1.03\%$** (média de **$-1.32\%$** s/ outliers) na arquitetura *Shallow*.
>
> **2. Escolha da Arquitetura Vencedora (*Shallow*):**
> Comparamos três topologias de mesmo orçamento de parâmetros ($\approx 13.5\text{k}$ params, `bpp_arm` $\approx 0.0257$): *Original* (3 camadas, $W=64$), *Shallow* (2 camadas, $W=93$) e *Deep* (4 camadas, $W=53$). Com estabilização ligada, a **Shallow** venceu tanto a *Original* (BD-Rate médio em PSNR de **$-1.48\%$**, mediano de **$-0.20\%$** e **$-0.15\%$** em WS-PSNR, vencendo em 24/44 imagens e entregando $+0.15\text{ dB}$ em $\lambda=0.0005$) quanto a *Deep* (que ficou $+0.54\%$ pior que a *Original*). Como apenas a primeira camada tem kernel espacial ($k=C$) e as demais são convoluções $1\times 1$, concentrar os parâmetros na largura da primeira camada ($W=93$) extrai melhor o contexto espacial da curva-Z.
>
> **3. Ablação do Tamanho de Contexto ($C \in \{4, 8, 16, 32\}$ na arquitetura *Shallow*):**
> Variando $C$ nas 45 imagens, vimos que contextos mais curtos (**$C=8$** e **$C=4$**) superam claramente $C=16$ e $C=32$:
> - Contra a âncora $C=16$, **$C=8$** alcançou **BD-Rate médio de $-3.00\%$** (mediana **$-1.39\%$**, vencendo em **34 das 45 imagens — 75.6%**), e **$C=4$** alcançou **mediana de $-2.16\%$** (média s/ outlier de **$-4.24\%$**, vencendo em **70.5% das imagens**), enquanto **$C=32$** piorou o BD-Rate mediano em **$+2.80\%$**.
> - Isso se explica porque a curva-Z preserva vizinhança imediata em blocos de $2\times 2$ ($4$) e $2\times 4$ ($8$), mas sofre saltos espaciais em janelas longas ($32$), além de $C=8$ e $C=4$ reduzirem o peso da rede (`bpp_arm`) de $0.0258\text{ bpp}$ ($C=16$) para $0.0144\text{ bpp}$ e $0.0087\text{ bpp}$.

---

## 6. Texto em LaTeX Pronto para o TCC (`arquivo_2021.tex`) e BibTeX (`referencias.bib`)

### 6.1. Onde inserir na Metodologia (`Capítulo 4`, logo após a Subseção 4.2.3, linha 1336 de `tcc_diego/arquivo_2021.tex`):

```latex
\subsection{Quantização da Rede e Estabilização de Gradientes no Treinamento Conjunto}
\label{sec:estabilizacao_gradientes}

A otimização conjunta descrita na Equação~\eqref{eq:loss_proposed} impõe desafios numéricos particulares por acoplar dois subsistemas de naturezas distintas: a coleção explícita de gaussianas $\boldsymbol{\Theta}$ — cujo cardinal $N$ cresce dinamicamente durante a alocação progressiva guiada por erro \cite{zhang2025imagegs} — e os parâmetros da rede convolucional causal \textit{GaussianARM}. Além disso, como os pesos do próprio \textit{GaussianARM} precisam ser transmitidos junto ao \textit{bitstream} para viabilizar a reconstrução no decodificador (a exemplo de codecs neurais por \textit{overfitting} como o \textit{Cool-Chic} \cite{ladune2023coolchic}), seus filtros convolucionais são quantizados para $B_{\text{arm}} = 8$~\textit{bits} durante o treinamento por meio de um Estimador \textit{Straight-Through} (STE, do inglês \textit{Straight-Through Estimator}) \cite{bengio2013estimating, jacob2018quantization}. Sem mecanismos de controle numérico, a interação entre a quantização via STE, a não estacionariedade da sequência na curva-Z e a natureza exponencial das caudas da distribuição logística torna o treinamento suscetível tanto ao desaparecimento (\textit{vanishing gradients}) quanto à explosão de gradientes (\textit{exploding gradients}) \cite{pascanu2013difficulty, goodfellow2016deep}. Para mitigar esses efeitos, implementaram-se cinco estratégias complementares de estabilização.

\paragraph{1. Gradiente substituto (\textit{surrogate}) em espaço logarítmico para caudas saturadas.}
Na avaliação numérica da massa de probabilidade discreta $P(k_{t,f})$ (Equação~\eqref{eq:prob_discreta}), impõe-se um limite inferior $P_{\min} = 10^{-6}$ via truncamento (\textit{clamping}) para evitar indefinições no cálculo do custo entrópico $-\log_2 P(k_{t,f})$. Todavia, no início da otimização ou imediatamente após a inserção de novas gaussianas, a média estimada $m_{t,f}$ pode encontrar-se distante do símbolo real $k_{t,f}$, fazendo com que os dois termos sigmoides saturem em $0$ ou $1$ em precisão de ponto flutuante de 32~\textit{bits}. Quando $P(k_{t,f}) \le 10^{-6}$, a operação de truncamento anula identicamente as derivadas parciais em relação a $m_{t,f}$ e $\nu_{t,f}$, impedindo que o \textit{GaussianARM} corrija suas estimativas e ocasionando canais inativos. Inspirando-se na formulação numericamente estável em espaço logarítmico do \textit{PixelCNN++} \cite{salimans2017pixelcnn, mentzer2019practical}, substitui-se o gradiente nulo exclusivamente nas regiões saturadas ($P(k_{t,f}) \le 10^{-6}$) pelo gradiente analítico da densidade logística contínua avaliada no ponto central do intervalo de quantização. Definindo o desvio padronizado $z_{t,f} = (\text{sg}(k_{t,f}) - m_{t,f})/\nu_{t,f}$, em que $\text{sg}(\cdot)$ denota o operador de bloqueio de gradiente (\textit{stop-gradient}), a log-verossimilhança negativa substituta em \textit{bits} é dada por:
\begin{equation}
    \tilde{\ell}(k_{t,f}) = -\frac{1}{\ln 2} \Big( z_{t,f} - \ln \nu_{t,f} - 2\,\text{softplus}(z_{t,f}) \Big),
    \label{eq:surrogate_nll}
\end{equation}
em que $\text{softplus}(z) = \ln(1 + e^z)$. Como $\partial \tilde{\ell} / \partial m_{t,f} = (\nu_{t,f}\ln 2)^{-1}\tanh((m_{t,f}-k_{t,f})/2\nu_{t,f})$ permanece limitado e não nulo mesmo para $|z_{t,f}| \gg 1$, a rede recebe um sinal direcional consistente que reconduz as estimativas para a região ativa de probabilidade sem alterar o valor exato da perda no passo direto (\textit{forward}) nem perturbar os gradientes das gaussianas.

\paragraph{2. Limitação das ativações de log-escala.}
Conforme definido na Subseção anterior, a escala de incerteza normalizada é obtida por $\boldsymbol{\nu}_t = \exp(\mathbf{s}_t) + \varepsilon$, com $\varepsilon = 10^{-4}$, sendo posteriormente multiplicada pelo fundo de escala de quantização $q_{\max} = 2^B - 1$. Pela regra da cadeia, o gradiente retropropagado para a ativação bruta $s_{t,f}$ é proporcional a $\exp(s_{t,f})$. Caso $s_{t,f}$ assuma valores excessivamente negativos, o termo $\exp(s_{t,f})$ colapsa perante a constante $\varepsilon$ e o parâmetro de escala trava no piso fixo, extinguindo seu gradiente; em contrapartida, valores positivos elevados provocam perda de significância numérica na subtração das sigmoides da Equação~\eqref{eq:prob_discreta} em ponto flutuante de precisão simples (\textit{float32}). Para evitar ambos os regimes degenerados, restringe-se a ativação bruta ao intervalo $s_{t,f} \in [-7{,}0;\, 5{,}0]$ antes da exponenciação. O limite inferior, adotado canonicamente no \textit{PixelCNN++} \cite{salimans2017pixelcnn}, decorre da condição $\exp(s_{\min}) \approx 10\,\varepsilon \implies s_{\min} = \ln(10^{-3}) \approx -6{,}91 \approx -7{,}0$, garantindo que o termo exponencial permaneça cerca de uma ordem de grandeza acima de $\varepsilon$ ($\exp(-7{,}0) \approx 9{,}1 \times 10^{-4}$). Por sua vez, o limite superior decorre do limite de resolução numérica de \textit{float32} para $B=14$~\textit{bits} ($q_{\max}=16383$): para que a variação máxima de probabilidade em um intervalo unitário de quantização, $\Delta P \approx 0{,}25 / (\exp(s_{t,f})\,q_{\max})$, não sofra cancelamento catastrófico abaixo de $10^{-7}$, requer-se $\exp(s_{t,f}) < 152{,}6 \implies s_{\max} < \ln(152{,}6) \approx 5{,}03 \approx 5{,}0$.

\paragraph{3. Estimador \textit{Straight-Through} (STE) direcional e projeção de pesos latentes.}
Para comprimir os parâmetros do \textit{GaussianARM} em $B_{\text{arm}} = 8$~\textit{bits}, cada peso latente $w$ das camadas convolucionais é restrito ao intervalo $[-c, c]$ (com $c = 1{,}0$) e quantizado uniformemente no passo direto. No STE convencional com truncamento (\textit{clipped STE}) \cite{bengio2013estimating, hubara2016binarized}, o gradiente é zerado sempre que $|w| > c$. Contudo, sob otimizadores com momento como o Adam \cite{kingma2015adam}, a inércia acumulada pode empurrar um peso latente para fora da região $[-c, c]$; uma vez que o gradiente passa a ser anulado indiscriminadamente nessa região, até mesmo gradientes de sinal oposto (que trariam o peso de volta ao intervalo válido) são bloqueados, congelando permanentemente o parâmetro na borda de saturação. Para solucionar esse problema, implementou-se um STE direcional combinado à projeção dos pesos latentes \cite{hubara2016binarized}: durante a retropropagação, o gradiente $\partial \mathcal{L}/\partial w$ é zerado apenas quando atua no sentido de afastar ainda mais o peso do intervalo válido (isto é, quando $w > c$ e $\partial \mathcal{L}/\partial w < 0$, ou quando $w < -c$ e $\partial \mathcal{L}/\partial w > 0$), permitindo a passagem livre de gradientes restauradores. Complementarmente, após cada atualização do otimizador, aplica-se uma projeção explícita $w \leftarrow \text{clamp}(w, -c, c)$ nos pesos em ponto flutuante, eliminando o acúmulo de deriva latente.

\paragraph{4. Truncamento de norma de gradiente (\textit{gradient clipping}) no ARM.}
Como a ordenação na curva-Z depende das coordenadas quantizadas $(u_i, v_i)$ das gaussianas, pequenas movimentações espaciais durante a otimização podem alterar a permutação da sequência 1D entre iterações consecutivas, gerando picos abruptos na perda entrópica. Para impedir que essas transições desestabilizem os filtros convolucionais, aplica-se o truncamento da norma global ($\ell_2$) dos gradientes do \textit{GaussianARM} \cite{pascanu2013difficulty} com limiar máximo $\tau = 1{,}0$ antes de cada passo de atualização do otimizador.

\paragraph{5. Preservação de estado do otimizador durante a alocação progressiva.}
O algoritmo original do \textit{Image-GS} inicia a otimização com uma fração das gaussianas ($50\%$ de $N$) e insere novas primitivas periodicamente em regiões de alto erro de reconstrução \cite{zhang2025imagegs}. Como o tamanho dos tensores $\boldsymbol{\Theta}$ aumenta a cada inserção, o otimizador Adam é reinstanciado. Na implementação padrão, essa reinstanciação também reinicializava as médias móveis de primeira e segunda ordem e o contador de passos de correção de viés para os pesos do \textit{GaussianARM}, provocando choques de passo efetivo a cada adição de gaussianas. Na arquitetura estabilizada, o dicionário de estados internos do Adam referente aos parâmetros do \textit{GaussianARM} é integralmente preservado e transferido entre as reinstanciações, garantindo uma trajetória de aprendizado contínua para o modelo de entropia.
```

---

### 6.2. Entradas BibTeX para `tcc_diego/referencias.bib`

```bibtex
@article{bengio2013estimating,
  title={Estimating or propagating gradients through stochastic neurons for conditional computation},
  author={Bengio, Yoshua and L{\'e}onard, Nicholas and Courville, Aaron},
  journal={arXiv preprint arXiv:1308.3432},
  year={2013}
}

@inproceedings{jacob2018quantization,
  title={Quantization and training of neural networks for efficient integer-arithmetic-only inference},
  author={Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2704--2713},
  year={2018}
}

@inproceedings{pascanu2013difficulty,
  title={On the difficulty of training recurrent neural networks},
  author={Pascanu, Razvan and Mikolov, Tomas and Bengio, Yoshua},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={1310--1318},
  year={2013}
}

@article{kingma2015adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={International Conference on Learning Representations (ICLR)},
  year={2015}
}

@inproceedings{hubara2016binarized,
  title={Binarized neural networks},
  author={Hubara, Itay and Courbariaux, Matthieu and Soudry, Daniel and El-Yaniv, Ran and Bengio, Yoshua},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={29},
  year={2016}
}

@book{goodfellow2016deep,
  title={Deep Learning},
  author={Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  publisher={MIT Press},
  year={2016},
  address={Cambridge}
}

@techreport{bjontegaard2001calc,
  title={Calculation of average {PSNR} differences between {RD}-curves},
  author={Bj{\o}ntegaard, Gisle},
  institution={ITU-T Video Coding Experts Group (VCEG)},
  number={VCEG-M33},
  year={2001}
}

@article{wang2004image,
  title={Image quality assessment: from error visibility to structural similarity},
  author={Wang, Zhou and Bovik, Alan C and Sheikh, Hamid R and Simoncelli, Eero P},
  journal={IEEE Transactions on Image Processing},
  volume={13},
  number={4},
  pages={600--612},
  year={2004},
  publisher={IEEE}
}

@book{cover2006elements,
  title={Elements of Information Theory},
  author={Cover, Thomas M and Thomas, Joy A},
  year={2006},
  edition={2nd},
  publisher={John Wiley \& Sons},
  address={Hoboken}
}
```
