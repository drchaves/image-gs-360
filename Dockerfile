# Ambiente reproduzível do Image-GS + ARM (torchac).
# Base com nvcc 12.4 e gcc 11 (Ubuntu 22.04), igual ao ambiente conda original,
# pois o gsplat e o torchac compilam extensões CUDA/C++ (JIT) em tempo de execução.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9" \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential ninja-build \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh

# Ambiente conda "image-gs" (python 3.11.10, pytorch 2.4.1, pytorch-cuda 12.4 + deps pip fixadas)
COPY environment.yml /tmp/environment.yml
# Conda recente exige aceitar os Termos de Serviço dos canais "defaults" em modo não interativo
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create -f /tmp/environment.yml && conda clean -afy
# O conda resolve um mkl novo demais, que quebra o PyTorch 2.4.1 ("undefined symbol: iJIT_NotifyEvent").
# Fixa o mesmo mkl do ambiente original.
RUN conda install -n image-gs -y "mkl=2023.1.0" "intel-openmp=2023.0.0" && conda clean -afy
SHELL ["conda", "run", "--no-capture-output", "-n", "image-gs", "/bin/bash", "-c"]

# Dependências extras usadas no projeto (versões do ambiente original)
RUN pip install --no-cache-dir torchac==0.9.3 pandas==3.0.6 seaborn==0.13.2 \
    && pip install --no-cache-dir --no-build-isolation \
       "git+https://github.com/rahul-goel/fused-ssim/@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"

# gsplat customizado (kernel CUDA do Image-GS)
COPY gsplat /opt/gsplat
# --no-build-isolation: o setup.py do gsplat importa o torch já instalado no env
# (o env var propaga a opção para o pip aninhado que o setup.py develop dispara)
RUN PIP_NO_BUILD_ISOLATION=0 pip install --no-cache-dir --no-build-isolation -e "/opt/gsplat[dev]"

WORKDIR /workspace
# O código do projeto é montado em /workspace (docker run -v $PWD:/workspace ...)
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "image-gs"]
CMD ["python", "main.py", "--help"]

