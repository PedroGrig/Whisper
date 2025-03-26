{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/PedroGrig/Whisper/blob/main/TRANSCRI%C3%87%C3%83O.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Configurar o ambiente (rode esta célula somente antes da primeira transcrição)\n",
        "!pip install git+https://github.com/openai/whisper.git\n",
        "!sudo apt update && sudo apt install ffmpeg"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wpT9JZpWbHLs",
        "outputId": "aa43efda-c935-4bc5-d62e-aab9f59f0e73",
        "collapsed": true
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting git+https://github.com/openai/whisper.git\n",
            "  Cloning https://github.com/openai/whisper.git to /tmp/pip-req-build-lzyn34wh\n",
            "  Running command git clone --filter=blob:none --quiet https://github.com/openai/whisper.git /tmp/pip-req-build-lzyn34wh\n",
            "  Resolved https://github.com/openai/whisper.git to commit 517a43ecd132a2089d85f4ebc044728a71d49f6e\n",
            "  Installing build dependencies ... \u001b[?25l\u001b[?25hdone\n",
            "  Getting requirements to build wheel ... \u001b[?25l\u001b[?25hdone\n",
            "  Preparing metadata (pyproject.toml) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: more-itertools in /usr/local/lib/python3.11/dist-packages (from openai-whisper==20240930) (10.6.0)\n",
            "Requirement already satisfied: numba in /usr/local/lib/python3.11/dist-packages (from openai-whisper==20240930) (0.60.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from openai-whisper==20240930) (2.0.2)\n",
            "Requirement already satisfied: tiktoken in /usr/local/lib/python3.11/dist-packages (from openai-whisper==20240930) (0.9.0)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (from openai-whisper==20240930) (2.6.0+cu124)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from openai-whisper==20240930) (4.67.1)\n",
            "Requirement already satisfied: triton>=2 in /usr/local/lib/python3.11/dist-packages (from openai-whisper==20240930) (3.2.0)\n",
            "Requirement already satisfied: llvmlite<0.44,>=0.43.0dev0 in /usr/local/lib/python3.11/dist-packages (from numba->openai-whisper==20240930) (0.43.0)\n",
            "Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.11/dist-packages (from tiktoken->openai-whisper==20240930) (2024.11.6)\n",
            "Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.11/dist-packages (from tiktoken->openai-whisper==20240930) (2.32.3)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (3.18.0)\n",
            "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (4.12.2)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (3.4.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (3.1.6)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (2025.3.0)\n",
            "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (12.4.127)\n",
            "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (12.4.127)\n",
            "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (12.4.127)\n",
            "Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (9.1.0.70)\n",
            "Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (12.4.5.8)\n",
            "Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (11.2.1.3)\n",
            "Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (10.3.5.147)\n",
            "Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (11.6.1.9)\n",
            "Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (12.3.1.170)\n",
            "Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (0.6.2)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (2.21.5)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (12.4.127)\n",
            "Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (12.4.127)\n",
            "Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch->openai-whisper==20240930) (1.13.1)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch->openai-whisper==20240930) (1.3.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (2025.1.31)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch->openai-whisper==20240930) (3.0.2)\n",
            "Hit:1 https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/ InRelease\n",
            "Hit:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease\n",
            "Hit:3 http://archive.ubuntu.com/ubuntu jammy InRelease\n",
            "Hit:4 http://security.ubuntu.com/ubuntu jammy-security InRelease\n",
            "Hit:5 http://archive.ubuntu.com/ubuntu jammy-updates InRelease\n",
            "Hit:6 http://archive.ubuntu.com/ubuntu jammy-backports InRelease\n",
            "Hit:7 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease\n",
            "Hit:8 https://ppa.launchpadcontent.net/graphics-drivers/ppa/ubuntu jammy InRelease\n",
            "Hit:9 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease\n",
            "Hit:10 https://r2u.stat.illinois.edu/ubuntu jammy InRelease\n",
            "Reading package lists... Done\n",
            "Building dependency tree... Done\n",
            "Reading state information... Done\n",
            "35 packages can be upgraded. Run 'apt list --upgradable' to see them.\n",
            "\u001b[1;33mW: \u001b[0mSkipping acquire of configured file 'main/source/Sources' as repository 'https://r2u.stat.illinois.edu/ubuntu jammy InRelease' does not seem to provide it (sources.list entry misspelt?)\u001b[0m\n",
            "Reading package lists... Done\n",
            "Building dependency tree... Done\n",
            "Reading state information... Done\n",
            "ffmpeg is already the newest version (7:4.4.2-0ubuntu0.22.04.1).\n",
            "0 upgraded, 0 newly installed, 0 to remove and 35 not upgraded.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Informe o nome do arquivo e pressione <Enter>\n",
        "# Se a audiência for muito grande, pegue um café... Não se esqueça de baixar o seu arquivo \"txt\" ao final.\n",
        "\n",
        "arquivo = input(\"DIGITE O NOME DO ARQUIVO:\")\n",
        "print(\"Aguarde...\\n\")\n",
        "!whisper {arquivo} --model medium --task transcribe --language pt --output_format txt"
      ],
      "metadata": {
        "id": "MyWZQipOWana",
        "collapsed": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c8324eca-6132-405d-f184-108db97406f6"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "DIGITE O NOME DO ARQUIVO:Charlie.mp3\n",
            "Aguarde...\n",
            "\n",
            "[00:00.000 --> 00:15.260]  Tão natural quanto a luz do dia Mas que preguiça boa me deixa aqui à toa\n",
            "[00:15.260 --> 00:24.260]  Hoje ninguém vai estragar meu dia Só vou gastar energia pra beijar sua boca\n",
            "[00:24.260 --> 00:34.260]  Fica comigo então, não me abandona não Alguém te perguntou como é que foi seu dia\n",
            "[00:34.260 --> 00:42.260]  Uma palavra amiga, uma notícia boa Isso faz falta no dia a dia\n",
            "[00:42.260 --> 00:54.260]  A gente nunca sabe quem são essas pessoas Eu só queria te lembrar\n",
            "[00:54.260 --> 01:04.260]  Que aquele tempo eu não podia fazer mais Por nós eu estava errado e você não tem que me perdoar\n",
            "[01:04.260 --> 01:14.260]  Mas também quero te mostrar Que existe um lado bom nessa história\n",
            "[01:14.260 --> 01:26.260]  Tudo que ainda temos a compartilhar E viver e cantar não importa qual seja o dia\n",
            "[01:26.260 --> 01:34.260]  Vamos viver, vadiar O que importa é nossa alegria\n",
            "[01:34.260 --> 01:42.260]  Vamos viver e cantar Não importa qual seja o dia\n",
            "[01:42.260 --> 01:50.260]  Vamos viver, vadiar O que importa é nossa alegria\n",
            "[01:50.260 --> 01:58.260]  Tão natural quanto a luz do dia Mas que preguiça boa me deixa aqui à toa\n",
            "[01:58.260 --> 02:08.260]  Hoje ninguém vai estragar meu dia Só vou gastar energia pra beijar sua boca\n",
            "[02:08.260 --> 02:18.260]  Eu só queria te lembrar Que aquele tempo eu não podia fazer mais\n",
            "[02:18.260 --> 02:24.260]  Por nós eu estava errado e você não tem que me perdoar\n",
            "[02:24.260 --> 02:34.260]  Mas também quero te mostrar Que existe um lado bom nessa história\n",
            "[02:34.260 --> 02:46.260]  Tudo que ainda temos a compartilhar E viver e cantar não importa qual seja o dia\n",
            "[02:46.260 --> 02:54.260]  Vamos viver, vadiar O que importa é nossa alegria\n",
            "[02:54.260 --> 03:02.260]  Vamos viver e cantar Não importa qual seja o dia\n",
            "[03:02.260 --> 03:10.260]  Vamos viver, vadiar O que importa é nossa alegria\n",
            "[03:10.260 --> 03:16.260]  Tão natural quanto a luz do dia\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit pyngrok\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "lDhtVjAXCcam",
        "outputId": "a1861923-297b-44a5-e0f5-8b180637e186"
      },
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.11/dist-packages (1.44.0)\n",
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.11/dist-packages (7.2.3)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (8.1.8)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (11.1.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.29.4)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.11/dist-packages (from streamlit) (3.1.44)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.4.2)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.11/dist-packages (from pyngrok) (6.0.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (1.31.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.11/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.1)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2025.1.31)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.11/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (25.3.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.36.2)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.23.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "\n",
        "st.title(\"Transcrição de Áudio 🎤\")\n",
        "st.write(\"Bem-vindo ao sistema de transcrição automática!\")\n",
        "\n",
        "# Upload do arquivo\n",
        "arquivo = st.file_uploader(\"Envie um arquivo de áudio\", type=[\"mp3\", \"wav\", \"ogg\"])\n",
        "\n",
        "if arquivo is not None:\n",
        "    st.success(\"Arquivo recebido! Iniciando transcrição...\")\n",
        "    # Aqui você pode chamar o Whisper para processar o arquivo\n",
        "    # Exemplo: transcricao = transcrever_audio(arquivo)\n",
        "    transcricao = \"Texto transcrito aqui...\"  # Apenas um placeholder\n",
        "    st.text_area(\"Transcrição:\", transcricao)\n",
        "\n",
        "st.sidebar.markdown(\"Feito com ❤️ usando Streamlit\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k6Nfhlj_LyHy",
        "outputId": "80e95876-a9bc-4474-c6a7-7dc743176be4"
      },
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from pyngrok import ngrok\n",
        "import time\n",
        "\n",
        "# Inicia o Streamlit em segundo plano\n",
        "!streamlit run app.py &\n",
        "\n",
        "# Aguarda alguns segundos para garantir que o Streamlit iniciou\n",
        "time.sleep(5)\n",
        "\n",
        "# Cria um túnel com ngrok na porta 8501 (porta padrão do Streamlit)\n",
        "public_url = ngrok.connect(port=\"8501\")\n",
        "\n",
        "# Exibe o link público gerado\n",
        "print(f\"🔗 Acesse o app aqui: {public_url}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Bvrt-B3CL4rU",
        "outputId": "55dc12f4-ad22-4d22-e016-b5199833015f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.124.156.168:8501\u001b[0m\n",
            "\u001b[0m\n"
          ]
        }
      ]
    }
  ]
}