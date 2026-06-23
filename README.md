# 🎬 Classificador de Sentimentos com Inteligência Artificial

Este projeto foi desenvolvido para a disciplina de **Fundamentos de Inteligência Artificial**. Trata-se de uma aplicação prática de *Machine Learning* e Processamento de Linguagem Natural (NLP) que utiliza Redes Neurais para analisar críticas de filmes e classificar a opinião do usuário como **Positiva** ou **Negativa**.

🌐 **[Clique aqui para acessar a aplicação online rodando na nuvem](https://classificador-filmes-ia-eekwnnbxdxjgpzoqkdvsjm.streamlit.app/)**

---

## 🧠 Arquitetura da Inteligência Artificial

O modelo foi construído utilizando **TensorFlow / Keras** e treinado com o **IMDb Movie Reviews Dataset**, contendo 50.000 avaliações de filmes. 

A arquitetura da Rede Neural Sequencial foi definida com as seguintes camadas:
1. **Embedding:** Converte o vocabulário (10.000 palavras) em vetores densos de 16 dimensões, mapeando a similaridade semântica entre as palavras.
2. **GlobalAveragePooling1D:** Reduz a dimensionalidade da sequência, extraindo um vetor médio de sentimentos para simplificar o cálculo e evitar *overfitting*.
3. **Dense (ReLU):** Camada oculta com 16 neurônios e função de ativação *Rectified Linear Unit*, responsável por encontrar padrões complexos e não-lineares nas frases.
4. **Dense (Sigmoid):** Um único neurônio de saída que "esmaga" o resultado em uma probabilidade matemática entre 0 e 1 (onde valores > 0.5 são positivos).

### ⚙️ Pipeline de Pré-processamento
Para permitir uma experiência de usuário fluida e em português, a aplicação conta com um pipeline invisível:
* **Tradução em Tempo Real:** Utiliza a biblioteca `deep-translator` para converter a entrada do português (PT-BR) para inglês (EN).
* **Limpeza por Regex:** Remove pontuações e caracteres especiais que poderiam gerar tokens desconhecidos (`<UNK>`).
* **Tokenização e Padding:** As palavras são convertidas em índices numéricos usando o dicionário do IMDb e ajustadas para um tamanho fixo de 250 tokens (`pad_sequences`).
* **Filtro de Outliers:** Garante que apenas palavras dentro do escopo do vocabulário treinado (índice < 10.000) sejam enviadas ao modelo.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10
* **Machine Learning:** TensorFlow & Keras
* **Interface Gráfica (Frontend):** Streamlit
* **Matemática e Matrizes:** NumPy (< 2.0.0)
* **Tradução de Dados:** Deep-Translator
* **Deploy / Nuvem:** Streamlit Community Cloud

---

## 🚀 Como executar o projeto localmente

Se desejar rodar o projeto na sua própria máquina (ideal para explorar a aceleração por hardware GPU no Windows via *DirectML*), siga as instruções abaixo:

### Pré-requisitos
* Ter o **Python 3.10** instalado (versões superiores podem ter conflitos com os *wheels* do TensorFlow).
* Ter o gerenciador de pacotes `pip` atualizado.

### Instalação

1. Clone este repositório ou baixe os arquivos.
2. Abra o terminal na pasta do projeto e instale as dependências:
```bash
pip install -r requirements.txt
