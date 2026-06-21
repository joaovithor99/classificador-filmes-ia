import re
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="IA Movie Review", page_icon="🎬", layout="centered")

# --- DIAGNÓSTICO DE HARDWARE (GPU) ---
dispositivos = tf.config.list_physical_devices()
gpu_disponivel = any(d.device_type == 'GPU' for d in dispositivos)

# --- CACHE DO MODELO ---
@st.cache_resource
def treinar_ia():
    vocab_size = 10000
    max_length = 250
    (train_data, train_labels), _ = imdb.load_data(num_words=vocab_size)
    train_data = pad_sequences(train_data, maxlen=max_length, padding='post')
    
    model = Sequential([
        Embedding(vocab_size, 16),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=5, batch_size=512, verbose=0)
    return model

# --- INTERFACE VISUAL ---
st.title("🎬 Classificador de Sentimentos")
st.markdown("Analise críticas de filmes com **Redes Neurais** em tempo real.")

if gpu_disponivel:
    st.success("🚀 Aceleração por Placa de Vídeo (GPU) Ativada!")
else:
    st.warning("⚠️ Rodando em CPU (Placa de vídeo não detectada).")

with st.spinner("Treinando a Rede Neural... Aguarde alguns segundos."):
    model = treinar_ia()
    st.success("Inteligência Artificial pronta!")

st.divider()

# --- NOVOS CAMPOS DE ENTRADA ---
nome_filme = st.text_input("🍿 Qual filme você está avaliando?", placeholder="Ex: Motoqueiro Fantasma")
texto_usuario = st.text_area("✍️ Digite a sua crítica (pode escrever em Português!):", 
                             placeholder="Ex: Acabei de sair do Cinemark, o filme é sensacional e a atuação é perfeita!")

if st.button("Analisar Sentimento"):
    if texto_usuario and nome_filme:
        
        # 1. Tradução Invisível (PT-BR para Inglês)
        with st.spinner("Traduzindo e interpretando..."):
            texto_ingles = GoogleTranslator(source='pt', target='en').translate(texto_usuario)
        
        # 2. Pré-processamento (COM A CORREÇÃO DE PONTUAÇÃO)
        # O re.sub remove tudo que não for letra ou espaço (limpa as pontuações!)
        texto_limpo = re.sub(r'[^\w\s]', '', texto_ingles)
        
        word_index = imdb.get_word_index()
        word_index = {k: (v + 3) for k, v in word_index.items()}
        
        # Agora divide as palavras já limpas
        palavras = texto_limpo.lower().split()
        # Adicionamos a trava de segurança matemática (< 10000)
        tokens = [word_index.get(p, 2) if word_index.get(p, 2) < 10000 else 2 for p in palavras]
        tokens_pad = pad_sequences([tokens], maxlen=250, padding='post')
        
        # 3. A IA faz a previsão
        previsao = model.predict(tokens_pad, verbose=0)[0][0]
        
        st.divider()
        st.markdown(f"### Veredito para: **{nome_filme}**")
        
        # 4. Exibindo o Resultado
        if previsao > 0.5:
            st.success("✅ Opinião **POSITIVA**")
            st.progress(float(previsao))
            st.write(f"**Confiança da IA:** {previsao*100:.2f}%")
        else:
            st.error("❌ Opinião **NEGATIVA**")
            st.progress(float(1 - previsao))
            st.write(f"**Confiança da IA:** {(1-previsao)*100:.2f}%")
            
        # --- EXPLICAÇÃO DIDÁTICA ---
        st.info("ℹ️ **O que é a Confiança da IA?** \n\n"
                "É o nível de certeza matemática da Rede Neural. Como a máquina não tem sentimentos, ela calcula probabilidades baseadas no que aprendeu no treinamento:\n"
                "- Perto de **50%**: A IA está na dúvida. A crítica pode ser neutra, mista ou conter sarcasmo.\n"
                "- Perto de **100%**: A IA tem certeza absoluta de que encontrou padrões claros de um texto muito positivo ou muito negativo.")
        
        st.caption(f"*(Curiosidade para o professor: O texto original foi traduzido no backend para \"{texto_ingles}\" para a rede neural analisar, preservando a eficiência do modelo treinado no dataset do IMDb).*")
        
    else:
        st.error("Por favor, preencha o nome do filme e a crítica antes de analisar.")

st.sidebar.info("Projeto: Fundamentos de IA\n\nHardware: Aceleração via DirectML")
