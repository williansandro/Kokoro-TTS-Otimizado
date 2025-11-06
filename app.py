# ============= INTEGRAÇÃO COM KOKORO =============
# Adicione isso no app.py ou beta.py existente

from kokoro import KPipeline
import gradio as gr
from KokoroTextNormalizer import KokoroTextNormalizer  # Importar a classe criada acima

class KokoroTTSWithNormalizer:
    """
    Wrapper do Kokoro que normaliza o texto ANTES de processar com o modelo.
    Garante que números, datas, etc. sejam pronunciados corretamente.
    """
    
    def __init__(self, lang_code: str = 'a'):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.normalizer = KokoroTextNormalizer()
        self.lang_code = lang_code
    
    def normalizar_e_processar(self, texto: str, voice: str, speed: float = 1.0):
        """
        1. Normaliza o texto (números -> palavras)
        2. Processa com Kokoro
        3. Retorna áudio com timestamps
        """
        
        # PASSO 1: Normalizar texto
        print(f"🔤 Texto original: {texto[:100]}...")
        texto_normalizado = self.normalizer.normalizar(texto)
        print(f"✅ Texto normalizado: {texto_normalizado[:100]}...")
        
        # PASSO 2: Processar com Kokoro
        print(f"🎙️ Processando com Kokoro (voz: {voice}, velocidade: {speed}x)...")
        generator = self.pipeline(
            texto_normalizado, 
            voice=voice, 
            speed=speed, 
            split_pattern=r'\n+'
        )
        
        # PASSO 3: Coletar resultados
        resultados = []
        for resultado in generator:
            resultados.append({
                'graphemes': resultado.graphemes,
                'phonemes': resultado.phonemes,
                'audio': resultado.audio,
                'tokens': resultado.tokens
            })
        
        return resultados, texto_normalizado


# ============= FUNÇÃO PARA USAR NO GRADIO =============

def gerar_audio_com_normalizacao(
    texto: str,
    idioma_display: str,
    voz: str,
    velocidade: float,
    mostrar_texto_normalizado: bool = True
):
    """
    Função que integra com Gradio.
    Normaliza o texto e gera áudio com o Kokoro.
    """
    
    try:
        # Mapa de idiomas
        idioma_map = {
            '🇺🇸 American English': 'a',
            '🇬🇧 British English': 'b',
            '🇯🇵 Japanese': 'j',
            '🇨🇳 Mandarin Chinese': 'z',
            '🇪🇸 Spanish': 'e',
            '🇫🇷 French': 'f',
            '🇮🇳 Hindi': 'h',
            '🇮🇹 Italian': 'i',
            '🇧🇷 Brazilian Portuguese': 'p',
        }
        
        lang_code = idioma_map.get(idioma_display, 'a')
        
        # Criar instância com normalização
        kokoro_normalizado = KokoroTTSWithNormalizer(lang_code=lang_code)
        
        # Processar
        resultados, texto_normalizado = kokoro_normalizado.normalizar_e_processar(
            texto, voz, velocidade
        )
        
        if not resultados:
            return None, "❌ Nenhum áudio gerado", texto
        
        # Concatenar áudio de todos os segmentos
        import numpy as np
        import soundfile as sf
        from pathlib import Path
        
        audios = [r['audio'].numpy() for r in resultados]
        audio_completo = np.concatenate(audios)
        
        # Salvar
        output_dir = Path('./kokoro_audio')
        output_dir.mkdir(exist_ok=True)
        arquivo_saida = output_dir / f"audio_{idioma_display.split()[-1]}.wav"
        
        sf.write(str(arquivo_saida), audio_completo, 24000)
        
        msg_sucesso = f"✅ Áudio gerado com sucesso!\n"
        if mostrar_texto_normalizado:
            msg_sucesso += f"\n📝 Texto normalizado:\n{texto_normalizado}"
        
        return str(arquivo_saida), msg_sucesso, texto_normalizado
    
    except Exception as e:
        return None, f"❌ Erro: {str(e)}", texto


# ============= ATUALIZAR INTERFACE GRADIO =============
# Substitua a função anterior no seu app.py assim:

def criar_interface_com_normalizacao():
    """Interface Gradio com normalização de texto"""
    
    with gr.Blocks(title="Kokoro TTS + Text Normalizer") as app:
        gr.Markdown("# 🎙️ Kokoro TTS com Normalização de Texto")
        gr.Markdown("""
        ✨ **Recursos:**
        - Converte números para palavras (2 → two)
        - Datas em formato correto (05/03/1842 → March fifth, one thousand eight hundred and forty-two)
        - Percentuais, frações, operações matemáticas
        - Unidades de medida (10 km → ten kilometers)
        - Valores monetários (R$ 50 → fifty reais)
        - Numerais romanos (XIV → the Fourteenth)
        - **PRESERVA 100% da informação do texto original**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # INPUT
                gr.Markdown("### 📝 Texto de Entrada")
                texto_input = gr.Textbox(
                    label="",
                    placeholder="Digite seu texto com números, datas, etc...",
                    lines=8,
                    max_lines=20,
                    autofocus=True
                )
                
                gr.Markdown("### 🌍 Idioma")
                idioma_select = gr.Dropdown(
                    choices=[
                        '🇺🇸 American English',
                        '🇬🇧 British English',
                        '🇯🇵 Japanese',
                        '🇨🇳 Mandarin Chinese',
                        '🇪🇸 Spanish',
                        '🇫🇷 French',
                        '🇮🇳 Hindi',
                        '🇮🇹 Italian',
                        '🇧🇷 Brazilian Portuguese',
                    ],
                    value='🇺🇸 American English',
                    label="",
                    interactive=True
                )
                
                gr.Markdown("### 🎤 Voz")
                voz_select = gr.Dropdown(
                    choices=['af_bella', 'af_heart', 'af_nova', 'af_alloy', 'af_aoede'],
                    value='af_bella',
                    label="",
                    interactive=True
                )
                
                gr.Markdown("### ⚡ Velocidade")
                speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label=""
                )
                
                with gr.Row():
                    processar_btn = gr.Button("🎵 Gerar Áudio", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # OUTPUT
                gr.Markdown("### 🔊 Áudio")
                audio_output = gr.Audio(label="", interactive=False)
                
                gr.Markdown("### 📊 Status")
                status_text = gr.Textbox(label="", interactive=False, lines=3)
                
                gr.Markdown("### 📝 Texto Normalizado")
                texto_normalizado_output = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=5
                )
        
        # Event
        processar_btn.click(
            fn=gerar_audio_com_normalizacao,
            inputs=[texto_input, idioma_select, voz_select, speed_slider],
            outputs=[audio_output, status_text, texto_normalizado_output]
        )
    
    return app


# ============= EXEMPLOS PARA TESTAR =============

exemplos_teste = [
    ["He bought 2 swords on 05/15/1842.", "🇺🇸 American English", "af_bella"],
    ["The journey lasted 15 days through 47 villages.", "🇺🇸 American English", "af_bella"],
    ["She paid $10.50 for a 1/2 liter bottle.", "🇺🇸 American English", "af_bella"],
    ["The dragon was 386 years old with 99.9% power.", "🇺🇸 American English", "af_bella"],
    ["In 2025, the reward was 1,500 euros for 10 km.", "🇺🇸 American English", "af_bella"],
    ["Agent 007 followed protocol 87-B and the 3 to 1 ratio.", "🇺🇸 American English", "af_bella"],
]

if __name__ == "__main__":
    app = criar_interface_com_normalizacao()
    gr.Examples(examples=exemplos_teste, inputs=[texto_input, idioma_select, voz_select])
    app.launch(share=True)
