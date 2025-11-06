import gradio as gr
import soundfile as sf
import subprocess
import os
import re
import gc
import torch
from kokoro import KPipeline
import logging
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÃO DE LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURAÇÕES DE IDIOMAS E VOZES
# ============================================

IDIOMAS = {
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

VOZES_POR_IDIOMA = {
    'a': ['af_bella', 'af_heart', 'af_nova', 'af_alloy', 'af_aoede', 'af_jessica', 'af_kore', 'af_nicole', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa'],
    'b': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
    'j': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'],
    'e': ['ef_dora', 'em_alex', 'em_santa'],
    'f': ['ff_siwis'],
    'h': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'i': ['if_sara', 'im_nicola'],
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],
}

TAMANHO_CHUNK = {
    'j': 1500,
    'default': 3000
}

# ============================================
# PASTA DE SAÍDA LOCAL
# ============================================

# Criar pasta de saída na raiz do projeto (não em /content/)
OUTPUT_DIR = Path('./kokoro_audio')
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================
# VARIÁVEIS GLOBAIS
# ============================================

processing_state = {
    'active': False,
    'stop_requested': False,
    'pipeline': None,
    'lang_code': None
}

# ============================================
# FUNÇÕES DE GERENCIAMENTO
# ============================================

def get_pipeline(lang_code: str) -> KPipeline:
    """Retorna pipeline cacheado ou cria novo"""
    if processing_state['pipeline'] is None or processing_state['lang_code'] != lang_code:
        processing_state['lang_code'] = lang_code
        processing_state['pipeline'] = KPipeline(lang_code=lang_code)
    return processing_state['pipeline']

def dividir_texto(texto: str, max_chars: int, idioma: str) -> list:
    """Divide o texto em chunks inteligentemente"""
    
    if not texto.strip():
        return []

    if idioma == 'j':
        chunks = []
        current_chunk = ""
        for char in texto:
            current_chunk += char
            if len(current_chunk) >= max_chars:
                pontuacao = ['。', '！', '？', '、', '\n']
                idx = -1
                for p in pontuacao:
                    if p in current_chunk[-50:]:
                        idx = current_chunk.rfind(p) + 1
                        break
                if idx > 0:
                    chunks.append(current_chunk[:idx])
                    current_chunk = current_chunk[idx:]
                else:
                    chunks.append(current_chunk)
                    current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    else:
        paragrafos = texto.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragrafos:
            if len(para) > max_chars:
                frases = re.split(r'(?<=[.!?])\s+', para)
                for frase in frases:
                    if len(current_chunk) + len(frase) > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = frase + " "
                    else:
                        current_chunk += frase + " "
            else:
                if len(current_chunk) + len(para) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
                else:
                    current_chunk += para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

def limpar_memoria():
    """Limpa memória entre chunks"""
    try:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    except Exception as e:
        logger.warning(f"Erro ao limpar memória: {e}")

def gerar_nome_arquivo(idioma_display: str) -> str:
    """Gera nome único para arquivo de saída"""
    import uuid
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(uuid.uuid4())[:8]
    idioma_abrev = idioma_display.split()[-1][:3].lower()
    return f"audio_{idioma_abrev}_{timestamp}_{random_id}.wav"

# ============================================
# PROCESSAMENTO DE ÁUDIO
# ============================================

def processar_audio(texto: str, idioma_display: str, voz: str, velocidade: float) -> Tuple[Optional[str], str]:
    """Processa o áudio com salvamento LOCAL"""
    
    processing_state['active'] = True
    processing_state['stop_requested'] = False

    if not texto.strip():
        processing_state['active'] = False
        return None, "❌ Erro: Texto vazio"

    # Diretório temporário de trabalho
    temp_dir = OUTPUT_DIR / 'temp'
    temp_dir.mkdir(exist_ok=True)

    try:
        # Obter código de idioma
        idioma = IDIOMAS.get(idioma_display)
        if not idioma:
            return None, "❌ Erro: Idioma não reconhecido"

        tamanho_chunk = TAMANHO_CHUNK.get(idioma, TAMANHO_CHUNK['default'])

        # Dividir texto
        chunks = dividir_texto(texto, tamanho_chunk, idioma)
        if not chunks:
            return None, "❌ Erro: Texto vazio após processamento"

        logger.info(f"🔊 Processando {len(chunks)} chunks com idioma '{idioma_display}'")

        # Obter pipeline
        pipeline = get_pipeline(idioma)

        # Processar
        arquivos_partes = []
        sr = 24000

        for chunk_num, chunk in enumerate(chunks, 1):
            if processing_state['stop_requested']:
                logger.info("⏹️ Processamento cancelado")
                processing_state['active'] = False
                return None, "⏹️ Processamento cancelado"

            logger.info(f"🔄 Chunk {chunk_num}/{len(chunks)}")

            try:
                generator = pipeline(chunk, voice=voz, speed=velocidade)
                
                for seg_num, (gs, ps, audio) in enumerate(generator):
                    if processing_state['stop_requested']:
                        processing_state['active'] = False
                        return None, "⏹️ Processamento cancelado"

                    # SALVAR EM PASTA LOCAL (não em /content/)
                    arquivo = temp_dir / f'seg_{chunk_num}_{seg_num}.wav'
                    sf.write(str(arquivo), audio, sr)
                    arquivos_partes.append(arquivo)

            except Exception as e:
                logger.error(f"Erro ao processar chunk {chunk_num}: {e}")
                return None, f"❌ Erro: {str(e)}"

            limpar_memoria()

        if processing_state['stop_requested']:
            processing_state['active'] = False
            return None, "⏹️ Processamento cancelado"

        # Concatenar com FFmpeg (LOCAL)
        logger.info(f"🔗 Concatenando {len(arquivos_partes)} segmentos com FFmpeg")

        try:
            # Nome do arquivo de saída
            output_filename = gerar_nome_arquivo(idioma_display)
            output_path = OUTPUT_DIR / output_filename

            if len(arquivos_partes) == 1:
                # Se houver apenas um arquivo, copiar direto
                import shutil
                shutil.copy(str(arquivos_partes[0]), str(output_path))
            else:
                # Usar FFmpeg para concatenar
                concat_list_path = temp_dir / 'concat_list.txt'
                
                with open(str(concat_list_path), 'w') as f:
                    for arquivo in arquivos_partes:
                        f.write(f"file '{arquivo.absolute()}'\n")

                resultado = subprocess.run([
                    'ffmpeg', '-f', 'concat', '-safe', '0',
                    '-i', str(concat_list_path),
                    '-c', 'copy', '-y', str(output_path)
                ], capture_output=True, text=True, timeout=300)

                if resultado.returncode != 0:
                    logger.error(f"FFmpeg error: {resultado.stderr}")
                    return None, f"❌ Erro ao concatenar"

                concat_list_path.unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return None, "❌ Timeout: Concatenação demorou muito"
        except Exception as e:
            logger.error(f"Erro na concatenação: {e}")
            return None, f"❌ Erro: {str(e)}"

        # Verificar arquivo final
        if not output_path.exists():
            return None, "❌ Erro: Arquivo não foi criado"

        # Ler informações
        try:
            audio_final, sr_final = sf.read(str(output_path))
            duracao = len(audio_final) / sr_final
            minutos = duracao / 60

            logger.info(f"✅ Sucesso! Duração: {minutos:.2f} minutos")
            logger.info(f"📁 Salvo em: {output_path}")

            # Limpar arquivos temporários
            import shutil
            shutil.rmtree(str(temp_dir), ignore_errors=True)

            processing_state['active'] = False
            
            # Retornar caminho do arquivo (Gradio vai ler automaticamente)
            return str(output_path), f"✅ Sucesso! {minutos:.2f} min | Salvo em ./kokoro_audio/{output_filename}"

        except Exception as e:
            logger.error(f"Erro ao ler arquivo: {e}")
            return None, f"❌ Erro: {str(e)}"

    except Exception as e:
        logger.error(f"❌ Erro geral: {str(e)}", exc_info=True)
        processing_state['active'] = False
        return None, f"❌ Erro: {str(e)}"

def parar_processamento():
    """Para o processamento"""
    processing_state['stop_requested'] = True
    return "⏹️ Parando..."

def atualizar_vozes(idioma_display: str):
    """Atualiza as vozes"""
    idioma = IDIOMAS.get(idioma_display, 'a')
    vozes = VOZES_POR_IDIOMA.get(idioma, [])
    return gr.Dropdown(choices=vozes, value=vozes[0] if vozes else None, interactive=True)

# ============================================
# INTERFACE GRADIO
# ============================================

def criar_interface():
    """Cria a interface Gradio"""
    
    with gr.Blocks(title="Kokoro TTS", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ Kokoro TTS - Otimizado")
        gr.Markdown("✅ Salva arquivos localmente em `./kokoro_audio/`")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Texto")
                texto_input = gr.Textbox(
                    label="",
                    placeholder="Digite o texto...",
                    lines=8,
                    max_lines=20,
                    autofocus=True
                )

                gr.Markdown("### 🌍 Idioma")
                idioma_select = gr.Dropdown(
                    choices=list(IDIOMAS.keys()),
                    value=list(IDIOMAS.keys())[0],
                    label="",
                    interactive=True
                )

                gr.Markdown("### 🎤 Voz")
                vozes_iniciais = VOZES_POR_IDIOMA['a']
                voz_select = gr.Dropdown(
                    choices=vozes_iniciais,
                    value=vozes_iniciais[0],
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
                    processar_btn = gr.Button("🎵 Gerar", variant="primary", size="lg", scale=2)
                    parar_btn = gr.Button("⏹️ Parar", variant="stop", size="lg", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Status")
                status_text = gr.Textbox(label="", interactive=False, value="✅ Pronto", lines=2)

                gr.Markdown("### 🔊 Áudio")
                audio_output = gr.Audio(label="", interactive=False)

        # Eventos
        idioma_select.change(fn=atualizar_vozes, inputs=idioma_select, outputs=voz_select)
        processar_btn.click(fn=processar_audio, inputs=[texto_input, idioma_select, voz_select, speed_slider], outputs=[audio_output, status_text])
        parar_btn.click(fn=parar_processamento, outputs=status_text)

    return app

if __name__ == "__main__":
    app = criar_interface()
    app.launch(share=True)
