#!/usr/bin/env python3
"""
🎵 Qwen3-TTS Studio - Application de synthèse vocale
Interface web moderne avec Gradio
"""
import os
import tempfile
import json
import requests
import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel

# Configuration Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
# Updated with user's available models
OLLAMA_MODELS = [
    "qwen2.5-coder:3b",
    "qwen3:1.7b", 
    "qwen2.5-coder:7b",
    "granite-embedding:278m",
    "nomic-embed-text:latest",
    "qwen2.5-coder:1.5b",
    "granite-embedding:latest",
    "fluffy/l3-8b-stheno-v3.2:latest",
    "deepscaler:latest",
    "qwen2.5-coder:0.5b"
]

# ============================================================
# Configuration
# ============================================================
MODELS = {
    "0.6B Base (Rapide, ~2GB)": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B Base (Qualité, ~4GB)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

LANGUAGES = [
    "French", "English", "Chinese", "Japanese", "Korean",
    "German", "Spanish", "Italian", "Portuguese", "Russian"
]

# Cache du modèle
_cached_model = None
_cached_model_name = None
_cached_device = None

# ============================================================
# Fonctions principales
# ============================================================

def get_device_info():
    """Retourne les informations sur les devices disponibles"""
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    devices = []
    if cuda_available:
        devices.append("GPU NVIDIA (CUDA)")
    if mps_available:
        devices.append("GPU Apple Silicon (MPS)")
    devices.append("CPU (Stable)")
    
    return devices


def load_model(model_name: str, device_choice: str):
    """Charge le modèle avec mise en cache"""
    global _cached_model, _cached_model_name, _cached_device
    
    # Déterminer le device
    if "MPS" in device_choice:
        device = "mps"
        dtype = torch.float32  # float32 plus stable sur MPS
    elif "CUDA" in device_choice:
        device = "cuda"
        dtype = torch.float16 # float16 pour CUDA (plus rapide, moins de VRAM)
    else:
        device = "cpu"
        dtype = torch.float32
    
    model_path = MODELS.get(model_name, list(MODELS.values())[0])
    
    # Vérifier si on peut réutiliser le cache
    if (_cached_model is not None and 
        _cached_model_name == model_path and 
        _cached_device == device):
        return _cached_model, "✅ Modèle déjà en cache!"
    
    # Charger le nouveau modèle
    try:
        _cached_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype
        )
        _cached_model_name = model_path
        _cached_device = device
        return _cached_model, f"✅ Modèle chargé sur {device.upper()}!"
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"


def generate_voice_clone(
    model_name: str,
    device: str,
    text: str,
    language: str,
    ref_audio,
    ref_text: str,
    progress=gr.Progress()
):
    """Génère de l'audio avec Voice Clone"""
    if not text.strip():
        return None, "❌ Veuillez entrer un texte à synthétiser."
    
    if ref_audio is None:
        return None, "❌ Veuillez fournir un audio de référence."
    
    progress(0.2, desc="Chargement du modèle...")
    model, status = load_model(model_name, device)
    if model is None:
        return None, status
    
    progress(0.5, desc="Génération de l'audio...")
    
    try:
        # Gérer les différents formats d'audio de gradio
        if isinstance(ref_audio, tuple):
            sr_ref, audio_data = ref_audio
            # Sauvegarder temporairement
            temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_ref.name, audio_data, sr_ref)
            ref_audio_path = temp_ref.name
        else:
            ref_audio_path = ref_audio
        
        # Génération - utiliser x_vector_only_mode si pas de transcription
        has_ref_text = ref_text and ref_text.strip()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text if has_ref_text else None,
            x_vector_only_mode=not has_ref_text,  # Mode x-vector si pas de transcription
            non_streaming_mode=True
        )
        
        progress(0.9, desc="Sauvegarde...")
        
        # Sauvegarder le résultat
        output_path = os.path.join(tempfile.gettempdir(), "qwen_tts_output.wav")
        sf.write(output_path, wavs[0], sr)
        
        duration = len(wavs[0]) / sr
        return output_path, f"✅ Audio généré! Durée: {duration:.2f}s"
        
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"


def generate_voice_design(
    model_name: str,
    device: str,
    text: str,
    language: str,
    voice_prompt: str,
    progress=gr.Progress()
):
    """Génère de l'audio avec Voice Design (description de voix)"""
    if not text.strip():
        return None, "❌ Veuillez entrer un texte à synthétiser."
    
    if not voice_prompt.strip():
        return None, "❌ Veuillez décrire la voix souhaitée."
    
    progress(0.2, desc="Chargement du modèle...")
    model, status = load_model(model_name, device)
    if model is None:
        return None, status
    
    progress(0.5, desc="Génération de l'audio...")
    
    try:
        # Voice Design nécessite un modèle Instruct
        # Mais comme on utilise Base, on fait un fallback avec une voix par défaut
        # et on utilise le prompt comme contexte
        
        # Créer un audio de référence par défaut si nécessaire
        ref_audio_path = os.path.join(os.path.dirname(__file__), "ref_voice.wav")
        if not os.path.exists(ref_audio_path):
            os.system(f'say -v Thomas "Ceci est une voix de référence" -o /tmp/ref.aiff && '
                     f'afconvert -f WAVE -d LEI16 /tmp/ref.aiff {ref_audio_path}')
        
        # Génération avec la voix de référence
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text="Ceci est une voix de référence",
            non_streaming_mode=True
        )
        
        progress(0.9, desc="Sauvegarde...")
        
        output_path = os.path.join(tempfile.gettempdir(), "qwen_tts_output.wav")
        sf.write(output_path, wavs[0], sr)
        
        duration = len(wavs[0]) / sr
        return output_path, f"✅ Audio généré! Durée: {duration:.2f}s\n⚠️ Note: Voice Design complet nécessite un modèle Instruct."
        
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"


def chat_with_ollama(
    model_name: str,
    device: str,
    user_prompt: str,
    system_prompt: str,
    ollama_model: str,
    language: str,
    chat_history: list,
    custom_ref_audio=None,
    progress=gr.Progress()
):
    """
    Envoie un prompt à Ollama, récupère la réponse et la convertit en audio.
    """
    if not user_prompt.strip():
        return None, chat_history, "❌ Veuillez entrer un message."
    
    # Ajouter le message utilisateur à l'historique
    chat_history = chat_history or []
    chat_history.append(("Vous", user_prompt))
    
    progress(0.1, desc="Envoi à Ollama...")
    
    try:
        # Construire le prompt complet
        full_prompt = ""
        if system_prompt.strip():
            full_prompt = f"System: {system_prompt}\n\n"
        
        # Ajouter l'historique
        for role, content in chat_history:
            prefix = "User" if role == "Vous" else "Assistant"
            full_prompt += f"{prefix}: {content}\n"
        full_prompt += "Assistant:"
        
        # Appel à Ollama
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": ollama_model,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code != 200:
            return None, chat_history, f"❌ Erreur Ollama: {response.status_code}"
        
        result = response.json()
        ai_response = result.get("response", "").strip()
        
        if not ai_response:
            return None, chat_history, "❌ Ollama n'a pas retourné de réponse."
        
        # Ajouter la réponse à l'historique
        chat_history.append(("🤖 IA", ai_response))
        
        progress(0.4, desc="Chargement du modèle TTS...")
        
        # Charger le modèle TTS
        model, status = load_model(model_name, device)
        if model is None:
            return None, chat_history, status
        
        progress(0.6, desc="Génération de l'audio...")
        
        # Gérer l'audio de référence
        if custom_ref_audio is not None:
            # Utiliser l'audio personnalisé fourni par l'utilisateur
            if isinstance(custom_ref_audio, tuple):
                sr_ref, audio_data = custom_ref_audio
                temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(temp_ref.name, audio_data, sr_ref)
                ref_audio_path = temp_ref.name
            else:
                ref_audio_path = custom_ref_audio
        else:
            # Créer audio de référence par défaut si nécessaire
            ref_audio_path = os.path.join(os.path.dirname(__file__), "ref_voice.wav")
            if not os.path.exists(ref_audio_path):
                os.system(f'say -v Thomas "Bonjour je suis une voix de test" -o /tmp/ref.aiff && '
                         f'afconvert -f WAVE -d LEI16 /tmp/ref.aiff {ref_audio_path}')
        
        # Génération TTS avec x_vector_only_mode pour éviter la pollution du ref_text
        wavs, sr = model.generate_voice_clone(
            text=ai_response,
            language=language,
            ref_audio=ref_audio_path,
            x_vector_only_mode=True,  # Utilise seulement le timbre, pas le contenu
            non_streaming_mode=True
        )
        
        progress(0.9, desc="Sauvegarde...")
        
        output_path = os.path.join(tempfile.gettempdir(), "ollama_response.wav")
        sf.write(output_path, wavs[0], sr)
        
        duration = len(wavs[0]) / sr
        return output_path, chat_history, f"✅ Réponse générée! ({duration:.1f}s)"
        
    except requests.exceptions.ConnectionError:
        return None, chat_history, "❌ Impossible de se connecter à Ollama. Lancez 'ollama serve' dans un terminal."
    except Exception as e:
        return None, chat_history, f"❌ Erreur: {str(e)}"


# ============================================================
# Interface Gradio
# ============================================================

# CSS personnalisé pour un look moderne
custom_css = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.config-section {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px;
    padding: 1rem;
}

.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

.tab-nav button {
    font-weight: 600 !important;
}

.output-audio {
    border: 2px solid #667eea;
    border-radius: 12px;
    padding: 1rem;
}
"""

# Créer l'interface
with gr.Blocks(
    title="🎵 Qwen3-TTS Studio",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=custom_css
) as app:
    
    # En-tête
    gr.HTML("""
        <h1 class="main-title">🎵 Qwen3-TTS Studio</h1>
        <p class="subtitle">Synthèse vocale avancée avec intelligence artificielle</p>
    """)
    
    # Configuration globale
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="🤖 Modèle",
                info="Choisissez le modèle TTS"
            )
        with gr.Column(scale=1):
            device_dropdown = gr.Dropdown(
                choices=get_device_info(),
                value=get_device_info()[0],
                label="⚡ Device",
                info="CPU (stable) ou GPU (rapide)"
            )
        with gr.Column(scale=1):
            language_dropdown = gr.Dropdown(
                choices=LANGUAGES,
                value="French",
                label="🌍 Langue",
                info="Langue du texte"
            )
    
    gr.Markdown("---")
    
    # Onglets pour les modes
    with gr.Tabs():
        
        # === Onglet Voice Clone ===
        with gr.TabItem("🎤 Voice Clone", id="clone"):
            gr.Markdown("""
            ### Cloner une voix
            Fournissez un échantillon audio et sa transcription pour cloner la voix.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Option 1: Upload fichier
                    ref_audio_upload = gr.Audio(
                        label="📁 Importer un fichier audio",
                        type="filepath",
                        sources=["upload"]
                    )
                    
                    # Option 2: Enregistrer
                    ref_audio_record = gr.Audio(
                        label="🎙️ Ou enregistrer votre voix",
                        type="numpy",
                        sources=["microphone"]
                    )
                    
                    ref_text_input = gr.Textbox(
                        label="📝 Transcription (optionnel mais recommandé)",
                        placeholder="Tapez ce qui est dit dans l'audio de référence...",
                        lines=2
                    )
                
                with gr.Column(scale=1):
                    text_clone_input = gr.Textbox(
                        label="✍️ Texte à synthétiser",
                        placeholder="Entrez le texte que vous voulez faire dire...",
                        lines=5
                    )
                    
                    clone_btn = gr.Button(
                        "🚀 Générer l'audio",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"]
                    )
            
            with gr.Row():
                clone_output = gr.Audio(
                    label="🔊 Audio généré",
                    type="filepath",
                    elem_classes=["output-audio"]
                )
                clone_status = gr.Textbox(
                    label="📋 Statut",
                    interactive=False
                )
        
        # === Onglet Voice Design ===
        with gr.TabItem("✨ Voice Design", id="design"):
            gr.Markdown("""
            ### Créer une voix personnalisée
            Décrivez la voix que vous souhaitez en langage naturel.
            
            > ⚠️ **Note**: Cette fonctionnalité complète nécessite un modèle Instruct. 
            > Avec le modèle Base, une voix par défaut sera utilisée.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    voice_prompt_input = gr.Textbox(
                        label="🎭 Description de la voix",
                        placeholder="Ex: Une voix féminine douce et chaleureuse, avec un léger accent du sud...",
                        lines=4
                    )
                    
                    # Exemples de prompts
                    gr.Examples(
                        examples=[
                            ["Une voix masculine grave et posée, comme un narrateur de documentaire"],
                            ["Une voix féminine joyeuse et dynamique, comme une animatrice radio"],
                            ["Une voix douce et apaisante, parfaite pour la méditation"],
                            ["A deep male voice with a British accent, formal and elegant"],
                        ],
                        inputs=voice_prompt_input,
                        label="💡 Exemples de prompts"
                    )
                
                with gr.Column(scale=1):
                    text_design_input = gr.Textbox(
                        label="✍️ Texte à synthétiser",
                        placeholder="Entrez le texte que vous voulez faire dire...",
                        lines=5
                    )
                    
                    design_btn = gr.Button(
                        "🚀 Générer l'audio",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"]
                    )
            
            with gr.Row():
                design_output = gr.Audio(
                    label="🔊 Audio généré",
                    type="filepath",
                    elem_classes=["output-audio"]
                )
                design_status = gr.Textbox(
                    label="📋 Statut",
                    interactive=False
                )
        
        # === Onglet Ollama Chat ===
        with gr.TabItem("🦙 Parler avec Ollama", id="ollama"):
            gr.Markdown("""
            ### Discutez avec une IA et écoutez ses réponses
            Envoyez un message à Ollama, la réponse sera convertie en audio.
            
            > 💡 **Astuce**: Assurez-vous qu'Ollama est lancé (`ollama serve` dans un terminal).
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    ollama_model_dropdown = gr.Dropdown(
                        choices=OLLAMA_MODELS,
                        value="qwen3:0.6b",
                        label="🦙 Modèle Ollama",
                        info="Modèle de langage à utiliser",
                        allow_custom_value=True
                    )
                    
                    system_prompt_input = gr.Textbox(
                        label="🎭 System Prompt",
                        placeholder="Ex: Tu es un assistant amical qui répond en français de manière concise...",
                        value="Tu es un assistant vocal amical. Réponds de manière concise et naturelle en français, comme si tu parlais à quelqu'un. Limite tes réponses à 2-3 phrases maximum.",
                        lines=3
                    )
                    
                    # Upload de voix personnalisée
                    ollama_ref_audio = gr.Audio(
                        label="🎤 Voix personnalisée (optionnel - glissez un fichier ou enregistrez)",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    
                    clear_chat_btn = gr.Button(
                        "🗑️ Effacer la conversation",
                        variant="secondary"
                    )
                
                with gr.Column(scale=2):
                    # Historique de chat
                    chat_history_state = gr.State([])
                    
                    chatbox = gr.Chatbot(
                        label="💬 Conversation",
                        height=300
                    )
                    
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="✍️ Votre message",
                            placeholder="Tapez votre message ici...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button(
                            "📤 Envoyer",
                            variant="primary",
                            scale=1,
                            elem_classes=["generate-btn"]
                        )
            
            with gr.Row():
                ollama_audio_output = gr.Audio(
                    label="🔊 Réponse audio",
                    type="filepath",
                    autoplay=True,  # Lecture automatique !
                    elem_classes=["output-audio"]
                )
                ollama_status = gr.Textbox(
                    label="📋 Statut",
                    interactive=False
                )
    
    # Pied de page
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        <p>🚀 Propulsé par <strong>Qwen3-TTS</strong> d'Alibaba | 🦙 <strong>Ollama</strong> | 🍎 Optimisé pour Mac Apple Silicon</p>
    </div>
    """)
    
    # === Événements ===
    
    # Voice Clone - gérer les deux types d'entrée audio
    def handle_clone(model, device, text, lang, audio_upload, audio_record, ref_text, progress=gr.Progress()):
        # Priorité à l'enregistrement si disponible
        if audio_record is not None:
            return generate_voice_clone(model, device, text, lang, audio_record, ref_text, progress)
        elif audio_upload is not None:
            return generate_voice_clone(model, device, text, lang, audio_upload, ref_text, progress)
        else:
            return None, "❌ Veuillez importer ou enregistrer un audio de référence."
    
    clone_btn.click(
        fn=handle_clone,
        inputs=[
            model_dropdown, device_dropdown, text_clone_input, language_dropdown,
            ref_audio_upload, ref_audio_record, ref_text_input
        ],
        outputs=[clone_output, clone_status]
    )
    
    # Voice Design
    design_btn.click(
        fn=generate_voice_design,
        inputs=[
            model_dropdown, device_dropdown, text_design_input, 
            language_dropdown, voice_prompt_input
        ],
        outputs=[design_output, design_status]
    )
    
    # Ollama Chat
    def handle_ollama_chat(model, device, user_msg, sys_prompt, ollama_model, lang, history, ref_audio):
        audio, new_history, status = chat_with_ollama(
            model, device, user_msg, sys_prompt, ollama_model, lang, history, ref_audio
        )
        # Formater l'historique pour le chatbot Gradio 6 (format messages)
        formatted_history = []
        for msg in new_history:
            role = "user" if msg[0] == "Vous" else "assistant"
            formatted_history.append({"role": role, "content": msg[1]})
        return audio, new_history, formatted_history, status, ""
    
    send_btn.click(
        fn=handle_ollama_chat,
        inputs=[
            model_dropdown, device_dropdown, user_input, system_prompt_input,
            ollama_model_dropdown, language_dropdown, chat_history_state, ollama_ref_audio
        ],
        outputs=[ollama_audio_output, chat_history_state, chatbox, ollama_status, user_input]
    )
    
    # Aussi envoyer avec Entrée
    user_input.submit(
        fn=handle_ollama_chat,
        inputs=[
            model_dropdown, device_dropdown, user_input, system_prompt_input,
            ollama_model_dropdown, language_dropdown, chat_history_state, ollama_ref_audio
        ],
        outputs=[ollama_audio_output, chat_history_state, chatbox, ollama_status, user_input]
    )
    
    # Effacer la conversation
    def clear_conversation():
        return [], [], None, "🗑️ Conversation effacée."
    
    clear_chat_btn.click(
        fn=clear_conversation,
        outputs=[chat_history_state, chatbox, ollama_audio_output, ollama_status]
    )


# ============================================================
# Lancement
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎵 Qwen3-TTS Studio")
    print("=" * 60)
    print(f"\n📱 Devices disponibles: {get_device_info()}")
    print(f"🤖 Modèles: {list(MODELS.keys())}")
    print("\n🌐 Démarrage de l'interface web...")
    print("   L'application s'ouvrira automatiquement dans votre navigateur.\n")
    
    app.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )
