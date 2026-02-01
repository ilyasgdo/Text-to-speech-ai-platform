#!/usr/bin/env python3
"""
Qwen3-TTS - Script de synthèse vocale avancé
Utilise le modèle 1.7B avec support Voice Design et Voice Clone
"""
import os
import sys
import argparse
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # Modèle plus puissant
DEVICE = "cpu"  # CPU pour la stabilité (MPS peut avoir des problèmes)
DTYPE = torch.float32
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Chargement du modèle (singleton)
# ============================================================
_model = None

def get_model():
    """Charge le modèle une seule fois (lazy loading)"""
    global _model
    if _model is None:
        print(f"📥 Chargement du modèle {MODEL_NAME}...")
        print("   (Cette opération peut prendre quelques minutes)")
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE,
            torch_dtype=DTYPE
        )
        print("   ✅ Modèle chargé!")
    return _model

# ============================================================
# Fonctions principales
# ============================================================

def text_to_speech(
    text: str,
    output_file: str = "output.wav",
    language: str = "French",
    ref_audio: str = None,
    ref_text: str = None
) -> str:
    """
    Convertit du texte en parole.
    
    Args:
        text: Le texte à convertir en audio
        output_file: Nom du fichier de sortie (.wav)
        language: Langue du texte (French, English, Chinese, etc.)
        ref_audio: Chemin vers un audio de référence pour cloner la voix
        ref_text: Transcription de l'audio de référence
    
    Returns:
        Chemin vers le fichier audio généré
    """
    model = get_model()
    
    # Si pas d'audio de référence, créer un audio par défaut avec macOS
    if ref_audio is None:
        ref_audio = os.path.join(OUTPUT_DIR, "ref_voice.wav")
        ref_text = "Bonjour, ceci est un test de synthèse vocale"
        
        if not os.path.exists(ref_audio):
            print("   📢 Création d'un audio de référence...")
            os.system(f'say -v Thomas "{ref_text}" -o /tmp/ref_voice.aiff && '
                     f'afconvert -f WAVE -d LEI16 /tmp/ref_voice.aiff {ref_audio}')
    
    print(f"🎙️ Génération de la parole...")
    print(f"   Texte: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"   Langue: {language}")
    
    # Génération avec Voice Clone
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text or "",
        non_streaming_mode=True
    )
    
    # Sauvegarder
    output_path = os.path.join(OUTPUT_DIR, output_file)
    sf.write(output_path, wavs[0], sr)
    
    duration = len(wavs[0]) / sr
    print(f"   ✅ Audio généré: {output_file}")
    print(f"   ⏱️ Durée: {duration:.2f}s | 🔊 {sr} Hz")
    
    return output_path


def speak(text: str, language: str = "French"):
    """
    Génère et joue immédiatement le texte.
    
    Args:
        text: Le texte à dire
        language: La langue du texte
    """
    output_file = text_to_speech(text, "spoken.wav", language)
    os.system(f'open "{output_file}"')


# ============================================================
# Interface en ligne de commande
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS - Synthèse vocale avancée",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python qwen_tts_app.py "Bonjour le monde"
  python qwen_tts_app.py "Hello world" -l English -o hello.wav
  python qwen_tts_app.py "Texte" --ref-audio ma_voix.wav --ref-text "transcription"
        """
    )
    
    parser.add_argument("text", help="Texte à convertir en parole")
    parser.add_argument("-o", "--output", default="output.wav", 
                       help="Fichier de sortie (défaut: output.wav)")
    parser.add_argument("-l", "--language", default="French",
                       help="Langue: French, English, Chinese, Japanese, etc.")
    parser.add_argument("--ref-audio", help="Audio de référence pour cloner la voix")
    parser.add_argument("--ref-text", help="Transcription de l'audio de référence")
    parser.add_argument("--play", action="store_true", help="Jouer l'audio après génération")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎵 Qwen3-TTS - Synthèse vocale")
    print("=" * 60)
    
    output_path = text_to_speech(
        text=args.text,
        output_file=args.output,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text
    )
    
    if args.play:
        print("\n🔊 Lecture de l'audio...")
        os.system(f'open "{output_path}"')
    
    print("\n" + "=" * 60)
    print("✨ Terminé!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Mode interactif si aucun argument
        print("=" * 60)
        print("🎵 Qwen3-TTS - Mode interactif")
        print("=" * 60)
        
        text = input("\n📝 Entrez le texte à convertir: ")
        if text.strip():
            speak(text)
        else:
            print("❌ Aucun texte fourni.")
    else:
        main()
