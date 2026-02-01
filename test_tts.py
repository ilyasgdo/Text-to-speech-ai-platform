#!/usr/bin/env python3
"""
Script de test pour Qwen3-TTS sur Mac (Apple Silicon)
Utilise generate_voice_clone avec un fichier audio local
"""
import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

print("=" * 60)
print("Qwen3-TTS Test sur Mac")
print("=" * 60)

# Vérifier le support MPS (Metal Performance Shaders pour Mac)
print(f"\n🔍 Vérification du matériel...")
print(f"   - PyTorch version: {torch.__version__}")
print(f"   - MPS disponible: {torch.backends.mps.is_available()}")
print(f"   - MPS built: {torch.backends.mps.is_built()}")

# Utiliser CPU pour éviter les problèmes de précision numérique avec MPS
# MPS peut avoir des problèmes avec les modèles TTS et float16
device = "cpu"
print(f"   ℹ️ Utilisation du CPU (plus stable pour TTS)")

print(f"\n📥 Chargement du modèle Qwen3-TTS (0.6B Base)...")
print("   Cela peut prendre quelques minutes au premier lancement...")

try:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map=device,
        torch_dtype=torch.float32  # float32 pour éviter les erreurs de précision
    )
    print("   ✅ Modèle chargé avec succès!")
    
except Exception as e:
    print(f"   ❌ Erreur lors du chargement: {e}")
    raise

print(f"\n🎙️ Génération de la parole avec Voice Clone...")

# Texte de test en français
text_fr = "Bonjour ilyas comment vas tu je m'appelle anais et j'aime la glace "

# Audio de référence local (créé avec macOS say)
ref_audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_voice.wav")
# Transcription de l'audio de référence (correspond à ce que "say" a généré)
ref_text = "Bonjour, ceci est un test de synthèse vocale"

if not os.path.exists(ref_audio_path):
    print(f"   ⚠️ Audio de référence non trouvé, création en cours...")
    os.system(f'say -v Thomas "{ref_text}" -o /tmp/ref_voice.aiff && afconvert -f WAVE -d LEI16 /tmp/ref_voice.aiff {ref_audio_path}')

print(f"   Audio de référence: {ref_audio_path}")
print(f"   Transcription: {ref_text}")

try:
    # Génération avec Voice Clone en utilisant l'audio local et sa transcription
    wavs, sr = model.generate_voice_clone(
        text=text_fr,
        language="French",
        ref_audio=ref_audio_path,
        ref_text=ref_text,  # Transcription de l'audio de référence (obligatoire)
        non_streaming_mode=True
    )
    
    # Sauvegarder le résultat
    output_file = "output_test.wav"
    sf.write(output_file, wavs[0], sr)
    print(f"   ✅ Audio généré avec succès!")
    print(f"   📁 Fichier: {output_file}")
    print(f"   ⏱️ Durée: {len(wavs[0])/sr:.2f} secondes")
    print(f"   🔊 Fréquence: {sr} Hz")
    
except Exception as e:
    print(f"   ❌ Erreur lors de la génération: {e}")
    import traceback
    traceback.print_exc()
    raise

print(f"\n" + "=" * 60)
print("✨ Test terminé avec succès!")
print("=" * 60)
print(f"\n🎧 Pour écouter le résultat:")
print(f"   open {output_file}")
