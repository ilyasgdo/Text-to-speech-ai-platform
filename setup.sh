#!/bin/bash

# =============================================================================
# 🎵 Qwen3-TTS Studio - Installation & Lancement
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "🎵 Qwen3-TTS Studio - Installation"
echo "=============================================="

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier Python
echo -e "\n${YELLOW}📦 Vérification de Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python $PYTHON_VERSION trouvé${NC}"
else
    echo -e "${RED}❌ Python3 non trouvé. Veuillez installer Python 3.12+${NC}"
    exit 1
fi

# Créer l'environnement virtuel si nécessaire
if [ ! -d ".venv" ]; then
    echo -e "\n${YELLOW}🔧 Création de l'environnement virtuel...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✅ Environnement virtuel créé${NC}"
else
    echo -e "${GREEN}✅ Environnement virtuel existe déjà${NC}"
fi

# Activer l'environnement virtuel
echo -e "\n${YELLOW}🔌 Activation de l'environnement virtuel...${NC}"
source .venv/bin/activate

# Installer les dépendances
echo -e "\n${YELLOW}📥 Installation des dépendances...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Vérifier si Ollama est installé (optionnel)
echo -e "\n${YELLOW}🦙 Vérification d'Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama est installé${NC}"
    echo -e "${YELLOW}💡 Pour utiliser l'onglet Ollama, lancez: ollama serve${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama n'est pas installé (optionnel)${NC}"
    echo -e "   Pour l'installer: brew install ollama"
fi

echo -e "\n${GREEN}=============================================="
echo -e "✅ Installation terminée!"
echo -e "==============================================${NC}"

echo -e "\n${YELLOW}🚀 Lancement de l'application...${NC}\n"

# Lancer l'application
python qwen_tts_studio.py
