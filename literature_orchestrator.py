import os
import requests
import time
from google import genai
from google.genai import types
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse

# Initialisation du client Gemini
client = genai.Client()

def generate_with_retry(prompt, system_instruction, max_retries=4):
    """Fonction utilitaire pour appeler Gemini avec gestion des erreurs 503 (Exponential Backoff)."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2 # Température basse pour des réponses très analytiques et factuelles
    )
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro", # Modèle plus puissant et spécifique
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                wait_time = (2 ** attempt) * 2
                print(f"⏳ Serveur occupé. Nouvelle tentative dans {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    print("❌ API surchargée. Échec de l'agent.")
    return None

def agent_documentaliste(query: str, limit: int = 15):
    """Agent 1 : Interroge arXiv et Semantic Scholar avec gestion des limites de débit."""
    print(f"\n🔎 [AGENT 1 - DOCUMENTALISTE] Recherche pour : {query}")
    
    def search_ss(q):
        print(f"📡 Recherche sur Semantic Scholar ({q})...")
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": q, "limit": limit, "fields": "title,abstract,year,authors,url"}
        headers = {"User-Agent": "Mozilla/5.0"}
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=20)
                if response.status_code == 200:
                    return response.json().get('data', [])
                elif response.status_code == 429:
                    wait = (attempt + 1) * 5
                    print(f"⏳ Rate limit (429). Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"⚠️ Code erreur : {response.status_code}")
                    break
            except Exception as e:
                print(f"⚠️ Erreur SS : {e}")
        return []

    # Tentative avec la requête originale
    papers = search_ss(query)
    
    # Si échec, tentative avec une requête simplifiée
    if not papers:
        simple_query = "additive manufacturing thermal PINN analytical"
        print("💡 Tentative avec une requête simplifiée...")
        papers = search_ss(simple_query)

    if papers:
        papers_text = ""
        for idx, p in enumerate(papers):
            if not p.get('abstract'): continue
            authors = ", ".join([a['name'] for a in p.get('authors', [])[:3]])
            papers_text += f"ID: SS-{idx+1}\n"
            papers_text += f"Titre: {p.get('title')}\n"
            papers_text += f"Auteurs: {authors} et al. ({p.get('year')})\n"
            papers_text += f"URL: {p.get('url')}\n"
            papers_text += f"Abstract: {p.get('abstract')}\n\n"
        print(f"✅ {len(papers)} articles trouvés.")
        return papers_text

    print("📡 Repli sur arXiv...")
    # Simplification pour arXiv
    query_arxiv = query.replace('all:', '').replace('"', '')
    params_arxiv = {'search_query': query_arxiv, 'start': 0, 'max_results': limit}
    url_arxiv = "http://export.arxiv.org/api/query"
    
    try:
        response = requests.get(url_arxiv, params=params_arxiv, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            
            papers_text = ""
            for idx, entry in enumerate(entries):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                year = entry.find('atom:published', ns).text[:4]
                url = entry.find('atom:id', ns).text
                authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)][:3]
                papers_text += f"ID: AR-{idx+1}\n"
                papers_text += f"Titre: {title}\n"
                papers_text += f"Auteurs: {', '.join(authors)} et al. ({year})\n"
                papers_text += f"URL: {url}\n"
                papers_text += f"Abstract: {summary}\n\n"
            
            if papers_text:
                print(f"✅ {len(entries)} articles trouvés sur arXiv.")
                return papers_text
        elif response.status_code == 429:
            print("❌ arXiv Rate limit (429) atteint.")
        else:
            print(f"❌ Erreur arXiv Code: {response.status_code}")
    except Exception as e:
        print(f"❌ Échec total de la recherche arXiv : {e}")

    return ""

def agent_evaluateur(papers_text: str, block_name: str):
    """Agent 2 : Filtre les articles selon les contraintes MDAO/Gradients."""
    print(f"\n🧠 [AGENT 2 - ÉVALUATEUR] Lecture et filtrage pour {block_name}...")
    
    sys_inst = (
        "Tu es un ingénieur expert en MDAO (Multidisciplinary Design Analysis and Optimization). "
        "Ta tâche est de lire une liste d'articles scientifiques et de sélectionner UNIQUEMENT "
        "ceux dont les approches mathématiques ou physiques (modèles analytiques, physiques, ou de substitution "
        "comme les PINNs) sont différentiables et peuvent fournir des gradients par rapport aux paramètres d'entrée."
    )
    
    prompt = (
        f"Voici une liste d'articles récupérés pour le '{block_name}' :\n\n"
        f"{papers_text}\n\n"
        "Sélectionne les 3 ou 4 articles les plus pertinents pour créer une discipline GEMSEO basée sur les gradients.\n"
        "Pour chaque article retenu, fournis son Titre, son URL, et justifie explicitement pourquoi son approche "
        "est compatible avec une optimisation basée sur les gradients."
    )
    
    return generate_with_retry(prompt, sys_inst)

def agent_synthetiseur(filtered_papers: str, block_name: str):
    """Agent 3 : Rédige le rapport exécutif en Français."""
    print(f"\n✍️ [AGENT 3 - SYNTHÉTISEUR] Rédaction du rapport pour le {block_name}...")
    
    sys_inst = (
        "Tu es un chercheur académique sénior rédigeant une revue de littérature pour une architecture MDAO sous GEMSEO."
    )
    
    prompt = (
        f"À partir de cette sélection d'articles filtrés par ton collègue :\n\n{filtered_papers}\n\n"
        f"Rédige un rapport bibliographique formel en Français pour le '{block_name}'.\n"
        "Le rapport doit inclure :\n"
        "1. Une introduction sur le défi de ce bloc spécifique dans le contexte de l'ALM.\n"
        "2. Une synthèse des approches de l'état de l'art (SOTA) identifiées dans ces articles (avec citations et liens).\n"
        "3. Une conclusion recommandant la meilleure architecture mathématique différentiable à implémenter en Python/GEMSEO.\n"
        "Formatte le texte en Markdown propre."
    )
    
    return generate_with_retry(prompt, sys_inst)

if __name__ == "__main__":
    print("🚀 Lancement du Research Swarm pour l'architecture MDAO...")
    
    # Définition des 5 Blocs MDAO
    blocks = [
        {
            "name": "Bloc 1 : Simulation du Procédé (Thermique et État Matière)",
            "query": "additive manufacturing thermal history melt pool analytical PINN",
            "filename": "rapport_bloc1_procede.md"
        },
        {
            "name": "Bloc 2 : Prédiction des Propriétés Matériaux et Défauts",
            "query": "additive manufacturing porosity defect mechanical properties prediction differentiable",
            "filename": "rapport_bloc2_materiaux.md"
        },
        {
            "name": "Bloc 3 : Prédiction de l'État de Surface",
            "query": "additive manufacturing surface roughness prediction analytical surrogate gradient",
            "filename": "rapport_bloc3_surface.md"
        },
        {
            "name": "Bloc 4 : Simulation Opérationnelle (Durée de vie et Fatigue)",
            "query": "fatigue life prediction additive manufacturing components structural optimization gradient",
            "filename": "rapport_bloc4_fatigue.md"
        },
        {
            "name": "Bloc 5 : Formulation d'Optimisation et Figures de Mérite",
            "query": "multidisciplinary design optimization MDAO additive manufacturing formulation gradient based",
            "filename": "rapport_bloc5_optimisation.md"
        }
    ]
    
    for block in blocks:
        print(f"\n\n{'='*70}\nTraitement du {block['name']}\n{'='*70}")
        
        # 1. Recherche
        raw_papers = agent_documentaliste(block['query'], limit=15)
        
        if raw_papers:
            # 2. Évaluation (Filtrage différentiable)
            filtered_papers = agent_evaluateur(raw_papers, block['name'])
            
            if filtered_papers:
                # 3. Synthèse
                final_report = agent_synthetiseur(filtered_papers, block['name'])
                
                if final_report:
                    with open(block['filename'], "w", encoding="utf-8") as f:
                        f.write(final_report)
                    print(f"\n✅ Terminé ! Le rapport a été sauvegardé dans : {block['filename']}")
                else:
                    print(f"❌ Échec de la synthèse pour {block['name']}")
            else:
                print(f"❌ Échec de l'évaluation pour {block['name']}")
        else:
            print(f"❌ Aucun article pertinent trouvé pour {block['name']}")