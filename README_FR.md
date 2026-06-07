# NewAIConcept

Construction d'un système d'IA multimodal de zéro afin de comprendre le fonctionnement des LLM modernes — attention, vision, raisonnement, agents, et réseaux multi-agents — avec MLX sur Apple Silicon.

Chaque phase est autonome : code exécutable + tutoriel interactif montrant ce que le modèle fait réellement en interne.

---

## Prérequis

- Mac avec Apple Silicon (M1/M2/M3/M4)
- Python 3.14 via Homebrew (`/opt/homebrew/bin/python3.14`)
- MLX 0.31+ déjà installé
- Optionnel : `ANTHROPIC_API_KEY` pour les démos en direct des phases 4/5

---

## Carte du Projet

```
NewAIConcept/
├── phase1_transformer/    Modèle de langage — le cœur du transformer
├── phase2_vision/         Encodeur visuel — les images comme tokens
├── phase3_reasoning/      Chaîne de pensée — les tokens de réflexion
├── phase4_agents/         Personas — même modèle, identité différente
└── phase5_multiagent/     Multi-agents + réseau social MCP
```

---

## L'Intuition Fondamentale (à lire en premier)

Tout dans un LLM moderne — raisonnement, vision, personas, utilisation d'outils — se réduit à une seule opération :

> **Prédire le prochain token, conditionné par tous les tokens précédents.**

Un token n'est qu'un nombre représentant un fragment de données : un caractère, un sous-mot, un patch d'image, un code audio, un résidu de protéine. Le transformer ne se soucie pas de ce que le token représente. Il apprend des patterns statistiques entre les tokens à partir des données d'entraînement.

Cela implique :
- **Les données d'entraînement SONT l'ingénierie de features.** Vous définissez ce que le modèle sait par ce que vous mettez dans les données d'entraînement.
- **Les prompts ne sont pas des instructions.** Ce sont des tokens qui activent des patterns que le modèle a déjà appris.
- **Il n'y a pas de module de raisonnement, de module de vision, ou de module de mémoire.** Il n'y a qu'un seul mécanisme — l'attention — appliqué à une séquence de tokens de n'importe quelle source.
- **L'échelle est la seule différence** entre votre MiniGPT et Claude. Même architecture. Nombre différent de paramètres et volume de données d'entraînement.

---

## Phase 1 — Cœur du Transformer

### La question
Comment fonctionne réellement un modèle de langage au niveau mathématique ?

### La réponse
Chaque token est attentif à chaque token précédent, calcule un mélange pondéré de leurs valeurs, met à jour sa propre représentation, et répète cela sur N couches. À la fin, une couche linéaire mappe la représentation de chaque token vers une distribution de probabilité sur le vocabulaire. L'entraînement ajuste les poids pour que le prochain token correct obtienne une probabilité plus élevée. C'est l'intégralité du mécanisme.

### Ce que vous construisez
- `scaled_dot_product_attention` — l'opération fondamentale de tout LLM jamais construit
- `MultiHeadAttention` — H patterns d'attention parallèles, chacun dans un sous-espace de dimension d_model/H
- `TransformerBlock` — attention + MLP feedforward + connexions résiduelles + LayerNorm
- `MiniGPT` — modèle de langage complet au niveau des caractères entraîné sur Shakespeare (~840k paramètres)

### Concepts clés

| Concept | Ce que c'est |
|---|---|
| Embedding | Chaque caractère → vecteur de 128 nombres. Caractères similaires → vecteurs similaires après entraînement |
| Attention | Le token i note chaque token 0→i avec un produit scalaire, applique softmax aux scores, mélange leurs valeurs |
| Multi-head | Division de d_model=128 en 4 têtes × 32 dimensions. Chaque tête apprend différentes relations (syntaxe, position, sémantique) |
| Masque causal | Le token i ne peut pas observer les tokens i+1, i+2… Le futur est masqué à -∞ avant softmax → poids 0 |
| Flux résiduel | Entrée + sortie attention + sortie feedforward. Crée une autoroute de gradient permettant au réseau d'être profond |
| Perte | Entropie croisée : à quel point le modèle est surpris par le prochain caractère correct. log(65) ≈ 4.17 = devinette aléatoire |
| Perplexité | e^perte. "Entre combien de caractères le modèle choisissait-il effectivement ?" Plus bas = meilleur |
| Ablation de tête | Désactiver une tête à la fois. Mesurer l'augmentation de perte. Ce delta = ce que cette tête a contribué |

### Qui décide de n_heads ?
Vous — c'est un hyperparamètre. La seule contrainte : `d_model` doit être divisible par `n_heads`. Avec d_model=128 et n_heads=4, chaque tête obtient 32 dimensions. Plus de têtes = plus de perspectives parallèles mais chaque tête est plus étroite. GPT-2 utilise 12 têtes, GPT-3 en utilise 96. Le modèle apprend la spécialisation de chaque tête à partir des données d'entraînement — vous n'assignez pas de rôles.

### Les dimensions sont des matrices, pas des vecteurs
La représentation d'un seul token est un vecteur (128 nombres). La séquence complète de T tokens est une matrice (T × 128). Les scores d'attention sont une matrice (T × T). Les poids de projection Q, K, V sont des matrices (128 × 128). Lors de l'entraînement avec des batches et des têtes, tout devient un tenseur 4D (batch, têtes, T, T). Les 32 par tête sont la largeur du sous-espace — pas un vecteur unique.

```bash
cd phase1_transformer
python3.14 train.py        # ~5 min sur M1/M2 — entraîne sur Shakespeare, sauvegarde les poids
python3.14 tutorial.py     # interactif, appuyer sur Entrée à chaque section
```

### Sections du tutoriel
1. Embeddings — les caractères comme vecteurs de nombres bruts
2. Carte de chaleur d'attention — visualisation ASCII des relations entre tokens
3. Multi-head — les 4 têtes côte à côte, montrant différents patterns
4. Prédiction du prochain token — diagramme en barres sur tous les 65 caractères
5. Température — basse=confiant/répétitif, haute=créatif/aléatoire
6. Génération aux températures 0.4 / 0.8 / 1.2
7. Profondeur — comment les patterns d'attention changent dans les 4 couches
8. Ablation de tête — désactiver chaque tête, observer la hausse de perte et la dégradation du texte généré

---

## Phase 2 — Encodeur Visuel

### La question
Comment les modèles multimodaux comme Claude et GPT-4V traitent-ils les images ?

### La réponse
Découpez l'image en patches de taille fixe. Aplatissez chaque patch en vecteur. Projetez vers la même dimensionnalité que les tokens texte (d_model). Préfixez ces tokens d'image avant les tokens texte. Alimentez tout dans le même transformer. Les tokens d'image et texte s'observent mutuellement librement — le modèle voit l'image et le texte simultanément à travers un seul mécanisme d'attention.

### Ce que vous construisez
- `PatchEmbedder` — découpe une image 64×64 en patches 8×8 → 64 vecteurs de patch de 256 dimensions
- `MiniVisionEncoder` — Vision Transformer (ViT) : les tokens de patch s'observent sans masque causal
- `VisionLanguageProjector` — deux couches linéaires qui mappent vision_dim (256) → d_model (128)
- `MultiModalMiniGPT` — modèle de langage étendu pour accepter une image comme tokens préfixés

### Concepts clés

| Concept | Ce que c'est |
|---|---|
| Patch | Bloc de 8×8 pixels → 192 nombres aplatis → projetés vers vecteur de 256 dimensions |
| ViT | Vision Transformer — même architecture que le transformer de langage, mais sans masque causal (les patches voient tous les patches) |
| Projecteur | Pont linéaire appris : 256 dimensions vision → 128 dimensions langage. L'unique "couche de traduction" |
| Séquence combinée | `[img_0...img_63, text_0...text_N]` — 64 tokens d'image + N tokens texte, tous traités ensemble |
| Pas de masque causal en vision | Un patch en haut à droite doit voir celui en bas à gauche — pas d'ordre temporel |

### Équivalents dans le monde réel
- **Vision Claude** : backbone CLIP ou ViT similaire → projecteur linéaire → transformer de langage
- **GPT-4V** : même pattern, échelle différente
- **LLaVA** (open source) : CLIP ViT-L/14 → projecteur MLP 2 couches → LLaMA. Exactement ce que vous construisez, en plus grand

```bash
cd phase2_vision
python3.14 tutorial_vision.py     # pas d'entraînement nécessaire — fonctionne sur des images synthétiques
```

---

## Phase 3 — Raisonnement (Chaîne de Pensée)

### La question
Comment un modèle "réfléchit"-il avant de répondre ? Quel est le mécanisme du raisonnement étendu d'Anthropic ?

### La réponse
`<think>` n'est qu'un token. Le modèle a été entraîné sur des données où `<think>` apparaissait avant les étapes de raisonnement et `</think>` avant les réponses correctes. À l'inférence, le modèle reconnaît le pattern : "après `<think>`, générer le raisonnement ; après `</think>`, générer la réponse." Les tokens de réflexion s'accumulent dans la fenêtre de contexte. Quand le modèle génère la réponse finale, il se réfère à son propre bloc-notes. C'est l'intégralité du mécanisme — aucun module de raisonnement séparé.

### Ce que vous construisez
- Jeu de données synthétique : mêmes problèmes en deux formats — réponse directe vs chaîne de pensée
- `model_direct` — entraîné sans tokens de réflexion, saute directement à la réponse
- `model_cot` — entraîné avec `<think>...</think>`, apprend à utiliser le bloc-notes
- Comparaison de précision + analyse d'attention dans le bloc de réflexion

### Concepts clés

| Concept | Ce que c'est |
|---|---|
| Chaîne de pensée | `Q: inverse CAT <think> C-A-T → T-A-C </think> R: TAC` |
| Bloc-notes | Le bloc `<think>` — tokens que le modèle génère comme mémoire de travail avant de répondre |
| Pourquoi ça aide | Les tokens de réflexion entrent dans le contexte. La réponse en est conditionnée. Plus de contexte = meilleures prédictions |
| Température dans le raisonnement | Basse (0.1) pour la réflexion — vous voulez l'étape suivante la plus probable. Plus haute pour la sortie créative finale |
| C'est artificiel | `<think>` est un crochet inventé par l'humain. Le modèle n'a pas de concept de "penser". Il a juste appris que des tokens intermédiaires utiles suivent ce crochet dans les données d'entraînement |
| C'est aussi réel | Les tokens de réflexion changent genuinement la réponse car ils sont dans la fenêtre de contexte. Retirez-les et la précision baisse — prouvé par l'expérience Phase 3 |

```bash
cd phase3_reasoning
python3.14 train_reasoning.py      # ~2 min — entraîne les modèles direct + COT
python3.14 tutorial_reasoning.py   # comparez-les, voyez l'attention, testez la précision
```

---

## Phase 4 — Personas d'Agents

### La question
Comment le même modèle devient-il différents agents avec différentes personnalités et capacités ?

### La réponse
Une persona est un ensemble de tokens préfixés à chaque entrée. Le modèle n'a pas de commutateur de mode interne. Il voit la séquence complète de tokens — préfixe persona + message utilisateur — et génère du texte conditionné sur tout cela. Le préfixe persona active différents patterns appris à partir des données d'entraînement. "Vous êtes un médecin" active des patterns de langage clinique. "Vous êtes un poète" active des patterns de vers. Mêmes poids tout au long.

### Les cinq personas en action (même question : "Devrais-je prendre de l'aspirine pour un mal de tête ?")

```
Assistant Général    → "Oui, 325-650mg avec de l'eau. Éviter si sous anticoagulants."
Conseiller Médical   → "Considérer les diagnostics différentiels d'abord. Éliminer l'HSA pour début soudain..."
Enseignant Socratique → "Quel type de mal de tête ? Avez-vous essayé l'hydratation d'abord ?"
Avocat du Diable     → "La tentation réflexe de l'aspirine ignore la cause profonde. Êtes-vous hydraté ?"
Poète                → "Prenez la petite lune blanche / avec de l'eau / laissez-la dissoudre la tempête."
```

Même modèle. Même question. Cinq réponses. Une seule différence de préfixe.

```bash
cd phase4_agents
python3.14 tutorial_personas.py
ANTHROPIC_API_KEY=sk-... python3.14 tutorial_personas.py   # réponses Claude en direct
```

---

## Phase 5 — Multi-Agents + MCP

### La question
Comment plusieurs agents communiquent-ils, utilisent-ils des outils, et forment-ils un réseau ?

### La réponse
Le Model Context Protocol (MCP) est le standard ouvert d'Anthropic pour connecter les agents aux outils. Un outil n'est qu'une fonction Python enregistrée sur un serveur. Un agent l'appelle en générant un pattern de token structuré. La boucle exécute la fonction, ajoute le résultat au contexte, et rappelle le modèle. Plusieurs agents partageant un bus de messages peuvent déléguer des tâches, collecter des résultats, et synthétiser des réponses — un réseau.

### Pourquoi MCP est important
Avant MCP (2024), chaque entreprise construisait son propre format d'appel d'outil. Claude utilisait un schéma, OpenAI un autre, LangChain un autre. Rien n'était interopérable. MCP standardise trois choses : **Outils** (fonctions que le modèle peut appeler), **Ressources** (données qu'il peut lire), et **Prompts** (templates réutilisables). Tout agent compatible MCP peut utiliser tout serveur compatible MCP. C'est la couche d'infrastructure pour l'écosystème d'agents.

### La topologie du réseau

```
Utilisateur → [Coordinateur]
                  │ send_message → [Chercheur]  → search() → faits
                  │ send_message → [Analyste]    → calculate() → chiffres
                  │ get_messages ←────────────────────────────────────────
                  │ synthèse
Utilisateur ← [Coordinateur] réponse finale
```

```bash
cd phase5_multiagent
python3.14 tutorial_multiagent.py
ANTHROPIC_API_KEY=sk-... python3.14 tutorial_multiagent.py   # réseau d'agents en direct
```

---

## Extension de Modalité

La même architecture s'étend à toute modalité où les données peuvent être tokenisées :

| Modalité | Tokeniseur | Le token représente | Label d'entraînement |
|---|---|---|---|
| Texte | BPE / caractère | sous-mot / char | prochain token |
| Image | Patch + linéaire | patch de pixels 8×8 | description d'image |
| Audio | EnCodec / DAC | ~12ms de son | prochain code audio |
| Musique | EnCodec + tag genre | segment de forme d'onde | prochain segment |
| Vidéo | Patch par frame | patch spatio-temporel | prochaine frame |
| Protéine | Vocabulaire d'acides aminés | un résidu | structure repliée |
| Finance | Intervalles de prix | snapshot OHLCV | prochain intervalle de prix |

---

## Architecture Centrale (partagée entre toutes les phases)

```
Entrée (chars texte, patches d'image, codes audio, ...)
        │
        ▼
  Tokenisation → IDs entiers
        │
        ▼
  Recherche dans table d'embedding + encodage positionnel
  chaque ID de token → vecteur de dimension d_model
        │
        ▼
  ┌─────────────────────────────────────┐
  │   TransformerBlock  × N_COUCHES     │  N couches = N étapes de raisonnement
  │                                     │
  │   LayerNorm(x)                      │
  │       ↓                             │
  │   MultiHeadAttention                │  les tokens s'observent mutuellement
  │   Q = x·Wq,  K = x·Wk,  V = x·Wv  │  H têtes × head_dim = d_model
  │   scores = QKᵀ / √head_dim         │  matrice d'attention (T×T)
  │   weights = softmax(scores + mask)  │  masque causal pour le langage
  │   out = weights · V                 │
  │       ↓ + résiduel                  │
  │   LayerNorm(x)                      │
  │       ↓                             │
  │   FeedForward                       │  chaque token traite indépendamment
  │   x → Linear(d_model→4·d_model)    │
  │     → GELU                          │
  │     → Linear(4·d_model→d_model)    │
  │       ↓ + résiduel                  │
  └─────────────────────────────────────┘
        │
        ▼
  LayerNorm → Linear(d_model → vocab_size)
        │
        ▼
  Logits → softmax → probabilité sur le prochain token
```

C'est l'architecture derrière GPT-2, GPT-3, GPT-4, Claude, Llama, Qwen, Gemini, Mistral. Les dimensions, le nombre de couches et le nombre de têtes diffèrent. La structure ne change pas.

---

## Phase 7 — Architecture de Sentience

La carte du projet ci-dessus couvre les fondations d'ingénierie. La Phase 7 les applique à une question plus difficile : qu'est-ce qu'il faudrait pour donner à un système d'IA une identité unique, persistante et façonnée par l'expérience ?

Trois déficits d'ingénierie dans les LLM actuels motivent le travail :

| Lacune | Problème |
|-----|---------|
| **Unicité** | Chaque instance de modèle est identique octet par octet — "qui est-ce ?" n'a pas de réponse significative |
| **Persistance** | L'identité se réinitialise à la fin de session |
| **Plasticité** | Les poids sont gelés à l'inférence — l'expérience ne peut pas modifier le système |

La solution est une architecture à trois niveaux : **Niveau 1** (neurones à entraînement par labels) · **Niveau 2** (PersonalityConnectome — vecteur unique de 12 traits ensemencé, fichier cerveau JSON persistant, mise à jour hebbienne bornée) · **Niveau 3** (MetaConnectome + InquiryLayer — détection de lacunes typées, fonction d'utilité pondérée par la personnalité pour l'enquête).

**Benchmark : 9.2 / 10 · 13 / 13 tests réussis**

Code complet, livre blanc et suite de benchmark : **[github.com/inaim-finailabz/jarvis-sentience](https://github.com/inaim-finailabz/jarvis-sentience)**

Article (citable) : **[doi.org/10.5281/zenodo.20532665](https://doi.org/10.5281/zenodo.20532665)**

---

## Pourquoi MLX

MLX est le framework ML d'Apple optimisé pour l'architecture mémoire unifiée des puces M-series. Le CPU et le GPU partagent la même mémoire physique — pas de copie de données entre eux. Les modèles qui tiennent en RAM (tous ceux ici) fonctionnent efficacement sans GPU discret.

Le code correspond directement à PyTorch : remplacez `mlx.core` par `torch`, `mlx.nn` par `torch.nn`, ajoutez des appels `.to(device)`. Les concepts sont identiques. MLX est choisi ici parce qu'il fonctionne bien sur le Mac que vous avez déjà.
