"""Logique d'inférence pour la classification de sentiment.

Le modèle CamemBERT chargé sort des labels 5 étoiles
(`'1 star'`, `'2 stars'`, ..., `'5 stars'`). Le métier (Aubergine Hôtels)
veut 3 classes (`négatif`, `neutre`, `positif`).

→ Ton travail dans ce fichier :
   1. Implémenter `predict_sentiment()` (récupère scores 5★ + appelle map).
   2. Implémenter `map_stars_to_sentiment()` (mapping 5★ → 3 classes).
   3. Justifier le seuil retenu en commentaire (cf. brief).

Le pipeline transformers est chargé une seule fois au démarrage, dans
`main.py` (lifespan), et stocké dans `state["pipeline"]`. Tu le récupères
en argument.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger
from numpy import mean

from app.schemas import Sentiment, SentimentOut


def map_stars_to_sentiment(sentiment_5_classes: list) -> str:
    """Mappe un label 5 étoiles ('1 star', ..., '5 stars') en 3 classes métier.

    Le sentiment positif est la somme de star 4 et 5
    Le sentiment neutre est la star 3
    Le sentiment négatif est la somme de star 1 et 2

    Args:
        star_label: label produit par le modèle (ex: '4 stars').

    Returns:
        Sentiment 3 classes.

    Raises:
        ValueError: si `star_label` n'est pas dans le format attendu.
    """
    score_5_classes = {d['label']: d['score'] for d in sorted(sentiment_5_classes, key=lambda x: int(x['label'].split()[0]))}
    logger.info("Scores 5 classes: {}", score_5_classes)

    # scores_3_classes = [
    #     score_5_classes['1 star'] + score_5_classes['2 stars'],
    #     score_5_classes['3 stars'],
    #     score_5_classes['4 stars'] + score_5_classes['5 stars']

    # Récupérer le label avec le score max parmi les 5 classes, puis faire le mapping vers les 3 classes
    max_label = max(score_5_classes, key=score_5_classes.get)
    max_score = score_5_classes[max_label]
    star_count = int(max_label.split()[0])

    logger.info("Label max: {}, score max: {}, start_count:{}", max_label, max_score, star_count)
 
    if star_count == 1 or star_count == 2:
        sentiment_3_classes = "négatif"
    elif star_count == 3:
        sentiment_3_classes = "neutre"
    else:
        sentiment_3_classes = "positif"
    
    logger.debug("Prediction 3 classes: 1 sentiment: {}", sentiment_3_classes)
    
    return sentiment_3_classes

def predict_sentiment(pipeline: Any, text: str, model_name: str) -> SentimentOut:
    """Inférence de sentiment sur un texte FR.

    Args:
        pipeline: pipeline `transformers.pipeline("text-classification", ...)`
            chargé au démarrage de l'API.
        text: texte FR de la review.
        model_name: identifiant HF du modèle (passé pour traçabilité).

    Returns:
        SentimentOut avec sentiment 3 classes, scores 5★ bruts, et latence ms.
    """

    timer1 = time.perf_counter()

    sentiment = pipeline(
        text,
        top_k=None
    )
    
    timer2 = time.perf_counter()

    score_5_classes = {d['label']: d['score'] for d in sorted(sentiment, key=lambda x: int(x['label'].split()[0]))}

    # logger.info("Requête /predict 5 classes: prediction={}", score_5_classes)

    # argmax = max(score_5_classes, key=score_5_classes.get)
    # logger.info("Requête /predict label max:{}", argmax)

    sentiment_3_classes = map_stars_to_sentiment(sentiment)

    return SentimentOut(
        sentiment = sentiment_3_classes,
        scores_5_stars=score_5_classes,
        model_name=model_name,
        latence_ms= timer2 - timer1
    )