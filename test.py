import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Analyse de matchs", layout="wide")
st.title("⚽ Analyseur de matchs manuel – version avancée avec base de données des équipes")

# ---------------- FICHIER DES FORMES ----------------
FORM_FILE = "teams_form.json"

if os.path.exists(FORM_FILE):
    with open(FORM_FILE, "r") as f:
        teams_form = json.load(f)
else:
    teams_form = {}

# Initialiser le DataFrame
if "matches_df" not in st.session_state:
    st.session_state.matches_df = pd.DataFrame(columns=[
        "home_team", "away_team", "cote_home", "cote_away",
        "home_wins", "home_draws", "home_losses",
        "home_goals_scored", "home_goals_against",
        "home_last5",
        "away_wins", "away_draws", "away_losses",
        "away_goals_scored", "away_goals_against",
        "away_last5"
    ])

# ---------------- FORMULAIRE ----------------
with st.form("match_form", clear_on_submit=True):
    st.subheader("Équipes et Cotes")
    home_team = st.text_input("Équipe Domicile")
    away_team = st.text_input("Équipe Extérieure")
    cote_home = st.number_input("Cote Domicile", 1.01, 10.0, 1.5)
    cote_away = st.number_input("Cote Extérieure", 1.01, 10.0, 1.5)

    st.subheader("Historique Domicile (saison)")
    home_wins = st.number_input("Victoires Domicile", 0, 50, 0)
    home_draws = st.number_input("Nuls Domicile", 0, 50, 0)
    home_losses = st.number_input("Défaites Domicile", 0, 50, 0)
    home_goals_scored = st.number_input("Buts marqués Domicile", 0, 200, 0)
    home_goals_against = st.number_input("Buts encaissés Domicile", 0, 200, 0)
    # pré-remplir avec la forme enregistrée si elle existe
    default_home_last5 = teams_form.get(home_team, "v,v,n,d,d")
    home_last5 = st.text_input("5 derniers matchs Domicile (ex: v,v,n,d,d)", value=default_home_last5)

    st.subheader("Historique Extérieur (saison)")
    away_wins = st.number_input("Victoires Extérieures", 0, 50, 0)
    away_draws = st.number_input("Nuls Extérieurs", 0, 50, 0)
    away_losses = st.number_input("Défaites Extérieures", 0, 50, 0)
    away_goals_scored = st.number_input("Buts marqués Extérieur", 0, 200, 0)
    away_goals_against = st.number_input("Buts encaissés Extérieur", 0, 200, 0)
    default_away_last5 = teams_form.get(away_team, "v,v,n,d,d")
    away_last5 = st.text_input("5 derniers matchs Extérieur (ex: v,d,v,n,d)", value=default_away_last5)

    submitted = st.form_submit_button("➕ Ajouter le match")

# ---------------- AJOUT DES DONNÉES ----------------
if submitted:
    # Mettre à jour la base de données des formes
    teams_form[home_team] = home_last5.lower()
    teams_form[away_team] = away_last5.lower()
    with open(FORM_FILE, "w") as f:
        json.dump(teams_form, f)

    st.session_state.matches_df = pd.concat([
        st.session_state.matches_df,
        pd.DataFrame([{
            "home_team": home_team,
            "away_team": away_team,
            "cote_home": cote_home,
            "cote_away": cote_away,
            "home_wins": home_wins,
            "home_draws": home_draws,
            "home_losses": home_losses,
            "home_goals_scored": home_goals_scored,
            "home_goals_against": home_goals_against,
            "home_last5": home_last5.lower(),
            "away_wins": away_wins,
            "away_draws": away_draws,
            "away_losses": away_losses,
            "away_goals_scored": away_goals_scored,
            "away_goals_against": away_goals_against,
            "away_last5": away_last5.lower(),
        }])
    ], ignore_index=True)
    st.success(f"✅ Match ajouté : {home_team} vs {away_team}")

# ---------------- ALGORITHME D’ANALYSE ----------------
def calculate_form_score(sequence):
    mapping = {"v": 3, "n": 1, "d": 0}
    seq = [mapping.get(x.strip(), 0) for x in sequence.split(",")]
    if len(seq) < 5:
        seq += [0] * (5 - len(seq))
    weights = np.array([5, 4, 3, 2, 1])
    return np.dot(seq, weights) / 15

def analyze(df):
    df = df.copy()
    results = []
    for _, row in df.iterrows():
        home_form = calculate_form_score(row["home_last5"])
        away_form = calculate_form_score(row["away_last5"])
        attack_strength = row["home_goals_scored"] - row["away_goals_against"]
        defense_strength = row["away_goals_scored"] - row["home_goals_against"]

        score = (home_form * 0.4 + (attack_strength - defense_strength) * 0.02) * 100
        prob_home = 1 / (1 + np.exp(-score / 10))
        prob_away = 1 - prob_home
        winner = row["home_team"] if prob_home > prob_away else row["away_team"]

        results.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "Winner": winner,
            "Probabilité victoire": round(max(prob_home, prob_away) * 100, 2),
            "Score Sécurité": round(abs(prob_home - prob_away) * 100, 1),
        })
    return pd.DataFrame(results)

# ---------------- AFFICHAGE ----------------
if len(st.session_state.matches_df) > 0:
    st.subheader("📊 Analyse des matchs saisis")
    df_analysis = analyze(st.session_state.matches_df)
    st.dataframe(df_analysis.sort_values(by="Score Sécurité", ascending=False), use_container_width=True)

    # Calcul des mises avec Kelly
    st.subheader("💰 Recommandation de mise (Kelly simplifié)")
    budget_total = st.number_input("Budget total (€)", 1, 10000, 100, step=10)
    df_analysis["cote_home"] = st.session_state.matches_df["cote_home"]
    df_analysis["cote_away"] = st.session_state.matches_df["cote_away"]

    mises = []
    for i, row in df_analysis.iterrows():
        cote = row["cote_home"] if row["Winner"] == row["home_team"] else row["cote_away"]
        p = row["Probabilité victoire"] / 100
        b = cote - 1
        q = 1 - p
        f_star = max((b * p - q) / b, 0)
        mises.append(round(f_star * budget_total, 2))

    df_analysis["Mise conseillée (€)"] = mises
    st.dataframe(df_analysis[["home_team","away_team","Winner","Probabilité victoire","Score Sécurité","Mise conseillée (€)"]], use_container_width=True)

    # Télécharger les résultats
    st.download_button("📥 Télécharger les résultats (CSV)",
                       df_analysis.to_csv(index=False).encode("utf-8"),
                       "analyse_matchs.csv",
                       "text/csv")

else:
    st.info("Ajoute au moins un match pour commencer l’analyse ⚙️")
