import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Analyse de matchs avancée", layout="wide")
st.title("⚽ Analyseur de matchs – Formes auto-mises à jour avec sélection rapide")

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

# ---------------- FORMULAIRE AVEC SÉLECTION D'ÉQUIPE ----------------
with st.form("match_form", clear_on_submit=True):
    st.subheader("Équipes et Cotes")

    saved_teams = list(teams_form.keys())
    saved_teams_sorted = sorted(saved_teams)

    # Équipe domicile
    st.markdown("**Équipe Domicile**")
    home_team = None
    col1, col2 = st.columns([3,7])
    with col1:
        for team in saved_teams_sorted:
            if st.button(team + " (domicile)"):
                st.session_state.temp_home_team = team
    with col2:
        home_team_input = st.text_input("Ou saisissez une nouvelle équipe Domicile")
    home_team = st.session_state.get("temp_home_team", None) or home_team_input

    # Équipe extérieure
    st.markdown("**Équipe Extérieure**")
    away_team = None
    col1, col2 = st.columns([3,7])
    with col1:
        for team in saved_teams_sorted:
            if st.button(team + " (extérieur)"):
                st.session_state.temp_away_team = team
    with col2:
        away_team_input = st.text_input("Ou saisissez une nouvelle équipe Extérieure")
    away_team = st.session_state.get("temp_away_team", None) or away_team_input

    # Cotes
    cote_home = st.number_input("Cote Domicile", 1.01, 10.0, 1.5)
    cote_away = st.number_input("Cote Extérieure", 1.01, 10.0, 1.5)

    # Historique domicile
    st.subheader("Historique Domicile")
    home_wins = st.number_input("Victoires Domicile", 0, 50, 0)
    home_draws = st.number_input("Nuls Domicile", 0, 50, 0)
    home_losses = st.number_input("Défaites Domicile", 0, 50, 0)
    home_goals_scored = st.number_input("Buts marqués Domicile", 0, 200, 0)
    home_goals_against = st.number_input("Buts encaissés Domicile", 0, 200, 0)
    default_home_last5 = teams_form.get(home_team, "v,v,n,d,d") if home_team else "v,v,n,d,d"
    home_last5 = st.text_input("5 derniers matchs Domicile (v,n,d)", value=default_home_last5)

    # Historique extérieur
    st.subheader("Historique Extérieur")
    away_wins = st.number_input("Victoires Extérieures", 0, 50, 0)
    away_draws = st.number_input("Nuls Extérieurs", 0, 50, 0)
    away_losses = st.number_input("Défaites Extérieures", 0, 50, 0)
    away_goals_scored = st.number_input("Buts marqués Extérieur", 0, 200, 0)
    away_goals_against = st.number_input("Buts encaissés Extérieur", 0, 200, 0)
    default_away_last5 = teams_form.get(away_team, "v,v,n,d,d") if away_team else "v,v,n,d,d"
    away_last5 = st.text_input("5 derniers matchs Extérieur (v,n,d)", value=default_away_last5)

    submitted = st.form_submit_button("➕ Ajouter le match")

# ---------------- AJOUT DES DONNÉES ----------------
if submitted and home_team and away_team:
    # Enregistrer les équipes et formes
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

# ---------------- ANALYSE ET MISE À JOUR DES FORMES ----------------
def calculate_form_score(sequence):
    mapping = {"v": 3, "n": 1, "d": 0}
    seq = [mapping.get(x.strip(), 0) for x in sequence.split(",")]
    if len(seq) < 5:
        seq += [0]*(5-len(seq))
    weights = np.array([5,4,3,2,1])
    return np.dot(seq, weights)/15

def analyze(df):
    df = df.copy()
    results = []
    for _, row in df.iterrows():
        home_form = calculate_form_score(row["home_last5"])
        away_form = calculate_form_score(row["away_last5"])
        attack_strength = row["home_goals_scored"] - row["away_goals_against"]
        defense_strength = row["away_goals_scored"] - row["home_goals_against"]

        score = (home_form*0.4 + (attack_strength-defense_strength)*0.02)*100
        prob_home = 1/(1+np.exp(-score/10))
        prob_away = 1 - prob_home
        winner = row["home_team"] if prob_home>prob_away else row["away_team"]

        results.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "Winner": winner,
            "Probabilité victoire": round(max(prob_home, prob_away)*100,2),
            "Score Sécurité": round(abs(prob_home-prob_away)*100,1)
        })
    return pd.DataFrame(results)

def update_form_after_match(df_analysis):
    for idx, row in df_analysis.iterrows():
        winner = row["Winner"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_seq = teams_form.get(home_team,"v,v,n,d,d").split(",")[:4]
        away_seq = teams_form.get(away_team,"v,v,n,d,d").split(",")[:4]

        if winner==home_team:
            home_seq = ["v"]+home_seq
            away_seq = ["d"]+away_seq
        elif winner==away_team:
            home_seq = ["d"]+home_seq
            away_seq = ["v"]+away_seq
        else:
            home_seq = ["n"]+home_seq
            away_seq = ["n"]+away_seq

        teams_form[home_team] = ",".join(home_seq)
        teams_form[away_team] = ",".join(away_seq)

    with open(FORM_FILE,"w") as f:
        json.dump(teams_form,f)

# ---------------- AFFICHAGE ----------------
if len(st.session_state.matches_df)>0:
    st.subheader("📊 Analyse des matchs")
    df_analysis = analyze(st.session_state.matches_df)
    df_analysis = df_analysis.sort_values(by="Score Sécurité",ascending=False)
    st.dataframe(df_analysis[["home_team","away_team","Winner","Probabilité victoire","Score Sécurité"]],use_container_width=True)

    # Mise Kelly
    st.subheader("💰 Recommandation de mise (Kelly simplifié)")
    budget_total = st.number_input("Budget total (€)",1,10000,100,step=10)
    df_analysis["cote_home"] = st.session_state.matches_df["cote_home"]
    df_analysis["cote_away"] = st.session_state.matches_df["cote_away"]
    mises=[]
    for i,row in df_analysis.iterrows():
        cote = row["cote_home"] if row["Winner"]==row["home_team"] else row["cote_away"]
        p = row["Probabilité victoire"]/100
        b = cote-1
        q = 1-p
        f_star = max((b*p-q)/b,0)
        mises.append(round(f_star*budget_total,2))
    df_analysis["Mise conseillée (€)"]=mises
    st.dataframe(df_analysis[["home_team","away_team","Winner","Probabilité victoire","Score Sécurité","Mise conseillée (€)"]],use_container_width=True)

    # Mettre à jour formes automatiquement
    update_form_after_match(df_analysis)
    st.success("✅ Formes mises à jour automatiquement")

    # Télécharger CSV
    st.download_button("📥 Télécharger résultats (CSV)", df_analysis.to_csv(index=False).encode("utf-8"), "analyse_matchs.csv","text/csv")
else:
    st.info("Ajoute au moins un match pour commencer l’analyse ⚙️")
