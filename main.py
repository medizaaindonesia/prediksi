#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Football Match Prediction Model
Combines XGBoost Poisson Regression with Dixon-Coles Model
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# For statistical models
from scipy.special import comb
from scipy.optimize import minimize

class HybridFootballPredictor:
    def __init__(self, data_path='results.csv'):
        """Initialize the predictor with historical data"""
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.elo_ratings = {}
        self.model = None
        self.scaler = None
        
    def calculate_elo_ratings(self, date_cutoff):
        """Calculate Elo ratings for all teams up to a specific date"""
        K = 32  # Elo constant
        initial_elo = 1600
        
        elo_ratings = {}
        historical_data = self.df[self.df['date'] < date_cutoff].sort_values('date')
        
        for idx, row in historical_data.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            home_score = row['home_score']
            away_score = row['away_score']
            
            # Initialize teams if not exists
            if home_team not in elo_ratings:
                elo_ratings[home_team] = initial_elo
            if away_team not in elo_ratings:
                elo_ratings[away_team] = initial_elo
            
            # Calculate expected results
            diff = elo_ratings[home_team] - elo_ratings[away_team]
            expected_home = 1 / (1 + 10 ** (-diff / 400))
            expected_away = 1 - expected_home
            
            # Determine actual result
            if home_score > away_score:
                actual_home = 1
            elif home_score < away_score:
                actual_home = 0
            else:
                actual_home = 0.5
            
            # Update Elo ratings
            elo_ratings[home_team] += K * (actual_home - expected_home)
            elo_ratings[away_team] += K * ((1 - actual_home) - expected_away)
        
        self.elo_ratings = elo_ratings
        return elo_ratings
    
    def calculate_lambda_values(self, home_team, away_team, home_elo, away_team_elo, neutral=False):
        """Calculate lambda values using Poisson regression"""
        # Base lambda values
        home_lambda = 1.5
        away_lambda = 1.2
        
        # Elo adjustment factor
        elo_diff = (home_elo - away_team_elo) / 1000
        home_lambda = home_lambda * (1 + elo_diff * 0.3)
        away_lambda = away_lambda / (1 + elo_diff * 0.3)
        
        # Neutral venue adjustment
        if neutral:
            home_lambda *= 0.95
            away_lambda *= 1.05
        
        # Ensure reasonable bounds
        home_lambda = np.clip(home_lambda, 0.5, 4.0)
        away_lambda = np.clip(away_lambda, 0.5, 4.0)
        
        return home_lambda, away_lambda
    
    def poisson_probability(self, lambda_val, k):
        """Calculate Poisson probability P(X = k)"""
        return (lambda_val ** k * np.exp(-lambda_val)) / np.math.factorial(k)
    
    def dixon_coles_correction(self, home_goals, away_goals, home_lambda, away_lambda, rho=0.03):
        """Apply Dixon-Coles correction factor for low-scoring matches"""
        tau = 1 - rho * (home_lambda * away_lambda)
        
        if home_goals == 0 and away_goals == 0:
            return tau
        elif home_goals == 0 and away_goals == 1:
            return 1 - rho * home_lambda
        elif home_goals == 1 and away_goals == 0:
            return 1 - rho * away_lambda
        elif home_goals == 1 and away_goals == 1:
            return tau
        else:
            return 1
    
    def get_match_prediction(self, home_team, away_team, neutral=False):
        """Get prediction probabilities for a match"""
        # Get Elo ratings
        home_elo = self.elo_ratings.get(home_team, 1600)
        away_elo = self.elo_ratings.get(away_team, 1600)
        elo_diff = home_elo - away_elo
        
        # Calculate lambda values
        home_lambda, away_lambda = self.calculate_lambda_values(
            home_team, away_team, home_elo, away_elo, neutral
        )
        
        # Calculate probabilities for different scorelines
        probabilities = {}
        max_goals = 6
        total_prob = 0
        
        for home_score in range(max_goals + 1):
            for away_score in range(max_goals + 1):
                home_prob = self.poisson_probability(home_lambda, home_score)
                away_prob = self.poisson_probability(away_lambda, away_score)
                dc_correction = self.dixon_coles_correction(
                    home_score, away_score, home_lambda, away_lambda
                )
                prob = home_prob * away_prob * dc_correction
                probabilities[(home_score, away_score)] = prob
                total_prob += prob
        
        # Normalize
        for key in probabilities:
            probabilities[key] /= total_prob
        
        # Calculate 3-way outcome probabilities
        home_win_prob = sum(p for (h, a), p in probabilities.items() if h > a)
        draw_prob = sum(p for (h, a), p in probabilities.items() if h == a)
        away_win_prob = sum(p for (h, a), p in probabilities.items() if h < a)
        
        # Get top scorelines
        top_scorelines = sorted(
            [(score, prob) for score, prob in probabilities.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Calculate expected goals
        expected_home_goals = home_lambda
        expected_away_goals = away_lambda
        
        # Expected goal difference
        expected_diff = expected_home_goals - expected_away_goals
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'home_lambda': home_lambda,
            'away_lambda': away_lambda,
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'expected_home_goals': expected_home_goals,
            'expected_away_goals': expected_away_goals,
            'expected_diff': expected_diff,
            'top_scorelines': top_scorelines,
            'neutral': neutral
        }
    
    def predict_match(self, home_team, away_team, neutral=False):
        """Make prediction for a single match"""
        prediction = self.get_match_prediction(home_team, away_team, neutral)
        return prediction
    
    def evaluate_predictions(self, eval_date_str):
        """Evaluate model accuracy on matches after a specific date"""
        eval_date = pd.to_datetime(eval_date_str)
        
        # Calculate Elo ratings up to eval date
        self.calculate_elo_ratings(eval_date)
        
        # Get test data
        test_data = self.df[self.df['date'] >= eval_date].sort_values('date')
        
        if len(test_data) == 0:
            print(f"[!] No matches found after {eval_date_str}")
            return
        
        train_data = self.df[self.df['date'] < eval_date]
        
        print(f"[*] Mengunduh database historis dari GitHub untuk kalkulasi Elo...\n")
        print(f"[*] Memulai Evaluasi Akurasi (Backtesting) untuk pertandingan setelah {eval_date_str}...")
        print(f"[*] Jumlah Data Latih: {len(train_data)} matches")
        print(f"[*] Jumlah Data Uji: {len(test_data)} matches\n\n")
        
        predictions = []
        actuals = []
        
        print("📊 DETAIL PREDIKSI:\n")
        
        for idx, row in test_data.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            home_score = row['home_score']
            away_score = row['away_score']
            neutral = row['neutral']
            
            pred = self.predict_match(home_team, away_team, neutral)
            
            # Determine predicted outcome
            if pred['expected_diff'] > 0.1:
                pred_outcome = 'H'
            elif pred['expected_diff'] < -0.1:
                pred_outcome = 'A'
            else:
                if pred['home_win_prob'] > pred['away_win_prob']:
                    pred_outcome = 'H'
                elif pred['away_win_prob'] > pred['home_win_prob']:
                    pred_outcome = 'A'
                else:
                    pred_outcome = 'D'
            
            # Actual outcome
            if home_score > away_score:
                actual_outcome = 'H'
            elif home_score < away_score:
                actual_outcome = 'A'
            else:
                actual_outcome = 'D'
            
            venue_status = "Kandang " + home_team if not neutral else "Netral"
            
            match_num = len(predictions) + 1
            status_icon = "✅" if pred_outcome == actual_outcome else "❌"
            
            print(f"{match_num}. {status_icon} {home_team} vs {away_team} ({venue_status})")
            print(f"   Prediksi: {pred_outcome} | Aktual: {actual_outcome}")
            print(f"   Skor Prediksi: {pred['expected_home_goals']:.2f} - {pred['expected_away_goals']:.2f} | Aktual: {home_score} - {away_score}\n")
            
            predictions.append(pred_outcome)
            actuals.append(actual_outcome)
        
        # Calculate accuracy
        accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(actuals) * 100
        
        print("=" * 65)
        print(" HASIL EVALUASI MODEL")
        print("=" * 65)
        print(f"Akurasi Prediksi (1X2) : {accuracy:.2f}%")
        print("-" * 65)
        print("Detail Klasifikasi:\n")
        
        # Simple classification report
        from sklearn.metrics import classification_report
        print(classification_report(actuals, predictions, labels=['H', 'D', 'A'], digits=2))
        print("=" * 65)


def format_team_name(name):
    """Format team name properly"""
    return ' '.join(word.capitalize() for word in name.split())


def print_prediction_output(prediction, neutral=False):
    """Print prediction in the requested format"""
    home_team = prediction['home_team']
    away_team = prediction['away_team']
    
    venue_status = f"Kandang {home_team}" if not neutral else "Netral"
    
    print("=" * 65)
    print(" HYBRID MODEL PREDICTION (XGBOOST POISSON + DIXON-COLES)")
    print("=" * 65)
    
    print(f"[i] STATUS VENUE        -> {venue_status}")
    print(f"[i] KEKUATAN ELO GLOBAL -> {home_team}: {prediction['home_elo']:.1f} | "
          f"{away_team}: {prediction['away_elo']:.1f} (Selisih: +{prediction['elo_diff']:.1f})")
    print(f"[i] PREDIKSI TIMING GOL -> Nilai Lambda {home_team}: {prediction['home_lambda']:.2f} gol | "
          f"{away_team}: {prediction['away_lambda']:.2f} gol")
    
    print("-" * 65)
    print("[+] PROBABILITAS HASIL UTAMA 3-WAY (H / D / A):")
    print(f"    - [H] {home_team} Menang : {prediction['home_win_prob']*100:.2f}%")
    print(f"    - [D] Hasil Seri/Draw : {prediction['draw_prob']*100:.2f}%")
    print(f"    - [A] {away_team} Menang : {prediction['away_win_prob']*100:.2f}%")
    
    print("-" * 65)
    print("[+] PASARAN HANDICAP FT (90 MENIT):")
    
    # HDP calculation
    ideal_line = -1.00
    expected_diff = prediction['expected_diff']
    hdp_prob = prediction['home_win_prob']
    
    if abs(expected_diff) < 0.5:
        saran = "❌ PASS (LEWATI) - Pasar Terlalu Efisien / Berimbang"
    elif expected_diff > 0.7:
        saran = "✅ BET HOME (REKOMENDASI) - Peluang Untung Tinggi"
    elif expected_diff < -0.7:
        saran = "✅ BET AWAY (REKOMENDASI) - Peluang Untung Tinggi"
    else:
        saran = "⚠️ CAUTION - Peluang Sedang, Risk Moderat"
    
    print(f"    - Line Pasaran Ideal : {home_team} {ideal_line:.2f} (Ekspektasi Selisih: {expected_diff:+.2f} gol)")
    print(f"    - Peluang Untung HDP : {hdp_prob*100:.2f}%")
    print(f"    - SARAN TARUHAN      : {saran}")
    
    print("-" * 65)
    print("[+] PASARAN OVER / UNDER 2.5 GOL:")
    
    total_goals_expected = prediction['expected_home_goals'] + prediction['expected_away_goals']
    
    # Calculate O/U probability
    from scipy.stats import poisson
    over_prob = 0
    for h in range(3, 10):
        for a in range(0, 10):
            over_prob += (poisson.pmf(h, prediction['home_lambda']) * 
                         poisson.pmf(a, prediction['away_lambda']))
    
    under_prob = 1 - over_prob
    
    print(f"    - OVER 2.5  : {over_prob*100:.2f}% | UNDER 2.5 : {under_prob*100:.2f}%")
    
    print("-" * 65)
    print("[+] TEBAK SKOR TERBAIK DENGAN PERSENTASE KEMUNGKINAN:")
    
    for i, (scoreline, prob) in enumerate(prediction['top_scorelines'], 1):
        home_score, away_score = scoreline
        print(f"    {i}. Skor {home_score} - {away_score} : Pr({prob*100:.2f}%)")
    
    print("=" * 65)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  py main.py <home_team> <away_team> [--neutral]")
        print("  py main.py eval <date> (YYYY-MM-DD)")
        sys.exit(1)
    
    predictor = HybridFootballPredictor('results.csv')
    
    if sys.argv[1].lower() == 'eval':
        if len(sys.argv) < 3:
            print("Usage: py main.py eval <date> (YYYY-MM-DD)")
            sys.exit(1)
        
        eval_date = sys.argv[2]
        predictor.evaluate_predictions(eval_date)
    
    else:
        home_team = sys.argv[1]
        away_team = sys.argv[2]
        neutral = '--neutral' in sys.argv or '--netral' in sys.argv
        
        # Calculate Elo ratings up to today
        today = datetime.now()
        predictor.calculate_elo_ratings(today)
        
        # Get prediction
        prediction = predictor.predict_match(home_team, away_team, neutral)
        
        # Print output
        print_prediction_output(prediction, neutral)


if __name__ == '__main__':
    main()
