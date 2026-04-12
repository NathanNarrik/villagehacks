import numpy as np
import xgboost as xgb

# ML brain of the system. Scores the uncertainty of a transcribed word based on its features.
class UncertaintyScorer:
    def __init__(self, model_path: str = "uncertainty_model.xgb"):
        self.model_path = model_path
        self.is_trained = False
        
        self.model = xgb.Booster()
        try:
            self.model.load_model(self.model_path)
            self.is_trained = True
            print(f"Loaded XGBoost model from {self.model_path}")
        except xgb.core.XGBoostError:
            print("Warning: No XGBoost model found. Running in MOCK mode for development.")

    def extract_features(self, word_data: dict, phonetic_distance: float = 0.0, history_score: float = 0.0) -> list:
        # Transforms raw word data into a numerical feature vector for XGBoost.
        duration = word_data.get("end_ms", 0) - word_data.get("start_ms", 0)
        
        # Features: [Duration, Phonetic Distance to Keyterm, Correction History]
        return [float(duration), float(phonetic_distance), float(history_score)]

    def score_word(self, word_data: dict) -> tuple[str, float]:
        # Evaluates a word and returns (ConfidenceLevel, RiskScore).
        # RiskScore: 0.0 (Perfectly certain) to 1.0 (Highly uncertain)
        # ConfidenceLevel: "HIGH", "MEDIUM", or "LOW"
        if not self.is_trained:
            # --- MOCK MODE ---
            # Simulate ML logic so you can build the rest of the pipeline today
            duration = word_data.get("end_ms", 0) - word_data.get("start_ms", 0)
            word = word_data.get("word", "").lower()
            
            # Fake a high risk score if it's a known tough medical word or spoken way too fast/slow
            if duration < 100 or duration > 800 or "metoformin" in word:
                risk_score = 0.85
            else:
                risk_score = 0.15
        else:
            # --- PRODUCTION ML MODE ---
            features = self.extract_features(word_data)
            dmatrix = xgb.DMatrix([features])
            risk_score = float(self.model.predict(dmatrix)[0])

        # Strict Gate Logic: This is what the API will look at.
        if risk_score >= 0.70:
            return "LOW", risk_score       # Trigger Tavily
        elif risk_score >= 0.40:
            return "MEDIUM", risk_score    # Flag for clinician review (Amber)
        else:
            return "HIGH", risk_score      # Leave it alone

if __name__ == "__main__":
    scorer = UncertaintyScorer()
    
    # Simulate an abnormally short word (e.g., mumble over the phone)
    bad_audio_word = {"word": "metoformin", "start_ms": 1000, "end_ms": 1050}
    
    # Simulate a clearly spoken normal word
    good_audio_word = {"word": "today", "start_ms": 2000, "end_ms": 2300}
    
    confidence, risk = scorer.score_word(bad_audio_word)
    print(f"Bad Word  -> Confidence: {confidence:<6} | Risk: {risk:.2f}")
    
    confidence, risk = scorer.score_word(good_audio_word)
    print(f"Good Word -> Confidence: {confidence:<6} | Risk: {risk:.2f}")