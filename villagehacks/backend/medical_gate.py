import re

# This class is a quick way to check if a medical report is valid. It uses Regex to check for keywords and patterns for words that look like a medical term, drug, or dosage.
class MedicalGate:
    
    def __init__(self):

        # Drug suffix
        # We compile the regex so it runs instantly. 
        self.drug_suffix_pattern = re.compile(
            r'\b\w+(?:'
            r'afil|asone|bicin|bital|caine|cillin|cycline|dipine|dronate|'
            r'eprazole|fenac|floxacin|gliptin|glitazone|iramine|lamide|'
            r'mab|mustine|mycin|nacin|nazole|olol|olone|onide|oprazole|'
            r'parin|phylline|pramine|pril|profen|sartan|semide|statin|'
            r'thiazide|tidine|triptan|vir|zepam|zolam'
            r')\b', 
            re.IGNORECASE
        )

        # Dosage pattern
        self.dosage_pattern = re.compile(
            r'\b\d+(?:\.\d+)?(?:mg|ml|mcg|g|kg|oz|units?)\b', 
            re.IGNORECASE
        )

        # Medical Keywords
        # An exact match for common clinical words that might precede or follow drugs
        self.clinical_keywords = {
            "dose", "dosage", "prescribed", "refill", "pharmacy", 
            "symptom", "pain", "fever", "allergic", "allergy", "reaction"
        }

    def is_medical_term(self, word: str) -> bool:
        # Takes a single word and returns True if it passes the medical gate.
        # Clean the word of punctuation just in case
        clean_word = re.sub(r'[^\w\s]', '', word).strip().lower()
        
        if not clean_word:
            return False
            
        # Check 1: Is it an exact clinical keyword?
        if clean_word in self.clinical_keywords:
            return True
            
        # Check 2: Does it match a drug suffix? (e.g., "lisinopril")
        if self.drug_suffix_pattern.search(clean_word):
            return True
            
        # Check 3: Does it look like a dosage? (e.g., "500mg")
        if self.dosage_pattern.search(clean_word):
            return True
            
        return False

if __name__ == "__main__":
    gate = MedicalGate()
    
    test_words = ["hello", "metformin", "lisinopril", "the", "500mg", "appointment", "ibuprofen"]
    
    for w in test_words:
        is_medical = gate.is_medical_term(w)
        print(f"Word: {w:<15} | Passes Gate: {is_medical}")
