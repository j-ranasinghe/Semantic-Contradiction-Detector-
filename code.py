"""
Semantic Contradiction Detector
Assignment - Part 2

Approach: Enhanced hybrid system with better contradiction detection
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import re
from collections import defaultdict

# Required imports
try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    import torch
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
except ImportError as e:
    print(f"Missing dependencies: {e}")

@dataclass
class ContradictionResult:
    has_contradiction: bool
    confidence: float
    contradicting_pairs: List[Tuple[str, str]]
    explanation: str

class SemanticContradictionDetector:
    """
    Enhanced detector with improved contradiction detection.
    """
    
    def __init__(self, model_name: str = "default"):
        """Initialize the detector."""
        
        # Models
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.nli_model = pipeline(
            "text-classification",
            model="potsawee/deberta-v3-large-mnli",
            device=0 if torch.cuda.is_available() else -1,
            top_k=None  
        )
        
        # Thresholds (more sensitive)
        self.similarity_threshold = 0.3  
        self.contradiction_threshold = 0.5  
        
        # Enhanced aspect keywords
        self.aspect_keywords = {
            'performance': ['fast', 'slow', 'quick', 'sluggish', 'performance', 'speed', 'wait', 'waiting'],
            'durability': ['durable', 'fragile', 'sturdy', 'weak', 'broke', 'cracked', 'damage'],
            'quality': ['quality', 'excellent', 'poor', 'great', 'terrible', 'mediocre'],
            'battery': ['battery', 'charge', 'charging', 'power'],
            'camera': ['camera', 'photo', 'picture', 'image'],
            'service': ['service', 'support', 'helpful', 'unhelpful', 'rude', 'polite', 'resolved'],
            'shipping': ['shipping', 'delivery', 'arrived', 'wait'],
            'noise': ['noise', 'cancellation', 'quiet', 'loud', 'hear'],
        }
        
        # Antonym pairs for contradiction detection
        self.antonym_pairs = {
            ('fast', 'slow'), ('quick', 'slow'), ('instant', 'wait'),
            ('helpful', 'unhelpful'), ('polite', 'rude'), ('good', 'bad'),
            ('durable', 'fragile'), ('strong', 'weak'), ('sturdy', 'broke'),
            ('quiet', 'loud'), ('silent', 'noisy'),
            ('excellent', 'terrible'), ('great', 'poor'), ('amazing', 'awful'),
        }
        
        self.negations = {'no', 'not', 'never', 'nothing', 'none', 'nobody', 
                          'nowhere', 'neither', 'nor', "n't", 'cannot', 'cant', 'barely'}
        
        print("✓ Enhanced models loaded successfully")
    
    def preprocess(self, text: str) -> List[str]:
        """Preprocess text into sentences."""
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        return sentences
    
    def extract_numbers_with_context(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers with their context (time units, etc.)."""
        patterns = [
            (r'(\d+)\s*(second|sec|s)\b', 'seconds'),
            (r'(\d+)\s*(minute|min|m)\b', 'minutes'),
            (r'(\d+)\s*(hour|hr|h)\b', 'hours'),
            (r'(\d+)\s*(day|d)\b', 'days'),
            (r'(\d+)\s*(week|wk|w)\b', 'weeks'),
            (r'(\d+)\s*(month|mo)\b', 'months'),
        ]
        
        results = []
        for pattern, unit in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                value = int(match.group(1))
                results.append({
                    'value': value,
                    'unit': unit,
                    'text': match.group(0),
                    'position': match.span()
                })
        return results
    
    def extract_claims(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Extract factual claims with enhanced metadata."""
        claims = []
        
        for idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Extract aspects
            aspects = []
            for aspect, keywords in self.aspect_keywords.items():
                if any(kw in sentence_lower for kw in keywords):
                    aspects.append(aspect)
            
            # Check for negations
            has_negation = any(neg in sentence_lower.split() for neg in self.negations)
            
            # Enhanced sentiment analysis
            positive_words = ['great', 'excellent', 'amazing', 'stunning', 'beautiful', 
                             'fast', 'good', 'love', 'perfect', 'exceptional', 'helpful',
                             'lightning', 'resolved', 'polite', 'quiet', 'silent']
            negative_words = ['poor', 'bad', 'terrible', 'slow', 'disappointing', 
                             'broken', 'cracked', 'awful', 'worst', 'unhelpful', 'rude',
                             'wait', 'waiting', 'mediocre', 'loud', 'noisy']
            
            pos_count = sum(1 for w in positive_words if w in sentence_lower)
            neg_count = sum(1 for w in negative_words if w in sentence_lower)
            
            # Determine sentiment with negation handling
            if has_negation:
                sentiment = 'negative' if pos_count > neg_count else 'positive' if neg_count > pos_count else 'neutral'
            else:
                sentiment = 'positive' if pos_count > neg_count else 'negative' if neg_count > 0 else 'neutral'
            
            # Extract temporal information
            temporal_info = self.extract_numbers_with_context(sentence)
            
            claim = {
                'sentence': sentence,
                'index': idx,
                'aspects': aspects,
                'has_negation': has_negation,
                'sentiment': sentiment,
                'temporal_info': temporal_info,
                'embedding': None
            }
            
            claims.append(claim)
        
        return claims
    
    def check_antonym_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float]:
        """Check for antonym-based contradictions."""
        text_a = claim_a['sentence'].lower()
        text_b = claim_b['sentence'].lower()
        
        # Check for antonym pairs
        for word1, word2 in self.antonym_pairs:
            if (word1 in text_a and word2 in text_b) or (word2 in text_a and word1 in text_b):
                # Check if they share aspects
                if set(claim_a['aspects']) & set(claim_b['aspects']):
                    return True, 0.85
        
        return False, 0.0
    
    def check_temporal_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float]:
        """Check for temporal contradictions (e.g., '2 days' vs '3 weeks')."""
        if not claim_a['temporal_info'] or not claim_b['temporal_info']:
            return False, 0.0
        
        # Convert all to hours for comparison
        unit_to_hours = {
            'seconds': 1/3600,
            'minutes': 1/60,
            'hours': 1,
            'days': 24,
            'weeks': 24 * 7,
            'months': 24 * 30
        }
        
        for temp_a in claim_a['temporal_info']:
            for temp_b in claim_b['temporal_info']:
                hours_a = temp_a['value'] * unit_to_hours[temp_a['unit']]
                hours_b = temp_b['value'] * unit_to_hours[temp_b['unit']]
                
                # If one is much smaller than the other (10x+ difference)
                ratio = max(hours_a, hours_b) / (min(hours_a, hours_b) + 0.001)
                
                if ratio > 10:
                    # Check context 
                    text_a = claim_a['sentence'].lower()
                    text_b = claim_b['sentence'].lower()
                    
                    # Words indicating fast vs slow
                    fast_words = ['fast', 'quick', 'lightning', 'instant']
                    slow_words = ['wait', 'slow', 'delay', 'took']
                    
                    a_fast = any(w in text_a for w in fast_words)
                    b_fast = any(w in text_b for w in fast_words)
                    a_slow = any(w in text_a for w in slow_words)
                    b_slow = any(w in text_b for w in slow_words)
                    
                    if (a_fast and b_slow) or (a_slow and b_fast):
                        return True, 0.9
                    
                    # Or if they share aspects 
                    if set(claim_a['aspects']) & set(claim_b['aspects']):
                        return True, 0.85
        
        return False, 0.0
    
    def check_service_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float]:
        """Special check for service quality contradictions."""
        if 'service' not in claim_a['aspects'] or 'service' not in claim_b['aspects']:
            return False, 0.0
        
        text_a = claim_a['sentence'].lower()
        text_b = claim_b['sentence'].lower()
        
        # Check for opposite service descriptions
        negative_service = ['unhelpful', 'rude', 'terrible', 'poor', 'bad']
        positive_service = ['helpful', 'resolved', 'polite', 'great', 'excellent', 'discount', 'gift']
        
        a_neg = any(w in text_a for w in negative_service)
        b_pos = any(w in text_b for w in positive_service)
        
        if a_neg and b_pos:
            return True, 0.9
        
        return False, 0.0
    
    def check_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float]:
        """Enhanced contradiction checking with multiple strategies."""
        
        #Antonym-based
        ant_result, ant_conf = self.check_antonym_contradiction(claim_a, claim_b)
        if ant_result:
            return True, ant_conf
        
        # Temporal contradictions
        temp_result, temp_conf = self.check_temporal_contradiction(claim_a, claim_b)
        if temp_result:
            return True, temp_conf
        
        # Service contradictions
        serv_result, serv_conf = self.check_service_contradiction(claim_a, claim_b)
        if serv_result:
            return True, serv_conf
        
        # Sentiment flip on same aspect
        common_aspects = set(claim_a['aspects']) & set(claim_b['aspects'])
        if common_aspects:
            if claim_a['sentiment'] == 'positive' and claim_b['sentiment'] == 'negative':
                return True, 0.75
            elif claim_a['sentiment'] == 'negative' and claim_b['sentiment'] == 'positive':
                return True, 0.75
        
        # NLI model (more lenient threshold)
        try:
            premise = claim_a['sentence']
            hypothesis = claim_b['sentence']
            
            result = self.nli_model(f"{premise} [SEP] {hypothesis}")[0]
            
            # Extract contradiction score
            contradiction_score = 0.0
            for item in result:
                if item['label'] == 'CONTRADICTION':
                    contradiction_score = item['score']
                    break
            
            # Also check reverse direction
            result_rev = self.nli_model(f"{hypothesis} [SEP] {premise}")[0]
            contradiction_score_rev = 0.0
            for item in result_rev:
                if item['label'] == 'CONTRADICTION':
                    contradiction_score_rev = item['score']
                    break
            
            max_contradiction = max(contradiction_score, contradiction_score_rev)
            
            if max_contradiction >= self.contradiction_threshold:
                return True, max_contradiction
                
        except Exception as e:
            print(f"NLI error: {e}")
        
        return False, 0.0
    
    def analyze(self, text: str) -> ContradictionResult:
        """Main analysis pipeline with improved detection."""
        sentences = self.preprocess(text)
        
        if len(sentences) < 2:
            return ContradictionResult(
                has_contradiction=False,
                confidence=0.0,
                contradicting_pairs=[],
                explanation="Not enough sentences to detect contradictions"
            )
        
        claims = self.extract_claims(sentences)
        
        # Compute embeddings
        sentence_texts = [c['sentence'] for c in claims]
        embeddings = self.sentence_encoder.encode(sentence_texts, convert_to_tensor=True)
        
        for i, claim in enumerate(claims):
            claim['embedding'] = embeddings[i]
        
        # Find contradictions 
        contradicting_pairs = []
        contradiction_details = []
        
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                # Compute similarity
                similarity = torch.nn.functional.cosine_similarity(
                    claims[i]['embedding'].unsqueeze(0),
                    claims[j]['embedding'].unsqueeze(0)
                ).item()
                

                is_contradiction, confidence = self.check_contradiction(claims[i], claims[j])
                
                if is_contradiction:
                    contradicting_pairs.append((claims[i]['sentence'], claims[j]['sentence']))
                    contradiction_details.append({
                        'pair': (i, j),
                        'confidence': confidence,
                        'similarity': similarity,
                        'common_aspects': list(set(claims[i]['aspects']) & set(claims[j]['aspects']))
                    })
        
        # Generate result
        if contradicting_pairs:
            best_contradiction = max(contradiction_details, key=lambda x: x['confidence'])
            overall_confidence = best_contradiction['confidence']
            
            aspects = ', '.join(best_contradiction['common_aspects']) if best_contradiction['common_aspects'] else 'related topics'
            explanation = f"Found {len(contradicting_pairs)} contradicting pair(s). "
            explanation += f"Highest confidence contradiction (score: {overall_confidence:.2f}) detected regarding {aspects}."
            
            return ContradictionResult(
                has_contradiction=True,
                confidence=overall_confidence,
                contradicting_pairs=contradicting_pairs,
                explanation=explanation
            )
        else:
            return ContradictionResult(
                has_contradiction=False,
                confidence=0.0,
                contradicting_pairs=[],
                explanation="No semantic contradictions detected"
            )


def evaluate(detector: SemanticContradictionDetector, 
             test_data: List[Dict]) -> Dict[str, float]:
    """Evaluate detector performance."""
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    results = []
    
    for sample in test_data:
        result = detector.analyze(sample['text'])
        
        predicted = result.has_contradiction
        actual = sample['has_contradiction']
        
        if predicted and actual:
            true_positives += 1
        elif predicted and not actual:
            false_positives += 1
        elif not predicted and actual:
            false_negatives += 1
        else:
            true_negatives += 1
        
        results.append({
            'id': sample['id'],
            'predicted': predicted,
            'actual': actual,
            'confidence': result.confidence,
            'correct': predicted == actual
        })
    
    accuracy = (true_positives + true_negatives) / len(test_data) if len(test_data) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }
    
    return metrics, results


# Sample data
SAMPLE_REVIEWS = [
    {
        "id": 1,
        "text": "This laptop is incredibly fast. Boot time is under 10 seconds. However, I find myself waiting 5 minutes just to open Chrome. The performance is unmatched in this price range.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 50), (51, 110)]
    },
    {
        "id": 2,
        "text": "The camera quality is stunning in daylight. Night mode works well too. I've taken beautiful photos at my daughter's evening recital. Great for any lighting condition.",
        "has_contradiction": False,
        "contradiction_spans": []
    },
    {
        "id": 3,
        "text": "I've never had a phone this durable. Dropped it multiple times with no damage. The screen cracked on the first drop though. Build quality is exceptional.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 70), (71, 115)]
    },
    {
        "id": 4,
        "text": "Customer service was unhelpful and rude. They resolved my issue within minutes and even gave me a discount. Worst support experience I've ever had.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 40), (41, 110)]
    },
    {
        "id": 5,
        "text": "The noise cancellation is mediocre at best. I can still hear my coworkers clearly. But honestly, for the price, you can't expect studio-quality isolation.",
        "has_contradiction": False,
        "contradiction_spans": []
    },
    {
        "id": 6,
        "text": "Shipping was lightning fast - arrived in 2 days. The three-week wait was worth it though. Amazon Prime really delivers.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 45), (46, 85)]
    },
    {
        "id": 7,
        "text": "This blender is whisper quiet. My baby sleeps right through it. The noise is so loud I have to wear ear protection. Perfect for early morning smoothies.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 60), (61, 115)]
    },
    {
        "id": 8,
        "text": "Not the cheapest option, but definitely worth the premium price. The quality justifies the cost. You get what you pay for with this brand.",
        "has_contradiction": False,
        "contradiction_spans": []
    }
]


if __name__ == "__main__":
    print("Enhanced Semantic Contradiction Detector - Part 2")
    
    detector = SemanticContradictionDetector()
    
    print("\n")
    print("Running on Sample Data")
    
    for review in SAMPLE_REVIEWS:
        print(f"\nReview {review['id']}")
        print(f"Text: {review['text'][:100]}")
        print(f"Ground Truth: {review['has_contradiction']}")
        
        result = detector.analyze(review["text"])
        
        print(f"\nResult:")
        print(f"  Has Contradiction: {result.has_contradiction}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Explanation: {result.explanation}")
        
        if result.contradicting_pairs:
            print(f"\n  Contradicting Pairs:")
            for i, (sent_a, sent_b) in enumerate(result.contradicting_pairs, 1):
                print(f"    {i}. '{sent_a}'")
                print(f"       '{sent_b}'")
    
    print("\n")
    print("Evaluation Metrics")
    
    metrics, detailed_results = evaluate(detector, SAMPLE_REVIEWS)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1_score']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
