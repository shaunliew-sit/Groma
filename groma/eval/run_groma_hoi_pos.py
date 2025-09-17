#!/usr/bin/env python3
"""
POS-Based HOI Triplet Extraction Script for Groma

This script uses spaCy Part-of-Speech tagging to extract any VERB token as potential actions,
eliminating the need for predefined action vocabularies. It relies on dependency parsing
to ensure verbs connect humans to objects for valid HOI triplets.

Usage:
    python -m groma.eval.run_groma_hoi_pos \
        --model-name {path_to_groma_model} \
        --image-file {path_to_image} \
        --output-dir {output_directory}
"""

import os
import copy
import torch
import argparse
import requests
import json
import re
import spacy
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from transformers.image_transforms import center_to_corners_format
from transformers import AutoTokenizer, AutoImageProcessor, BitsAndBytesConfig
from collections import defaultdict
import numpy as np

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates


class POSBasedHOIExtractor:
    """Extract Human-Object Interaction triplets using spaCy POS tagging"""

    def __init__(self):
        # Load spaCy for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Minimal exclusion list for auxiliary/modal verbs (optional)
        self.skip_verbs = {
            'be', 'have', 'do', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can'
        }

        # Human identifiers
        self.human_terms = {'man', 'woman', 'person', 'child', 'boy', 'girl', 'people', 'someone'}

    def parse_grounded_response(self, response_text):
        """Parse grounded response to extract entities and their region IDs"""
        # Pattern to match <p>text</p> <roi><r#></roi>
        pattern = r'<p>\s*([^<]+?)\s*</p>\s*<roi>\s*<r(\d+)>\s*</roi>'
        matches = re.findall(pattern, response_text)

        entities = []
        for text, region_id in matches:
            entities.append({
                'text': text.strip(),
                'region_id': int(region_id),
                'type': self._classify_entity(text.strip())
            })

        return entities

    def _classify_entity(self, text):
        """Classify entity as human or object"""
        text_lower = text.lower()

        # Check for human terms
        for human_term in self.human_terms:
            if human_term in text_lower:
                return 'human'

        return 'object'

    def extract_hoi_triplets(self, response_text, entities, coordinates_info):
        """Extract HOI triplets using POS-based verb detection"""
        triplets = []

        # Remove grounding markup for text analysis
        clean_text = re.sub(r'<[^>]+>', '', response_text)
        clean_text = clean_text.replace('</s>', '').strip()

        if self.nlp:
            triplets.extend(self._extract_with_pos_spacy(clean_text, entities, coordinates_info))
        else:
            print("WARNING: spaCy not available, falling back to basic pattern matching")
            triplets.extend(self._extract_with_basic_patterns(clean_text, entities, coordinates_info))

        return self._deduplicate_triplets(triplets)

    def _extract_with_pos_spacy(self, text, entities, coordinates_info):
        """Use spaCy POS tagging to extract any VERB as potential action"""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        triplets = []

        # Group entities by type
        humans = [e for e in entities if e['type'] == 'human']
        objects = [e for e in entities if e['type'] == 'object']

        print(f"DEBUG: Found {len(humans)} humans and {len(objects)} objects")
        for h in humans:
            print(f"  Human: '{h['text']}' (region {h['region_id']})")
        for o in objects:
            print(f"  Object: '{o['text']}' (region {o['region_id']})")

        if not humans or not objects:
            return triplets

        print(f"DEBUG: Processing text: {text}")
        print(f"DEBUG: spaCy POS analysis:")

        # Display all tokens with their POS tags and dependencies
        sentence_tokens = list(doc)
        for i, token in enumerate(sentence_tokens):
            print(f"  [{i:2}] {token.text:12} {token.lemma_:12} {token.pos_:8} {token.dep_:10}")

        # Find all VERB tokens (no vocabulary restriction)
        verb_tokens = []
        for token in doc:
            if token.pos_ == 'VERB' and token.lemma_.lower() not in self.skip_verbs:
                verb_tokens.append(token)
                print(f"DEBUG: Found VERB token: '{token.text}' (lemma: '{token.lemma_}', dep: '{token.dep_}')")

        # Display sentence structure analysis
        print(f"DEBUG: Sentence structure analysis:")
        print(f"  Full sentence: '{doc.text}'")
        print(f"  Detected verbs and their expected subjects:")

        # Analyze each verb's context
        for verb in verb_tokens:
            verb_idx = sentence_tokens.index(verb)
            print(f"    Verb '{verb.text}' at pos {verb_idx}:")

            # Show dependency children
            children = list(verb.children)
            if children:
                print(f"      Children: {[(child.text, child.dep_) for child in children]}")
            else:
                print(f"      No direct children")

            # Show head relationship
            if verb.head != verb:
                print(f"      Head: '{verb.head.text}' ({verb.head.pos_})")

        print(f"DEBUG: Expected correct triplets based on sentence structure:")
        self._analyze_expected_triplets(doc, humans, objects)

        # For each verb, try to find human-object connections
        for verb_token in verb_tokens:
            action = verb_token.lemma_.lower()

            # Find human subject for this verb
            human_entity = self._find_human_subject_for_verb(verb_token, humans, doc)

            # Find object for this verb
            object_entity = self._find_object_for_verb(verb_token, objects, doc)

            if human_entity and object_entity:
                # Find corresponding coordinate information
                human_coords = None
                object_coords = None

                for coord_info in coordinates_info:
                    if coord_info['region_id'] == human_entity['region_id']:
                        human_coords = coord_info['coordinates']
                    if coord_info['region_id'] == object_entity['region_id']:
                        object_coords = coord_info['coordinates']

                # Create triplet with bounding box information
                triplet = {
                    'human': human_entity,
                    'human_bbox': human_coords,
                    'action': action,
                    'object': object_entity,
                    'object_bbox': object_coords,
                    'confidence': 0.9,
                    'extraction_method': 'pos_spacy'
                }
                triplets.append(triplet)
                print(f"✅ POS SUCCESS: {human_entity['text']} (R{human_entity['region_id']}) -> {action} -> {object_entity['text']} (R{object_entity['region_id']})")
            else:
                print(f"❌ POS FAILED: verb '{action}' - human: {human_entity is not None}, object: {object_entity is not None}")

        return triplets

    def _find_human_subject_for_verb(self, verb_token, humans, doc):
        """Find human subject for a verb using enhanced dependency parsing"""
        print(f"DEBUG: Looking for human subject for verb '{verb_token.text}' (dep: {verb_token.dep_})")

        # Method 1: Direct subject dependency
        for child in verb_token.children:
            if child.dep_ in ['nsubj', 'nsubjpass']:
                print(f"  Found direct subject dependency: '{child.text}' ({child.dep_})")
                # Match to human entities
                for human in humans:
                    if self._text_overlap_enhanced(child.text, human['text']):
                        print(f"  ✅ Direct subject match: '{human['text']}'")
                        return human

        # Method 2: For adverbial clauses (advcl), determine correct subject
        if verb_token.dep_ == 'advcl':
            print(f"  Verb '{verb_token.text}' is adverbial clause (advcl)")

            # Special handling for different types of adverbial clauses
            sentence_tokens = list(verb_token.sent)
            verb_idx = sentence_tokens.index(verb_token)

            # Check if there's a subordinating conjunction before this verb (like "while")
            subordinating_conj = None
            for i in range(verb_idx - 1, -1, -1):
                if sentence_tokens[i].pos_ == 'SCONJ':  # Subordinating conjunction
                    subordinating_conj = sentence_tokens[i].text.lower()
                    break

            print(f"  Found subordinating conjunction: '{subordinating_conj}'")

            # For "while" clauses, look for the nearest human after the "while"
            if subordinating_conj == 'while':
                print(f"  'While' clause detected - looking for new subject after 'while'")

                # Look for human between "while" and this verb
                while_pos = -1
                for i, token in enumerate(sentence_tokens):
                    if token.text.lower() == 'while':
                        while_pos = i
                        break

                if while_pos != -1:
                    # Find human between "while" and verb
                    # Look for "another man" specifically in while clauses
                    for human in humans:
                        human_words = human['text'].lower().split()

                        # Check each position between "while" and verb
                        for i in range(while_pos + 1, verb_idx):
                            token_text = sentence_tokens[i].text.lower()

                            # Special priority for "another man" in while clauses
                            if 'another' in human['text'].lower() and 'another' in token_text:
                                print(f"  ✅ While-clause subject (another): '{human['text']}' at pos {i}")
                                return human
                            elif any(hw in token_text for hw in human_words):
                                # Store this as a candidate but keep looking for "another man"
                                candidate_human = human
                                candidate_pos = i

                    # If we found "another man", it would have returned already
                    # Otherwise, look for any human match
                    for human in humans:
                        human_words = human['text'].lower().split()
                        for i in range(while_pos + 1, verb_idx):
                            if any(hw in sentence_tokens[i].text.lower() for hw in human_words):
                                print(f"  ✅ While-clause subject: '{human['text']}' at pos {i}")
                                return human

            # First check if advcl has its own explicit subject
            for child in verb_token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    print(f"  Found advcl own subject: '{child.text}'")
                    for human in humans:
                        if self._text_overlap_enhanced(child.text, human['text']):
                            print(f"  ✅ Advcl explicit subject: '{human['text']}'")
                            return human

            # If it's not a while clause and no explicit subject, inherit from main clause
            if subordinating_conj != 'while':
                print(f"  Advcl has no own subject, inheriting from main clause")
                head_verb = verb_token.head
                if head_verb.pos_ == 'VERB':
                    main_subject = self._find_human_subject_for_verb(head_verb, humans, doc)
                    if main_subject:
                        print(f"  ✅ Inherited subject from main clause: '{main_subject['text']}'")
                        return main_subject

        # Method 3: For coordinated verbs (conj)
        if verb_token.dep_ == 'conj':
            print(f"  Verb '{verb_token.text}' is coordinated (conj)")
            head_verb = verb_token.head
            if head_verb.pos_ == 'VERB':
                coord_subject = self._find_human_subject_for_verb(head_verb, humans, doc)
                if coord_subject:
                    print(f"  ✅ Inherited subject from coordinated verb: '{coord_subject['text']}'")
                    return coord_subject

        # Method 4: Context-based analysis with improved clause detection
        sentence = verb_token.sent
        sentence_tokens = list(sentence)
        verb_idx = sentence_tokens.index(verb_token)

        print(f"  Context search: verb '{verb_token.text}' at position {verb_idx}")

        # Improved clause boundary detection
        clause_boundaries = self._find_clause_boundaries(sentence_tokens, verb_idx)
        clause_start, clause_end = clause_boundaries

        print(f"  Clause boundaries: [{clause_start}:{clause_end}] for verb at {verb_idx}")

        # Find the closest human in the same clause, preferring those before the verb
        best_human = None
        best_score = -1

        for human in humans:
            human_words = human['text'].lower().split()

            # Find all positions where this human appears
            human_positions = []
            for i, token in enumerate(sentence_tokens):
                if any(hw in token.text.lower() for hw in human_words):
                    human_positions.append(i)

            for human_pos in human_positions:
                # Check if human is in the same clause
                if clause_start <= human_pos <= clause_end:
                    # Calculate score: prefer humans before verb, closer is better
                    if human_pos < verb_idx:
                        score = 1000 - (verb_idx - human_pos)  # Higher score for closer
                        print(f"  Human '{human['text']}' at pos {human_pos}: score {score} (before verb)")
                    else:
                        score = 100 - (human_pos - verb_idx)   # Lower score for after verb
                        print(f"  Human '{human['text']}' at pos {human_pos}: score {score} (after verb)")

                    if score > best_score:
                        best_score = score
                        best_human = human

        if best_human:
            print(f"  ✅ Context match: '{best_human['text']}' (score: {best_score})")
            return best_human

        print(f"  ❌ No human subject found for '{verb_token.text}'")
        return None

    def _find_clause_boundaries(self, sentence_tokens, verb_idx):
        """Find clause boundaries around a verb position"""
        clause_markers = ['while', 'although', 'though', 'whereas', 'when', 'if', 'because', 'since']

        # Find start of clause (look backwards for clause markers)
        clause_start = 0
        for i in range(verb_idx - 1, -1, -1):
            if sentence_tokens[i].text.lower() in clause_markers:
                clause_start = i + 1
                break

        # Find end of clause (look forwards for clause markers, commas, or sentence end)
        clause_end = len(sentence_tokens) - 1
        for i in range(verb_idx + 1, len(sentence_tokens)):
            if (sentence_tokens[i].text.lower() in clause_markers or
                sentence_tokens[i].text in [',', ';'] or
                sentence_tokens[i].pos_ == 'PUNCT'):
                clause_end = i - 1
                break

        return clause_start, clause_end

    def _analyze_expected_triplets(self, doc, humans, objects):
        """Analyze the sentence to determine expected correct triplets"""
        print(f"  Manual sentence analysis for: '{doc.text}'")

        # Split the sentence by main structural elements
        text = doc.text

        # Look for main clauses separated by "while"
        if ' while ' in text.lower():
            parts = text.split(' while ')
            print(f"    Clause 1: '{parts[0].strip()}'")
            if len(parts) > 1:
                print(f"    Clause 2: '{parts[1].strip()}'")

        # Dynamic analysis based on actual sentence content
        print(f"    Expected triplets based on sentence structure:")

        if ' while ' in text.lower():
            parts = text.split(' while ')

            # Analyze first clause
            clause1 = parts[0].strip()
            if 'holding' in clause1.lower():
                print(f"      1. First human + 'hold' + first object (from clause 1)")

            # Analyze second clause
            if len(parts) > 1:
                clause2 = parts[1].strip()
                if 'sitting' in clause2.lower():
                    print(f"      2. Second human + 'sit' + chair object (from clause 2)")
                if 'holding' in clause2.lower():
                    print(f"      3. Second human + 'hold' + second object (from clause 2)")
        else:
            print(f"      Analyzing single clause sentence")

        print(f"    Available humans: {[h['text'] for h in humans]}")
        print(f"    Available objects: {[o['text'] for o in objects]}")
        print()

    def _find_object_for_verb(self, verb_token, objects, doc):
        """Find object for a verb using dependency parsing with enhanced validation"""
        print(f"DEBUG: Looking for object for verb '{verb_token.text}'")

        # Skip verbs that typically don't have physical objects
        intransitive_contexts = ['stand', 'sit', 'walk', 'run', 'sleep', 'laugh', 'cry', 'smile']
        if verb_token.lemma_.lower() in intransitive_contexts:
            # Check if it's being used intransitively
            has_physical_object = any(child.dep_ == 'dobj' for child in verb_token.children)
            if not has_physical_object:
                print(f"  Verb '{verb_token.lemma_}' used intransitively, checking prepositions carefully")

        # Method 1: Direct object dependency
        for child in verb_token.children:
            if child.dep_ == 'dobj':
                print(f"  Found direct object: '{child.text}'")

                # Validate that this is a real object, not a pronoun or abstract concept
                if self._is_valid_physical_object(child, objects):
                    for obj in objects:
                        if self._text_overlap_enhanced(child.text, obj['text']):
                            print(f"  ✅ Matched valid direct object: '{obj['text']}'")
                            return obj
                else:
                    print(f"  ❌ Rejected direct object '{child.text}' (not physical/valid)")

        # Method 2: Prepositional object (with semantic validation)
        for child in verb_token.children:
            if child.dep_ == 'prep':
                prep = child.text.lower()
                print(f"  Found preposition: '{prep}'")

                # Look for objects in prepositional phrase
                for prep_child in child.children:
                    if prep_child.dep_ == 'pobj':
                        print(f"  Found prepositional object: '{prep_child.text}' (via '{child.text}')")

                        # Validate prepositional object based on preposition and verb
                        if self._is_valid_prepositional_object(verb_token, prep, prep_child, objects):
                            for obj in objects:
                                if self._text_overlap_enhanced(prep_child.text, obj['text']):
                                    print(f"  ✅ Matched valid prep object: '{obj['text']}'")
                                    return obj
                        else:
                            print(f"  ❌ Rejected prep object '{prep_child.text}' via '{prep}' (invalid for {verb_token.text})")

        # Method 3: Contextual object (with strict validation)
        sentence = verb_token.sent
        sentence_tokens = list(sentence)
        verb_idx = sentence_tokens.index(verb_token)

        print(f"  Searching context for valid objects after verb position {verb_idx}")

        for obj in objects:
            obj_words = obj['text'].lower().split()

            # Find object position in sentence
            obj_pos = -1
            for i, token in enumerate(sentence_tokens):
                if any(ow in token.text.lower() for ow in obj_words):
                    obj_pos = i
                    break

            if obj_pos != -1 and obj_pos > verb_idx:
                # Additional semantic validation
                if self._is_semantically_compatible(verb_token, obj):
                    print(f"  ✅ Contextual valid object: '{obj['text']}' at position {obj_pos}")
                    return obj
                else:
                    print(f"  ❌ Contextual object '{obj['text']}' not semantically compatible with '{verb_token.text}'")

        print(f"  ❌ No valid object found for '{verb_token.text}'")
        return None

    def _is_valid_physical_object(self, token, objects):
        """Check if token represents a valid physical object"""
        # Skip pronouns
        if token.pos_ == 'PRON':
            print(f"    Rejected: '{token.text}' is a pronoun")
            return False

        # Skip abstract concepts
        abstract_words = {'idea', 'concept', 'thought', 'feeling', 'emotion', 'time', 'way', 'thing'}
        if token.lemma_.lower() in abstract_words:
            print(f"    Rejected: '{token.text}' is abstract")
            return False

        # Must match one of our detected objects
        for obj in objects:
            if token.text.lower() in obj['text'].lower():
                print(f"    Valid: '{token.text}' matches detected object '{obj['text']}'")
                return True

        print(f"    Rejected: '{token.text}' not in detected objects")
        return False

    def _is_valid_prepositional_object(self, verb_token, prep, prep_obj_token, objects):
        """Validate prepositional objects based on verb-preposition semantics"""
        verb = verb_token.lemma_.lower()
        prep_obj = prep_obj_token.text.lower()

        print(f"    Validating: {verb} + {prep} + {prep_obj}")

        # Skip pronouns as prepositional objects
        if prep_obj_token.pos_ == 'PRON':
            print(f"    Rejected: prep object '{prep_obj}' is pronoun")
            return False

        # Validate common verb-preposition combinations
        valid_combinations = {
            'sit': ['in', 'on'],           # sit in chair, sit on bench
            'stand': ['on', 'by', 'near'], # stand on platform (not "beside him")
            'hold': ['with'],              # hold with hands
            'put': ['in', 'on'],          # put in box, put on table
            'place': ['in', 'on'],        # place in container
        }

        if verb in valid_combinations:
            if prep not in valid_combinations[verb]:
                print(f"    Rejected: '{prep}' invalid for verb '{verb}'")
                return False

        # Special case: "beside him" is not a valid HOI object
        if prep == 'beside' and prep_obj_token.pos_ == 'PRON':
            print(f"    Rejected: 'beside him' is not a physical object interaction")
            return False

        # Must match one of our detected objects
        for obj in objects:
            if prep_obj in obj['text'].lower():
                print(f"    Valid: prep object '{prep_obj}' matches '{obj['text']}'")
                return True

        print(f"    Rejected: prep object '{prep_obj}' not in detected objects")
        return False

    def _is_semantically_compatible(self, verb_token, obj_entity):
        """Check if verb-object combination makes semantic sense"""
        verb = verb_token.lemma_.lower()
        obj_text = obj_entity['text'].lower()

        # Define incompatible combinations
        incompatible = {
            'stand': ['phone', 'cellphone', 'racket'],  # don't stand a phone
            'sit': ['phone', 'cellphone', 'racket'],    # don't sit a phone
            'sleep': ['chair', 'racket', 'phone'],      # don't sleep a chair
        }

        if verb in incompatible:
            for incomp_word in incompatible[verb]:
                if incomp_word in obj_text:
                    print(f"    Rejected: '{verb}' incompatible with '{obj_text}'")
                    return False

        print(f"    Valid: '{verb}' compatible with '{obj_text}'")
        return True

    def _text_overlap_enhanced(self, text1, text2):
        """Enhanced text overlap detection with flexible matching"""
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()

        print(f"    Comparing '{text1_lower}' with '{text2_lower}'")

        # Exact match
        if text1_lower == text2_lower:
            print(f"    ✅ Exact match!")
            return True

        # Partial match (one contains the other)
        if text1_lower in text2_lower or text2_lower in text1_lower:
            print(f"    ✅ Partial match!")
            return True

        # Word-level overlap
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())

        if words1 and words2:
            overlap = len(words1.intersection(words2))
            min_words = min(len(words1), len(words2))

            # Lower threshold for better matching
            if overlap / min_words > 0.3:
                print(f"    ✅ Word overlap match! ({overlap}/{min_words})")
                return True

            # Special cases for single word matches
            if overlap > 0 and min_words == 1:
                print(f"    ✅ Single word match!")
                return True

        # Clean text matching (remove articles/common words)
        clean_text1 = self._clean_text_for_matching(text1_lower)
        clean_text2 = self._clean_text_for_matching(text2_lower)

        if clean_text1 and clean_text2:
            if clean_text1 in clean_text2 or clean_text2 in clean_text1:
                print(f"    ✅ Cleaned text match: '{clean_text1}' vs '{clean_text2}'")
                return True

        print(f"    ❌ No match")
        return False

    def _clean_text_for_matching(self, text):
        """Clean text by removing articles and common words for better matching"""
        stop_words = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'with', 'his', 'her'}
        words = [w for w in text.split() if w not in stop_words]
        return ' '.join(words)

    def _extract_with_basic_patterns(self, text, entities, coordinates_info):
        """Fallback pattern-based extraction when spaCy is not available"""
        triplets = []

        humans = [e for e in entities if e['type'] == 'human']
        objects = [e for e in entities if e['type'] == 'object']

        print(f"DEBUG: Basic pattern matching on text: {text}")
        print(f"DEBUG: Available humans: {[h['text'] for h in humans]}")
        print(f"DEBUG: Available objects: {[o['text'] for o in objects]}")

        # Enhanced pattern matching for complex sentences
        patterns = [
            # Pattern 1: "A man is sitting in a white chair holding a tennis racket"
            r'(A\s+man|another\s+man|A\s+woman)\s+is\s+(\w+ing)\s+(?:in|on|beside|with)\s+([^,]+?)\s+(\w+ing)\s+([^,\s]+(?:\s+[^,\s]+)*)',

            # Pattern 2: "man is standing beside him, holding a cell phone"
            r'(\w+)\s+is\s+(\w+ing)\s+[^,]+,\s+(\w+ing)\s+(a\s+[^,\s]+(?:\s+[^,\s]+)*)',

            # Pattern 3: Simple "subject is action object"
            r'(A\s+man|another\s+man|A\s+woman)\s+is\s+(\w+ing)\s+(?:in|on|with|beside)?\s*([^,\s]+(?:\s+[^,\s]+)*)',

            # Pattern 4: "holding something"
            r'(\w+ing)\s+(a\s+[^,\s]+(?:\s+[^,\s]+)*)',
        ]

        for i, pattern in enumerate(patterns):
            print(f"  Trying pattern {i+1}: {pattern}")
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                groups = match.groups()
                print(f"    Match groups: {groups}")

                if len(groups) >= 2:
                    # Extract components based on pattern
                    if len(groups) == 5:  # Pattern 1: complex sentence
                        human_text, action1, object1_text, action2, object2_text = groups

                        # Process first action
                        triplets.extend(self._process_basic_action(human_text, action1, object1_text, humans, objects))
                        # Process second action
                        triplets.extend(self._process_basic_action(human_text, action2, object2_text, humans, objects))

                    elif len(groups) == 4:  # Pattern 2: "standing..., holding..."
                        human_text, action1, action2, object2_text = groups

                        # Process the holding action (more likely to be relevant)
                        triplets.extend(self._process_basic_action(human_text, action2, object2_text, humans, objects))

                    elif len(groups) == 3:  # Pattern 3: simple subject-action-object
                        human_text, action, object_text = groups
                        triplets.extend(self._process_basic_action(human_text, action, object_text, humans, objects))

                    elif len(groups) == 2:  # Pattern 4: action-object only
                        action, object_text = groups
                        # Try to match with any available human
                        for human in humans:
                            triplets.extend(self._process_basic_action(human['text'], action, object_text, humans, objects))
                            break  # Just use first human for simplicity

        return triplets

    def _process_basic_action(self, human_text, action, object_text, humans, objects):
        """Process a single action from basic pattern matching"""
        single_triplets = []

        # Clean up the action
        action_clean = action.lower().strip()
        if action_clean.endswith('ing'):
            action_clean = action_clean[:-3]

        # Handle irregular verbs
        verb_mappings = {
            'sitt': 'sit', 'runn': 'run', 'gett': 'get', 'cutt': 'cut', 'putt': 'put', 'us': 'use'
        }
        if action_clean in verb_mappings:
            action_clean = verb_mappings[action_clean]

        print(f"    Processing: '{human_text}' -> '{action_clean}' -> '{object_text}'")

        # Find matching human
        human_entity = None
        human_text_lower = human_text.lower()
        for human in humans:
            if (human_text_lower in human['text'].lower() or
                human['text'].lower() in human_text_lower or
                any(word in human['text'].lower() for word in human_text_lower.split())):
                human_entity = human
                break

        # Find matching object
        object_entity = None
        object_text_lower = object_text.lower().strip()
        for obj in objects:
            obj_text_lower = obj['text'].lower()
            # More flexible matching
            obj_words = set(obj_text_lower.split()) - {'a', 'an', 'the'}
            text_words = set(object_text_lower.split()) - {'a', 'an', 'the'}

            if (obj_words and text_words and
                (obj_words.intersection(text_words) or
                 any(ow in object_text_lower for ow in obj_words) or
                 any(tw in obj_text_lower for tw in text_words))):
                object_entity = obj
                break

        if human_entity and object_entity:
            triplet = {
                'human': human_entity,
                'action': action_clean,
                'object': object_entity,
                'confidence': 0.7,
                'extraction_method': 'basic_pattern_enhanced'
            }
            single_triplets.append(triplet)
            print(f"    ✅ BASIC SUCCESS: {human_entity['text']} -> {action_clean} -> {object_entity['text']}")
        else:
            print(f"    ❌ BASIC FAILED: human={human_entity is not None}, object={object_entity is not None}")

        return single_triplets

    def _deduplicate_triplets(self, triplets):
        """Remove duplicate triplets"""
        seen = set()
        unique_triplets = []

        for triplet in triplets:
            key = (triplet['human']['region_id'], triplet['action'], triplet['object']['region_id'])
            if key not in seen:
                seen.add(key)
                unique_triplets.append(triplet)

        return unique_triplets


class HOIVisualizer:
    """Visualize HOI triplets on images"""

    def __init__(self, image):
        self.image = image
        self.original_width, self.original_height = image.size

        # Colors for different elements
        self.colors = {
            'human': 'red',
            'object': 'green',
            'action': 'blue',
            'triplet': 'yellow'
        }

    def visualize_triplets(self, triplets, coordinates_info, output_path):
        """Create visualization of HOI triplets - ONLY show regions involved in triplets"""

        print(f"DEBUG: Visualizing {len(triplets)} triplets")

        # Print detailed triplet information
        print("DEBUG: Triplet details:")
        for i, triplet in enumerate(triplets):
            human_text = triplet['human']['text']
            human_region = triplet['human']['region_id']
            action = triplet['action']
            object_text = triplet['object']['text']
            object_region = triplet['object']['region_id']
            print(f"  Triplet {i+1}: '{human_text}' (R{human_region}) -> {action} -> '{object_text}' (R{object_region})")

        # Create a copy of the image for drawing
        viz_image = self.image.copy()
        draw = ImageDraw.Draw(viz_image)

        # Load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Create mapping of region_id to coordinates
        coords_by_region = {}
        for info in coordinates_info:
            coords_by_region[info['region_id']] = info

        print(f"DEBUG: Available coordinate regions: {sorted(coords_by_region.keys())}")

        # Extract unique regions involved in triplets
        regions_in_triplets = set()
        for triplet in triplets:
            regions_in_triplets.add(triplet['human']['region_id'])
            regions_in_triplets.add(triplet['object']['region_id'])

        print(f"DEBUG: Regions needed for triplets: {sorted(regions_in_triplets)}")

        # Check if all needed regions have coordinates
        missing_regions = regions_in_triplets - set(coords_by_region.keys())
        if missing_regions:
            print(f"WARNING: Missing coordinates for regions: {missing_regions}")
        else:
            print("DEBUG: All triplet regions have coordinates")

        # Draw ALL regions involved in triplets, with proper handling for same-region cases
        # Create a comprehensive mapping of regions to draw
        regions_to_draw = {}

        for triplet in triplets:
            human_region = triplet['human']['region_id']
            human_text = triplet['human']['text']
            object_region = triplet['object']['region_id']
            object_text = triplet['object']['text']
            action = triplet['action']

            print(f"DEBUG: Processing triplet: '{human_text}' (R{human_region}) -> {action} -> '{object_text}' (R{object_region})")

            # Add human region to drawing list
            if human_region not in regions_to_draw:
                regions_to_draw[human_region] = {
                    'type': 'human',
                    'text': human_text,
                    'color': self.colors['human']
                }

            # Add object region to drawing list
            # If same region as human, we'll show both labels
            if object_region not in regions_to_draw:
                regions_to_draw[object_region] = {
                    'type': 'object',
                    'text': object_text,
                    'color': self.colors['object']
                }
            elif object_region == human_region:
                # Same region - combine labels
                existing = regions_to_draw[object_region]
                combined_text = f"{existing['text']} + {object_text}"
                regions_to_draw[object_region] = {
                    'type': 'combined',
                    'text': combined_text,
                    'color': 'purple'  # Special color for combined
                }
                print(f"DEBUG: Combined region {object_region}: {combined_text}")

        print(f"DEBUG: Total regions to draw: {len(regions_to_draw)}")

        # Now draw all the regions
        for region_id, region_info in regions_to_draw.items():
            if region_id not in coords_by_region:
                print(f"WARNING: Region {region_id} not found in coordinates")
                continue

            coords = coords_by_region[region_id]['coordinates']
            w, h = viz_image.size
            pixel_box = [
                coords[0] * w, coords[1] * h, coords[2] * w, coords[3] * h
            ]

            color = region_info['color']
            region_text = region_info['text']
            region_type = region_info['type']

            # Draw bounding box
            draw.rectangle(pixel_box, outline=color, width=3)

            # Draw label
            label = f"R{region_id}: {region_text}"
            text_y = max(5, pixel_box[1] - 25)

            try:
                label_bbox = draw.textbbox((pixel_box[0], text_y), label, font=font_small)
                draw.rectangle(label_bbox, fill=color)
                draw.text((pixel_box[0], text_y), label, fill="white", font=font_small)
            except Exception as e:
                draw.text((pixel_box[0], text_y), label, fill=color, font=font_small)

            print(f"DEBUG: Drew region {region_id} ({region_type}): {region_text}")

        # Draw triplet connections
        for i, triplet in enumerate(triplets):
            human_region = triplet['human']['region_id']
            object_region = triplet['object']['region_id']
            action = triplet['action']
            human_text = triplet['human']['text']
            object_text = triplet['object']['text']

            print(f"DEBUG: Drawing triplet {i+1}: '{human_text}' (R{human_region}) -> {action} -> '{object_text}' (R{object_region})")

            human_coords_info = coords_by_region.get(human_region)
            object_coords_info = coords_by_region.get(object_region)

            if human_coords_info and object_coords_info:
                # Get centers
                human_center = self._get_bbox_center(human_coords_info['coordinates'])
                object_center = self._get_bbox_center(object_coords_info['coordinates'])

                # Skip drawing connection if human and object are at the same position (same region)
                if human_region == object_region:
                    print(f"DEBUG: Skipping connection - same region {human_region}")
                    continue

                # Draw connection
                draw.line([human_center, object_center], fill=self.colors['action'], width=4)

                # Draw action label at midpoint
                mid_x = (human_center[0] + object_center[0]) // 2
                mid_y = (human_center[1] + object_center[1]) // 2

                action_text = f"{action.upper()}"
                action_bbox = draw.textbbox((mid_x, mid_y), action_text, font=font)
                draw.rectangle(action_bbox, fill=self.colors['action'])
                draw.text((mid_x, mid_y), action_text, fill="white", font=font)

                print(f"DEBUG: Drew connection from {human_center} to {object_center} with action '{action}'")
            else:
                print(f"DEBUG: Could not find coordinates for triplet - Human: {human_coords_info is not None}, Object: {object_coords_info is not None}")

        # Save visualization
        viz_image.save(output_path, "JPEG")
        print(f"DEBUG: Saved visualization to {output_path}")

        return viz_image

    def _get_region_description_from_triplets(self, region_id, triplets):
        """Get the correct description for a region from the triplets"""
        # Check if this region is a human in any triplet
        for triplet in triplets:
            if triplet['human']['region_id'] == region_id:
                return triplet['human']['text']

        # Check if this region is an object in any triplet
        for triplet in triplets:
            if triplet['object']['region_id'] == region_id:
                return triplet['object']['text']

        return f"Region {region_id}"

    def _get_bbox_center(self, coords):
        """Get center point of bounding box"""
        w, h = self.original_width, self.original_height

        center_x = int((coords[0] + coords[2]) * w / 2)
        center_y = int((coords[1] + coords[3]) * h / 2)

        return (center_x, center_y)


def load_image(image_file):
    """Load image from file or URL"""
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def generate_hoi_response(model, tokenizer, vis_processor, image, prompt):
    """Generate response using Groma model"""

    # Build conversation
    conversations = []

    # Initial setup
    instruct = "Here is an image with region crops from it. "
    instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
    instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
    answer = 'Thank you for the image! How can I assist you with it?'
    conversations.append((conv_templates['llava'].roles[0], instruct))
    conversations.append((conv_templates['llava'].roles[1], answer))

    # Add the HOI question
    conversations.append((conv_templates['llava'].roles[0], prompt))
    conversations.append((conv_templates['llava'].roles[1], ''))

    prompt_full = conv_templates['llava'].get_prompt(conversations)
    inputs = tokenizer([prompt_full])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    with torch.inference_mode():
        with torch.autocast(device_type="cuda"):
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config,
            )

    output_ids = outputs.sequences
    input_token_len = input_ids.shape[1]

    # Extract coordinates
    pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
    pred_boxes = center_to_corners_format(pred_boxes)
    box_idx_token_ids = model.box_idx_token_ids

    # Decode response
    response_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
    response_text = response_text.strip()

    # Extract coordinates
    coordinates_info = extract_coordinates_from_response(
        response_text, pred_boxes, box_idx_token_ids, output_ids, input_token_len
    )

    return response_text, coordinates_info


def extract_coordinates_from_response(response_text, pred_boxes, box_idx_token_ids, output_ids, input_token_len):
    """Extract coordinate information from grounded response"""
    coordinates_info = []
    output_tokens = output_ids[0, input_token_len:]
    selected_box_inds = []

    for pos, token_id in enumerate(output_tokens):
        if token_id.item() in box_idx_token_ids:
            box_idx = box_idx_token_ids.index(token_id.item())
            if box_idx < len(pred_boxes):
                selected_box_inds.append(box_idx)

    for box_idx in selected_box_inds:
        box_coords = pred_boxes[box_idx].tolist()
        region_token = f"<r{box_idx}>"
        object_description = extract_object_description(response_text, region_token)

        coordinates_info.append({
            'region_id': box_idx,
            'region_token': region_token,
            'coordinates': box_coords,
            'object_description': object_description,
            'type': 'detected_region'
        })

    return coordinates_info


def extract_object_description(response_text, region_token):
    """Extract object description from grounded response"""
    escaped_token = re.escape(region_token)

    patterns = [
        rf'<p>([^<]+)</p><roi>{escaped_token}</roi>',
        rf'<p>([^<]+)</p>\s*<roi>\s*{escaped_token}\s*</roi>',
        rf'<p>([^<]+)</p>.*?{escaped_token}'
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1).strip()

    return f"Region {region_token}"


def eval_hoi_pos(args):
    """Main evaluation function using POS-based approach"""

    # Model setup
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    vis_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    kwargs = {}
    if args.quant_type == 'fp16':
        kwargs['torch_dtype'] = torch.float16
    elif args.quant_type == '8bit':
        kwargs['load_in_8bit'] = True
    elif args.quant_type == '4bit':
        int4_quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4'
        )
        kwargs = {'quantization_config': int4_quant_cfg}

    if args.quant_type == '8bit' or args.quant_type == '4bit':
        model = GromaModel.from_pretrained(model_name, **kwargs)
    else:
        model = GromaModel.from_pretrained(model_name, **kwargs).cuda()
    model.init_special_token_id(tokenizer)

    # Initialize POS-based extractor
    hoi_extractor = POSBasedHOIExtractor()

    # Load and process image
    raw_image = load_image(args.image_file)
    original_width, original_height = raw_image.size
    processed_image = raw_image.resize((448, 448))
    image = vis_processor.preprocess(processed_image, return_tensors='pt')['pixel_values'].to('cuda')

    # Create output directory
    image_name = os.path.splitext(os.path.basename(args.image_file))[0]
    output_dir = os.path.join(args.output_dir, f'{image_name}_hoi_pos')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The finalized prompt
    prompt = "[grounding] Describe what each person is doing with objects individually. Focus on actions only."

    print(f"\n{'='*80}")
    print(f"POS-BASED HOI TRIPLET EXTRACTION FOR: {os.path.basename(args.image_file)}")
    print(f"{'='*80}")
    print(f"Prompt: {prompt}")
    print()

    # Generate response
    response_text, coordinates_info = generate_hoi_response(
        model, tokenizer, vis_processor, image, prompt
    )

    print(f"Response: {response_text}")
    print(f"Detected {len(coordinates_info)} regions")
    print()

    # Parse entities from response
    entities = hoi_extractor.parse_grounded_response(response_text)

    print("Parsed Entities:")
    for entity in entities:
        print(f"  - {entity['type'].upper()}: '{entity['text']}' (Region {entity['region_id']})")
    print()

    # Extract HOI triplets using POS approach
    hoi_triplets = hoi_extractor.extract_hoi_triplets(response_text, entities, coordinates_info)

    print("Extracted HOI Triplets (POS-based):")
    for i, triplet in enumerate(hoi_triplets, 1):
        human_text = triplet['human']['text']
        human_region = triplet['human']['region_id']
        action = triplet['action']
        object_text = triplet['object']['text']
        object_region = triplet['object']['region_id']
        confidence = triplet['confidence']
        method = triplet.get('extraction_method', 'unknown')

        # Display bounding box info if available
        bbox_info = ""
        if 'human_bbox' in triplet and 'object_bbox' in triplet:
            human_bbox = triplet['human_bbox']
            object_bbox = triplet['object_bbox']
            if human_bbox and object_bbox:
                bbox_info = f" | Human bbox: [{human_bbox[0]:.3f}, {human_bbox[1]:.3f}, {human_bbox[2]:.3f}, {human_bbox[3]:.3f}] | Object bbox: [{object_bbox[0]:.3f}, {object_bbox[1]:.3f}, {object_bbox[2]:.3f}, {object_bbox[3]:.3f}]"

        print(f"  {i}. Human: '{human_text}' (R{human_region}) -> Action: '{action}' -> Object: '{object_text}' (R{object_region}) (Confidence: {confidence:.2f}, Method: {method}){bbox_info}")

    print()

    # Create visualization
    visualizer = HOIVisualizer(raw_image)
    viz_path = os.path.join(output_dir, f'{image_name}_hoi_pos_visualization.jpg')

    try:
        viz_image = visualizer.visualize_triplets(hoi_triplets, coordinates_info, viz_path)
        print(f"✅ Visualization saved: {viz_path}")
    except Exception as e:
        print(f"❌ Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # Save results to JSON
    results = {
        'image_file': args.image_file,
        'image_dimensions': {'width': original_width, 'height': original_height},
        'prompt': prompt,
        'response': response_text,
        'entities': entities,
        'coordinates_info': coordinates_info,
        'hoi_triplets': hoi_triplets,
        'extraction_method': 'pos_based',
        'extraction_stats': {
            'total_entities': len(entities),
            'humans_detected': len([e for e in entities if e['type'] == 'human']),
            'objects_detected': len([e for e in entities if e['type'] == 'object']),
            'triplets_extracted': len(hoi_triplets)
        }
    }

    results_path = os.path.join(output_dir, f'{image_name}_hoi_pos_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved: {results_path}")
    print()

    # Summary
    print(f"{'='*80}")
    print(f"POS-BASED EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Entities Detected: {len(entities)} ({len([e for e in entities if e['type'] == 'human'])} humans, {len([e for e in entities if e['type'] == 'object'])} objects)")
    print(f"HOI Triplets: {len(hoi_triplets)}")
    print(f"Extraction Method: POS-based (no vocabulary restriction)")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POS-Based HOI Triplet Extraction with Groma")
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/",
                       help="Path to Groma model")
    parser.add_argument("--image-file", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--output-dir", type=str, default='hoi_pos_output',
                       help="Output directory for results")
    parser.add_argument("--quant_type", type=str, default='none',
                       choices=['none', 'fp16', '8bit', '4bit'],
                       help="Quantization type")

    args = parser.parse_args()

    eval_hoi_pos(args)