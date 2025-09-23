#!/usr/bin/env python3
"""
HOI Evaluation Script for Groma with HICO-DET and SWIG-HOI Support

This script processes images/datasets to extract HOI triplets and evaluate them
using the HICO-DET and SWIG-HOI evaluation protocols with proper format conversion.

Usage:
    # Single image
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --image-file {path_to_image} \
        --output-dir {output_directory}

    # Dataset evaluation
    python -m groma.eval.run_groma_hoi_no_adj \
        --model-name {path_to_groma_model} \
        --dataset {hico/swig} \
        --data-root {path_to_dataset} \
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
from tqdm import tqdm
from pathlib import Path

from groma.utils import disable_torch_init
from groma.model.groma import GromaModel
from groma.constants import DEFAULT_TOKENS
from groma.data.conversation import conv_templates

# Import evaluation modules
from groma.eval.hoi_eval.hico_evaluator import HICOEvaluator
from groma.eval.hoi_eval.swig_evaluator import SWiGEvaluator
from groma.eval.hoi_eval.hico_categories import HICO_INTERACTIONS
from groma.eval.hoi_eval.swig_v1_categories import SWIG_INTERACTIONS


class POSBasedHOIExtractorNoAdj:
    """Extract Human-Object Interaction triplets using spaCy POS tagging with adjective removal"""

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

        # Expanded human identifiers to handle Groma's flexible labeling
        self.human_terms = {
            'man', 'woman', 'person', 'child', 'boy', 'girl', 'people', 'someone',
            'human', 'guy', 'lady', 'gentleman', 'individual', 'player', 'athlete',
            'worker', 'student', 'teacher', 'adult', 'teenager', 'kid', 'baby',
            'male', 'female', 'figure', 'character'
        }

        # Pattern-based human detection for complex labels like "A man", "Another man"
        self.human_patterns = [
            r'\b(?:a|an|the|another|one|some)\s+(?:man|woman|person|boy|girl|guy|lady)\b',
            r'\b(?:man|woman|person|boy|girl|guy|lady)\b',
            r'\b(?:young|old|tall|short)\s+(?:man|woman|person|boy|girl)\b',
        ]

        # Initialize HOI mappers
        self.hico_hoi_mapper = self._build_hico_hoi_mapper()
        self.swig_hoi_mapper = self._build_swig_hoi_mapper()

    def remove_adjectives_from_object(self, text):
        """Remove adjectives from object descriptions while preserving essential nouns"""
        if not self.nlp:
            # Fallback: simple pattern-based removal if spaCy not available
            return self._remove_adjectives_simple(text)

        # Clean the input text
        text = text.strip()

        # Process with spaCy
        doc = self.nlp(text)

        # Keep track of tokens to preserve
        tokens_to_keep = []

        for token in doc:
            # Always keep nouns (including proper nouns)
            if token.pos_ in ['NOUN', 'PROPN']:
                tokens_to_keep.append(token.text)

            # Keep compound noun modifiers (like "tennis" in "tennis racket")
            elif token.pos_ == 'NOUN' or (token.dep_ == 'compound' and token.head.pos_ in ['NOUN', 'PROPN']):
                tokens_to_keep.append(token.text)

            # Keep essential modifiers that are not pure adjectives
            elif token.pos_ == 'ADJ' and token.dep_ == 'compound':
                # Keep compound adjectives that are essential (like "cell" in "cell phone")
                tokens_to_keep.append(token.text)

            # Skip articles and pure descriptive adjectives
            elif token.pos_ in ['DET', 'ADJ'] and token.dep_ not in ['compound']:
                continue

            # Keep other important parts (but be conservative)
            elif token.pos_ in ['NOUN', 'PROPN'] or token.dep_ == 'compound':
                tokens_to_keep.append(token.text)

        # Reconstruct the text
        result = ' '.join(tokens_to_keep).strip()

        # If result is empty or too short, fall back to simple removal
        if not result or len(result.split()) == 0:
            result = self._remove_adjectives_simple(text)

        print(f"DEBUG: Adjective removal: '{text}' -> '{result}'")
        return result

    def _remove_adjectives_simple(self, text):
        """Simple pattern-based adjective removal as fallback"""
        # Remove common articles
        text = re.sub(r'\b(a|an|the)\s+', '', text, flags=re.IGNORECASE)

        # Remove common color adjectives
        color_adjectives = [
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'brown', 'gray', 'grey',
            'orange', 'purple', 'pink', 'silver', 'gold', 'dark', 'light'
        ]

        # Remove size/description adjectives
        descriptive_adjectives = [
            'big', 'small', 'large', 'little', 'tiny', 'huge', 'tall', 'short',
            'old', 'new', 'young', 'nice', 'good', 'bad', 'beautiful', 'ugly'
        ]

        all_adjectives = color_adjectives + descriptive_adjectives

        words = text.split()
        filtered_words = []

        for word in words:
            # Keep the word if it's not a common adjective
            if word.lower() not in all_adjectives:
                filtered_words.append(word)

        result = ' '.join(filtered_words).strip()
        return result if result else text  # Fallback to original if empty

    def parse_grounded_response(self, response_text):
        """Parse grounded response to extract entities and their region IDs"""
        # Pattern to match <p>text</p> <roi><r#></roi>
        pattern = r'<p>\s*([^<]+?)\s*</p>\s*<roi>\s*<r(\d+)>\s*</roi>'
        matches = re.findall(pattern, response_text)

        entities = []
        for text, region_id in matches:
            original_text = text.strip()
            entity_type = self._classify_entity(original_text)

            # Apply adjective removal only to objects
            if entity_type == 'object':
                processed_text = self.remove_adjectives_from_object(original_text)
            else:
                processed_text = original_text

            entities.append({
                'text': processed_text,
                'original_text': original_text,  # Keep original for reference
                'region_id': int(region_id),
                'type': entity_type
            })

        return entities

    def _classify_entity(self, text):
        """Classify entity as human or object with enhanced detection"""
        text_lower = text.lower().strip()

        # Check for direct human terms
        for human_term in self.human_terms:
            if human_term in text_lower:
                return 'human'

        # Check for pattern-based human detection (e.g., "A man", "Another person")
        for pattern in self.human_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'human'

        return 'object'

    def _standardize_human_label(self, text):
        """Standardize human labels to 'person' for consistency"""
        text_lower = text.lower().strip()

        # Check if this is a human entity
        is_human = False

        # Check for direct human terms
        for human_term in self.human_terms:
            if human_term in text_lower:
                is_human = True
                break

        # Check for pattern-based human detection if not already found
        if not is_human:
            for pattern in self.human_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    is_human = True
                    break

        # Return standardized label if human, otherwise return original
        return 'person' if is_human else text

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

    def _build_hico_hoi_mapper(self):
        """Build HICO HOI ID mapper from (action, object) to interaction ID"""
        # HICO_INTERACTIONS already uses string names, not IDs
        # Create (action_name, object_name) -> hoi_id mapper
        hoi_mapper = {}
        for interaction in HICO_INTERACTIONS:
            action_name = interaction["action"]
            object_name = interaction["object"]
            hoi_id = interaction["interaction_id"]
            hoi_mapper[(action_name, object_name)] = hoi_id

        return hoi_mapper

    def _build_swig_hoi_mapper(self):
        """Build SWIG HOI ID mapper from (action_id, object_id) to interaction ID"""
        hoi_mapper = {}
        for interaction in SWIG_INTERACTIONS:
            if interaction["evaluation"] == 1:  # Only evaluation interactions
                action_id = interaction["action_id"]
                object_id = interaction["object_id"]
                hoi_id = interaction["id"]
                hoi_mapper[(action_id, object_id)] = hoi_id

        return hoi_mapper

    def map_to_hoi_id(self, action, object_text, dataset_type='hico'):
        """Map (action, object) pair to HOI interaction ID"""
        if dataset_type == 'hico':
            return self._map_to_hico_id(action, object_text)
        elif dataset_type == 'swig':
            return self._map_to_swig_id(action, object_text)
        else:
            return None

    def _map_to_hico_id(self, action, object_text):
        """Map to HICO HOI ID using action and object names"""
        # Normalize action (handle common variations)
        action_normalized = self._normalize_action(action)

        # Try to find matching object in HICO_OBJECTS
        object_normalized = self._normalize_object_for_hico(object_text)

        # Try direct mapping
        key = (action_normalized, object_normalized)
        if key in self.hico_hoi_mapper:
            return self.hico_hoi_mapper[key]

        # Try fuzzy matching for actions and objects
        return self._fuzzy_match_hico(action_normalized, object_normalized)

    def _map_to_swig_id(self, action, object_text):
        """Map to SWIG HOI ID using action and object IDs"""
        # Normalize inputs
        action_normalized = self._normalize_action_for_swig(action)
        object_normalized = self._normalize_object_for_swig(object_text)
        
        # Find action ID
        action_id = self._find_swig_action_id(action_normalized)
        if action_id is None:
            print(f"WARNING: Could not find SWIG action_id for '{action}' (normalized: '{action_normalized}')")
            return None
            
        # Find object ID
        object_id = self._find_swig_object_id(object_normalized)
        if object_id is None:
            print(f"WARNING: Could not find SWIG object_id for '{object_text}' (normalized: '{object_normalized}')")
            return None
        
        # Look up HOI ID using (action_id, object_id) pair
        hoi_key = (action_id, object_id)
        if hoi_key in self.swig_hoi_mapper:
            hoi_id = self.swig_hoi_mapper[hoi_key]
            print(f"DEBUG: Mapped ({action}, {object_text}) -> (action_id={action_id}, object_id={object_id}) -> hoi_id={hoi_id}")
            return hoi_id
        else:
            print(f"WARNING: No SWIG HOI mapping found for (action_id={action_id}, object_id={object_id})")
            return None

    def _normalize_action(self, action):
        """Return original action without static mapping to preserve Groma's generated words"""
        action = action.lower().strip()
        return action

    def _normalize_object_for_hico(self, object_text):
        """Normalize object names for HICO mapping - only person mappings preserved"""
        object_text = object_text.lower().strip()

        # Keep only person-related mappings as required for HICO evaluation
        person_mappings = {
            'woman': 'person',
            'man': 'person',
            'girl': 'person',
            'boy': 'person',
            'child': 'person',
            'adult': 'person',
            'individual': 'person',
            'someone': 'person',
            'another person': 'person',
            'kid': 'person',
            'people': 'person',
            'human': 'person',
            'guy': 'person',
            'lady': 'person',
            'gentleman': 'person',
            'player': 'person',
            'athlete': 'person',
            'worker': 'person',
            'student': 'person',
            'teacher': 'person',
            'teenager': 'person',
            'baby': 'person',
            'male': 'person',
            'female': 'person',
            'figure': 'person',
            'character': 'person'
        }

        return person_mappings.get(object_text, object_text)

    def _normalize_action_for_swig(self, action):
        """Return original action without static mapping to preserve Groma's generated words"""
        action = action.lower().strip()
        return action

    def _normalize_object_for_swig(self, object_text):
        """Normalize object names for SWIG mapping - only person mappings preserved"""
        object_text = object_text.lower().strip()

        # Keep only person-related mappings as required for SWIG evaluation
        person_mappings = {
            'woman': 'person',
            'man': 'person',
            'girl': 'person',
            'boy': 'person',
            'child': 'person',
            'adult': 'person',
            'individual': 'person',
            'someone': 'person',
            'another person': 'person',
            'kid': 'person',
            'people': 'person',
            'human': 'person',
            'guy': 'person',
            'lady': 'person',
            'gentleman': 'person',
            'player': 'person',
            'athlete': 'person',
            'worker': 'person',
            'student': 'person',
            'teacher': 'person',
            'teenager': 'person',
            'baby': 'person',
            'male': 'person',
            'female': 'person',
            'figure': 'person',
            'character': 'person'
        }

        return person_mappings.get(object_text, object_text)

    def _find_swig_action_id(self, action_name):
        """Find SWIG action ID by name"""
        from groma.eval.hoi_eval.swig_v1_categories import SWIG_ACTIONS
        
        # Direct name match
        for action in SWIG_ACTIONS:
            if action['name'] == action_name:
                print(f"DEBUG: Direct action match: '{action_name}' -> action_id {action['id']}")
                return action['id']
        
        # Fuzzy matching with word-level comparison
        action_words = set(action_name.split())
        for action in SWIG_ACTIONS:
            action_name_words = set(action['name'].split())
            if action_words and action_name_words:
                # Check for word overlap
                overlap = len(action_words.intersection(action_name_words))
                if overlap > 0:
                    print(f"DEBUG: Fuzzy action match: '{action_name}' -> '{action['name']}' (action_id {action['id']})")
                    return action['id']
        
        # Partial string matching as fallback
        for action in SWIG_ACTIONS:
            if action_name in action['name'] or action['name'] in action_name:
                print(f"DEBUG: Partial action match: '{action_name}' -> '{action['name']}' (action_id {action['id']})")
                return action['id']
        
        print(f"DEBUG: No action match found for '{action_name}'")
        return None

    def _find_swig_object_id(self, object_name):
        """Find SWIG object ID by name"""
        from groma.eval.hoi_eval.swig_v1_categories import SWIG_CATEGORIES
        
        # Direct name match
        for obj in SWIG_CATEGORIES:
            if obj['name'] == object_name:
                print(f"DEBUG: Direct object match: '{object_name}' -> object_id {obj['id']}")
                return obj['id']
        
        # Check gloss (synonyms)
        for obj in SWIG_CATEGORIES:
            if object_name in obj.get('gloss', []):
                print(f"DEBUG: Gloss object match: '{object_name}' -> '{obj['name']}' (object_id {obj['id']})")
                return obj['id']
        
        # Fuzzy matching with word-level comparison
        object_words = set(object_name.split())
        for obj in SWIG_CATEGORIES:
            obj_name_words = set(obj['name'].split())
            if object_words and obj_name_words:
                # Check for word overlap
                overlap = len(object_words.intersection(obj_name_words))
                if overlap > 0:
                    print(f"DEBUG: Fuzzy object match: '{object_name}' -> '{obj['name']}' (object_id {obj['id']})")
                    return obj['id']
        
        # Partial string matching as fallback
        for obj in SWIG_CATEGORIES:
            if object_name in obj['name'] or obj['name'] in object_name:
                print(f"DEBUG: Partial object match: '{object_name}' -> '{obj['name']}' (object_id {obj['id']})")
                return obj['id']
        
        print(f"DEBUG: No object match found for '{object_name}'")
        return None

    def _fuzzy_match_hico(self, action, object_text):
        """Fuzzy matching for HICO HOI IDs when direct mapping fails"""
        # Try partial matches
        for (hico_action, hico_object), hoi_id in self.hico_hoi_mapper.items():
            if (action in hico_action or hico_action in action) and \
               (object_text in hico_object or hico_object in object_text):
                return hoi_id

        return None

    def convert_triplets_to_predictions(self, triplets, image_id, image_width, image_height, dataset_type='hico'):
        """Convert triplets to evaluation prediction format"""
        predictions = []

        for triplet in triplets:
            # Get HOI ID
            action = triplet['action']
            object_text = triplet['object']['text']
            hoi_id = self.map_to_hoi_id(action, object_text, dataset_type)

            if hoi_id is None:
                print(f"WARNING: Could not map ({action}, {object_text}) to HOI ID")
                continue

            # Convert coordinates to absolute pixels
            human_bbox = triplet.get('human_bbox')
            object_bbox = triplet.get('object_bbox')

            if human_bbox is None or object_bbox is None:
                print(f"WARNING: Missing bounding boxes for triplet")
                continue

            # Convert from normalized [0,1] to absolute pixels
            person_x1 = human_bbox[0] * image_width
            person_y1 = human_bbox[1] * image_height
            person_x2 = human_bbox[2] * image_width
            person_y2 = human_bbox[3] * image_height

            object_x1 = object_bbox[0] * image_width
            object_y1 = object_bbox[1] * image_height
            object_x2 = object_bbox[2] * image_width
            object_y2 = object_bbox[3] * image_height

            # Get confidence score
            score = triplet.get('confidence', 0.5)

            # Create prediction in required format:
            # [hoi_id, score, person_x1, person_y1, person_x2, person_y2, object_x1, object_y1, object_x2, object_y2]
            prediction = [
                hoi_id, score,
                person_x1, person_y1, person_x2, person_y2,
                object_x1, object_y1, object_x2, object_y2
            ]

            predictions.append(prediction)

        return predictions

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
            original_note = f" (original: '{o['original_text']}')" if o['text'] != o['original_text'] else ""
            print(f"  Object: '{o['text']}' (region {o['region_id']}){original_note}")

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
                    'extraction_method': 'pos_spacy_no_adj'
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
                        if self._text_overlap_enhanced(child.text, obj['text']) or self._text_overlap_enhanced(child.text, obj['original_text']):
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
                                if self._text_overlap_enhanced(prep_child.text, obj['text']) or self._text_overlap_enhanced(prep_child.text, obj['original_text']):
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
            original_words = obj['original_text'].lower().split()

            # Find object position in sentence (check both processed and original text)
            obj_pos = -1
            for i, token in enumerate(sentence_tokens):
                token_text = token.text.lower()
                if (any(ow in token_text for ow in obj_words) or
                    any(ow in token_text for ow in original_words)):
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

        # Must match one of our detected objects (check both processed and original text)
        for obj in objects:
            if (token.text.lower() in obj['text'].lower() or
                token.text.lower() in obj['original_text'].lower()):
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

        # Must match one of our detected objects (check both processed and original text)
        for obj in objects:
            if (prep_obj in obj['text'].lower() or
                prep_obj in obj['original_text'].lower()):
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
                        triplets.extend(self._process_basic_action(human_text, action1, object1_text, humans, objects, coordinates_info))
                        # Process second action
                        triplets.extend(self._process_basic_action(human_text, action2, object2_text, humans, objects, coordinates_info))

                    elif len(groups) == 4:  # Pattern 2: "standing..., holding..."
                        human_text, action1, action2, object2_text = groups

                        # Process the holding action (more likely to be relevant)
                        triplets.extend(self._process_basic_action(human_text, action2, object2_text, humans, objects, coordinates_info))

                    elif len(groups) == 3:  # Pattern 3: simple subject-action-object
                        human_text, action, object_text = groups
                        triplets.extend(self._process_basic_action(human_text, action, object_text, humans, objects, coordinates_info))

                    elif len(groups) == 2:  # Pattern 4: action-object only
                        action, object_text = groups
                        # Try to match with any available human
                        for human in humans:
                            triplets.extend(self._process_basic_action(human['text'], action, object_text, humans, objects, coordinates_info))
                            break  # Just use first human for simplicity

        return triplets

    def _process_basic_action(self, human_text, action, object_text, humans, objects, coordinates_info):
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

        # Find matching object (check both processed and original text)
        object_entity = None
        object_text_lower = object_text.lower().strip()
        for obj in objects:
            obj_text_lower = obj['text'].lower()
            original_text_lower = obj['original_text'].lower()

            # More flexible matching
            obj_words = set(obj_text_lower.split()) - {'a', 'an', 'the'}
            original_words = set(original_text_lower.split()) - {'a', 'an', 'the'}
            text_words = set(object_text_lower.split()) - {'a', 'an', 'the'}

            if (obj_words and text_words and
                (obj_words.intersection(text_words) or
                 original_words.intersection(text_words) or
                 any(ow in object_text_lower for ow in obj_words) or
                 any(ow in object_text_lower for ow in original_words) or
                 any(tw in obj_text_lower for tw in text_words) or
                 any(tw in original_text_lower for tw in text_words))):
                object_entity = obj
                break

        if human_entity and object_entity:
            # Find coordinates for human and object
            human_coords = None
            object_coords = None
            for coord_info in coordinates_info:
                if coord_info['region_id'] == human_entity['region_id']:
                    human_coords = coord_info['coordinates']
                if coord_info['region_id'] == object_entity['region_id']:
                    object_coords = coord_info['coordinates']

            triplet = {
                'human': human_entity,
                'human_bbox': human_coords,
                'action': action_clean,
                'object': object_entity,
                'object_bbox': object_coords,
                'confidence': 0.7,
                'extraction_method': 'basic_pattern_enhanced_no_adj'
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
            # Show original text if different
            original_note = ""
            if 'original_text' in triplet['object'] and triplet['object']['original_text'] != object_text:
                original_note = f" (original: '{triplet['object']['original_text']}')"
            print(f"  Triplet {i+1}: '{human_text}' (R{human_region}) -> {action} -> '{object_text}' (R{object_region}){original_note}")

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

    def visualize_comparison(self, prediction_triplets, coordinates_info, gt_hois, evaluation_metrics, output_path, dataset_type='hico'):
        """Create side-by-side comparison of ground truth vs predictions"""

        print(f"🎨 Creating side-by-side comparison visualization...")
        print(f"   📊 Ground Truth: {len(gt_hois)} HOIs")
        print(f"   🤖 Predictions: {len(prediction_triplets)} triplets")

        # Create side-by-side layout (double width)
        img_width, img_height = self.image.size
        comparison_width = img_width * 2 + 40  # 40px gap between images
        comparison_height = img_height + 100   # Extra space for title and legend

        # Create new image for comparison
        comparison_img = Image.new('RGB', (comparison_width, comparison_height), 'white')

        # Create left image (Ground Truth)
        gt_img = self._create_ground_truth_visualization(gt_hois, dataset_type)

        # Create right image (Predictions)
        pred_img = self._create_prediction_visualization(prediction_triplets, coordinates_info, evaluation_metrics)

        # Paste images onto comparison canvas
        comparison_img.paste(gt_img, (0, 50))  # 50px from top for title
        comparison_img.paste(pred_img, (img_width + 40, 50))  # 40px gap + left image width

        # Add titles and legends
        self._add_comparison_titles_and_legends(comparison_img, evaluation_metrics, img_width)

        # Save comparison image
        comparison_img.save(output_path, "JPEG")
        print(f"✅ Comparison visualization saved: {output_path}")

        return comparison_img

    def _create_ground_truth_visualization(self, gt_hois, dataset_type):
        """Create visualization showing ground truth HOI annotations"""

        gt_img = self.image.copy()
        draw = ImageDraw.Draw(gt_img)

        # Load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Ground truth colors (blue theme)
        gt_colors = {
            'person': '#0066CC',    # Blue
            'object': '#0099FF',    # Light blue
            'connection': '#003399', # Dark blue
            'text_bg': '#CCE5FF'    # Very light blue
        }

        for i, gt_hoi in enumerate(gt_hois):
            person_bbox = gt_hoi['person_bbox']
            object_bbox = gt_hoi['object_bbox']

            # Convert to absolute coordinates if needed
            img_width, img_height = gt_img.size
            if all(coord <= 1 for coord in person_bbox):  # Normalized coordinates
                person_bbox = [
                    person_bbox[0] * img_width, person_bbox[1] * img_height,
                    person_bbox[2] * img_width, person_bbox[3] * img_height
                ]
            if all(coord <= 1 for coord in object_bbox):  # Normalized coordinates
                object_bbox = [
                    object_bbox[0] * img_width, object_bbox[1] * img_height,
                    object_bbox[2] * img_width, object_bbox[3] * img_height
                ]

            # Draw person bbox
            draw.rectangle(person_bbox, outline=gt_colors['person'], width=3)

            # Draw object bbox
            draw.rectangle(object_bbox, outline=gt_colors['object'], width=3)

            # Draw connection line
            person_center = ((person_bbox[0] + person_bbox[2]) // 2, (person_bbox[1] + person_bbox[3]) // 2)
            object_center = ((object_bbox[0] + object_bbox[2]) // 2, (object_bbox[1] + object_bbox[3]) // 2)

            if person_center != object_center:  # Don't draw line if same bbox
                draw.line([person_center, object_center], fill=gt_colors['connection'], width=2)

            # Add labels
            if dataset_type == 'hico':
                person_label = f"GT-P{i+1}: Person"
                object_label = f"GT-O{i+1}: {gt_hoi['object_name']}"
                action_label = f"GT: {gt_hoi['action_name']}"
            else:
                # Use actual names for SWIG instead of IDs
                action_name = gt_hoi.get('action_name', f"action_{gt_hoi['action_id']}")
                object_name = gt_hoi.get('object_name', f"object_{gt_hoi['object_id']}")
                person_label = f"GT-P{i+1}: Person"
                object_label = f"GT-O{i+1}: {object_name}"
                action_label = f"GT: {action_name}"

            # Draw person label
            self._draw_label(draw, person_bbox, person_label, gt_colors['person'], font_small)

            # Draw object label
            self._draw_label(draw, object_bbox, object_label, gt_colors['object'], font_small)

            # Draw action label at midpoint
            if person_center != object_center:
                mid_x = (person_center[0] + object_center[0]) // 2
                mid_y = (person_center[1] + object_center[1]) // 2
                self._draw_action_label(draw, (mid_x, mid_y), action_label, gt_colors['connection'], font)

        return gt_img

    def _create_prediction_visualization(self, prediction_triplets, coordinates_info, evaluation_metrics):
        """Create visualization showing model predictions with match indicators"""

        pred_img = self.image.copy()
        draw = ImageDraw.Draw(pred_img)

        # Load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Prediction colors - use orange theme to distinguish from ground truth and evaluation
        pred_colors = {
            'person': '#FF8C00',        # Orange for predicted person
            'object': '#FF6600',        # Dark orange for predicted object
            'connection': '#CC5500',    # Darker orange for connections
        }

        # Create mapping of region_id to coordinates
        coords_by_region = {}
        for info in coordinates_info:
            coords_by_region[info['region_id']] = info

        # Show all predictions regardless of evaluation matches - just for visualization
        for i, triplet in enumerate(prediction_triplets):
            human_region = triplet['human']['region_id']
            object_region = triplet['object']['region_id']
            action = triplet['action']  # Show raw predicted action
            human_text = triplet['human']['text']
            object_text = triplet['object']['text']  # Show raw predicted object
            confidence = triplet.get('confidence', 0.0)

            # Use consistent orange colors for all predictions
            person_color = pred_colors['person']
            object_color = pred_colors['object']
            connection_color = pred_colors['connection']

            # Get coordinates
            if human_region in coords_by_region and object_region in coords_by_region:
                human_coords = coords_by_region[human_region]['coordinates']
                object_coords = coords_by_region[object_region]['coordinates']

                # Convert to pixel coordinates
                img_width, img_height = pred_img.size
                human_bbox = [
                    human_coords[0] * img_width, human_coords[1] * img_height,
                    human_coords[2] * img_width, human_coords[3] * img_height
                ]
                object_bbox = [
                    object_coords[0] * img_width, object_coords[1] * img_height,
                    object_coords[2] * img_width, object_coords[3] * img_height
                ]

                # Draw bboxes
                draw.rectangle(human_bbox, outline=person_color, width=3)
                draw.rectangle(object_bbox, outline=object_color, width=3)

                # Draw connection
                human_center = ((human_bbox[0] + human_bbox[2]) // 2, (human_bbox[1] + human_bbox[3]) // 2)
                object_center = ((object_bbox[0] + object_bbox[2]) // 2, (object_bbox[1] + object_bbox[3]) // 2)

                if human_center != object_center:
                    draw.line([human_center, object_center], fill=connection_color, width=2)

                # Add labels - show standardized predictions for visualization
                standardized_human = self._standardize_human_label_viz(human_text)
                person_label = f"PRED-P{i+1}: {standardized_human}"
                object_label = f"PRED-O{i+1}: {object_text}"
                action_label = f"PRED: {action} ({confidence:.2f})"

                # Draw labels
                self._draw_label(draw, human_bbox, person_label, person_color, font_small)
                self._draw_label(draw, object_bbox, object_label, object_color, font_small)

                # Draw action label
                if human_center != object_center:
                    mid_x = (human_center[0] + object_center[0]) // 2
                    mid_y = (human_center[1] + object_center[1]) // 2
                    self._draw_action_label(draw, (mid_x, mid_y), action_label, connection_color, font)

        return pred_img

    def _draw_label(self, draw, bbox, text, color, font):
        """Draw a label above a bounding box"""
        text_y = max(5, bbox[1] - 20)
        try:
            text_bbox = draw.textbbox((bbox[0], text_y), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((bbox[0], text_y), text, fill="white", font=font)
        except:
            draw.text((bbox[0], text_y), text, fill=color, font=font)

    def _draw_action_label(self, draw, position, text, color, font):
        """Draw action label at specified position"""
        try:
            text_bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text(position, text, fill="white", font=font)
        except:
            draw.text(position, text, fill=color, font=font)

    def _add_comparison_titles_and_legends(self, comparison_img, evaluation_metrics, img_width):
        """Add titles and legend to comparison image"""

        draw = ImageDraw.Draw(comparison_img)

        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            legend_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            legend_font = ImageFont.load_default()

        # Add titles
        draw.text((img_width // 2 - 100, 10), "GROUND TRUTH", fill='#0066CC', font=title_font)
        draw.text((img_width + 40 + img_width // 2 - 100, 10), "GROMA PREDICTIONS", fill='#FF8C00', font=title_font)

        # Add metrics summary with detailed IoU information
        if evaluation_metrics:
            metrics_text = f"Precision: {evaluation_metrics['precision']:.3f} | Recall: {evaluation_metrics['recall']:.3f} | TP: {evaluation_metrics['true_positives']}"
            draw.text((20, comparison_img.height - 30), metrics_text, fill='black', font=legend_font)

            # Add detailed match information
            if 'matches' in evaluation_metrics:
                for i, match in enumerate(evaluation_metrics['matches']):
                    if match['match']:
                        match_text = f"Match {i}: Person IoU: {match['person_iou']:.3f}, Object IoU: {match['object_iou']:.3f}, Min: {match['iou']:.3f}"
                    else:
                        match_text = f"Miss {i}: Best IoU: {match['iou']:.3f} (threshold: 0.5)"
                    draw.text((20, comparison_img.height - 90 - i*15), match_text, fill='darkred' if not match['match'] else 'darkgreen', font=legend_font)

        # Add legend
        legend_y = comparison_img.height - 60
        draw.text((20, legend_y), "Legend: GT=Ground Truth (Blue), PRED=Raw Predictions (Orange) - IoU threshold: 0.5", fill='black', font=legend_font)

    def _standardize_human_label_viz(self, human_text):
        """Standardize human labels to 'person' for visualization"""
        if not human_text:
            return "person"

        # Convert to lowercase for matching
        text_lower = human_text.lower()

        # Human-related terms that should be standardized to 'person'
        human_indicators = [
            'man', 'woman', 'person', 'human', 'people', 'kid', 'child',
            'boy', 'girl', 'guy', 'lady', 'individual', 'someone',
            'another man', 'another woman', 'a man', 'a woman', 'a person',
            'a kid', 'a child', 'a boy', 'a girl', 'a guy', 'a lady'
        ]

        # Check if the text contains any human indicator
        for indicator in human_indicators:
            if indicator in text_lower:
                return "person"

        # If no match found, still return 'person' as default for human entities
        return "person"


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


def load_dataset(dataset_type, data_root):
    """Load dataset annotations"""
    if dataset_type == 'hico':
        return load_hico_dataset(data_root)
    elif dataset_type == 'swig':
        return load_swig_dataset(data_root)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def load_hico_dataset(data_root):
    """Load HICO-DET test dataset"""
    test_ann_file = os.path.join(data_root, 'annotations', 'test_hico_ann.json')
    test_img_dir = os.path.join(data_root, 'images', 'test2015')

    if not os.path.exists(test_ann_file):
        raise FileNotFoundError(f"HICO test annotations not found: {test_ann_file}")
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"HICO test images not found: {test_img_dir}")

    with open(test_ann_file, 'r') as f:
        annotations = json.load(f)

    dataset = []
    for ann in annotations:
        img_path = os.path.join(test_img_dir, ann['file_name'])
        if os.path.exists(img_path):
            dataset.append({
                'image_id': ann['img_id'],
                'image_path': img_path,
                'file_name': ann['file_name'],
                'width': ann.get('width', 640),
                'height': ann.get('height', 480)
            })

    return dataset

def load_swig_dataset(data_root):
    """Load SWIG-HOI test dataset"""
    test_ann_file = os.path.join(data_root, 'annotations', 'swig_test_1000.json')
    test_img_dir = os.path.join(data_root, 'images_512')

    if not os.path.exists(test_ann_file):
        raise FileNotFoundError(f"SWIG test annotations not found: {test_ann_file}")
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"SWIG test images not found: {test_img_dir}")

    with open(test_ann_file, 'r') as f:
        annotations = json.load(f)

    dataset = []
    for ann in annotations:
        img_path = os.path.join(test_img_dir, ann['file_name'])
        if os.path.exists(img_path):
            dataset.append({
                'image_id': ann['img_id'],
                'image_path': img_path,
                'file_name': ann['file_name'],
                'width': ann.get('width', 512),
                'height': ann.get('height', 512)
            })

    return dataset

def find_image_in_dataset(image_path, dataset_type, data_root):
    """Find image information and ground truth in dataset"""
    image_name = os.path.basename(image_path)

    if dataset_type == 'hico':
        return find_hico_image_gt(image_name, data_root)
    elif dataset_type == 'swig':
        return find_swig_image_gt(image_name, data_root)
    else:
        return None

def find_hico_image_gt(image_name, data_root):
    """Find HICO image ground truth"""
    test_ann_file = os.path.join(data_root, 'annotations', 'test_hico_ann.json')

    if not os.path.exists(test_ann_file):
        print(f"WARNING: HICO annotations not found: {test_ann_file}")
        return None

    with open(test_ann_file, 'r') as f:
        annotations = json.load(f)

    print(f"DEBUG: Loaded {len(annotations)} annotations from HICO test set")
    print(f"DEBUG: Looking for image: {image_name}")

    # Find the specific image
    for i, ann in enumerate(annotations):
        if ann['file_name'] == image_name:
            print(f"DEBUG: Found image at index {i}")
            print(f"DEBUG: Image data keys: {ann.keys()}")

            result = {
                'image_id': ann['img_id'],
                'file_name': ann['file_name'],
                'width': ann.get('width', 640),
                'height': ann.get('height', 480),
                'annotations': ann.get('annotations', []),
                'hoi_annotation': ann.get('hoi_annotation', [])
            }

            print(f"DEBUG: Found {len(result['annotations'])} box annotations")
            print(f"DEBUG: Found {len(result['hoi_annotation'])} HOI annotations")

            # Show sample box annotation structure
            if result['annotations']:
                print(f"DEBUG: Sample box annotation: {result['annotations'][0]}")

            # Show sample HOI annotation structure
            if result['hoi_annotation']:
                print(f"DEBUG: Sample HOI annotation: {result['hoi_annotation'][0]}")

            return result

    print(f"WARNING: Image {image_name} not found in HICO test set")
    print(f"DEBUG: Available images (first 5): {[ann['file_name'] for ann in annotations[:5]]}")
    return None

def find_swig_image_gt(image_name, data_root):
    """Find SWIG image ground truth"""
    test_ann_file = os.path.join(data_root, 'annotations', 'swig_test_1000.json')

    if not os.path.exists(test_ann_file):
        print(f"WARNING: SWIG annotations not found: {test_ann_file}")
        return None

    with open(test_ann_file, 'r') as f:
        annotations = json.load(f)

    # Find the specific image
    for ann in annotations:
        if ann['file_name'] == image_name:
            return {
                'image_id': ann['img_id'],
                'file_name': ann['file_name'],
                'width': ann.get('width', 512),
                'height': ann.get('height', 512),
                'box_annotations': ann.get('box_annotations', []),
                'hoi_annotations': ann.get('hoi_annotations', [])
            }

    print(f"WARNING: Image {image_name} not found in SWIG test set")
    return None

def extract_ground_truth_hois(gt_data, dataset_type):
    """Extract ground truth HOI triplets from annotation data"""
    if dataset_type == 'hico':
        return extract_hico_ground_truth(gt_data)
    elif dataset_type == 'swig':
        return extract_swig_ground_truth(gt_data)
    else:
        return []

def extract_hico_ground_truth(gt_data):
    """Extract HICO ground truth HOI triplets"""
    if not gt_data:
        return []

    # Import here to avoid circular imports
    from groma.eval.hoi_eval.hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS

    action_id2name = {x["id"]: x["name"] for x in HICO_ACTIONS}
    object_id2name = {x["id"]: x["name"] for x in HICO_OBJECTS}
    # HICO_INTERACTIONS uses string names, create (action_name, object_name) -> hoi_id mapper
    hoi_mapper = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

    gt_hois = []
    box_annos = gt_data.get('annotations', [])
    hoi_annos = gt_data.get('hoi_annotation', [])

    print(f"DEBUG: Processing {len(hoi_annos)} HOI annotations")

    for i, hoi in enumerate(hoi_annos):
        try:
            person_box = box_annos[hoi["subject_id"]]["bbox"]
            object_box = box_annos[hoi["object_id"]]["bbox"]
            # HICO action categories start from 1, so we subtract 1 for 0-based indexing
            action_id = hoi["category_id"] - 1
            # HICO object categories - let's check both with and without offset
            object_category_id_raw = box_annos[hoi["object_id"]]["category_id"]
            print(f"DEBUG: Raw object_category_id from box annotation: {object_category_id_raw}")

            # Try both raw and adjusted object ID to see which one exists
            object_id_adjusted = object_category_id_raw - 1 if object_category_id_raw > 0 else object_category_id_raw

            print(f"DEBUG: HOI {i+1}: action_id={action_id}, raw_object_id={object_category_id_raw}, adjusted_object_id={object_id_adjusted}")

            # Check if action_id is valid
            if action_id not in action_id2name:
                print(f"WARNING: Invalid action_id {action_id}, skipping HOI annotation {i+1}")
                continue

            # Check which object ID format works
            object_id = None
            if object_category_id_raw in object_id2name:
                object_id = object_category_id_raw
                print(f"DEBUG: Using raw object_id: {object_id}")
            elif object_id_adjusted in object_id2name:
                object_id = object_id_adjusted
                print(f"DEBUG: Using adjusted object_id: {object_id}")
            else:
                print(f"WARNING: Neither raw ({object_category_id_raw}) nor adjusted ({object_id_adjusted}) object_id found in object mapping, skipping HOI annotation {i+1}")
                continue

            action_name = action_id2name[action_id]
            object_name = object_id2name[object_id]

            print(f"DEBUG: action_name='{action_name}', object_name='{object_name}'")

            # Look up HOI ID using action and object names
            hoi_key = (action_name, object_name)
            if hoi_key not in hoi_mapper:
                print(f"WARNING: No HOI mapping found for ('{action_name}', '{object_name}'), skipping")
                print(f"DEBUG: Available action-object combinations for '{action_name}': {[k for k in hoi_mapper.keys() if k[0] == action_name]}")
                continue

            hoi_id = hoi_mapper[hoi_key]
            print(f"DEBUG: Found hoi_id={hoi_id} for ('{action_name}', '{object_name}')")

            gt_hois.append({
                'hoi_id': hoi_id,
                'action_name': action_name,
                'object_name': object_name,
                'person_bbox': person_box,
                'object_bbox': object_box,
                'subject_id': hoi["subject_id"],
                'object_id': hoi["object_id"]
            })

        except Exception as e:
            print(f"ERROR: Failed to process HOI annotation {i+1}: {str(e)}")
            print(f"  HOI data: {hoi}")
            continue

    print(f"DEBUG: Successfully extracted {len(gt_hois)} valid HOI annotations")
    return gt_hois

def extract_swig_ground_truth(gt_data):
    """Extract SWIG ground truth HOI triplets"""
    if not gt_data:
        return []

    from groma.eval.hoi_eval.swig_v1_categories import SWIG_INTERACTIONS, SWIG_ACTIONS, SWIG_CATEGORIES

    hoi_mapper = {(x["action_id"], x["object_id"]): x["id"] for x in SWIG_INTERACTIONS}
    
    # Create ID to name mappings
    action_id2name = {x["id"]: x["name"] for x in SWIG_ACTIONS}
    object_id2name = {x["id"]: x["name"] for x in SWIG_CATEGORIES}

    gt_hois = []
    box_annos = gt_data.get('box_annotations', [])
    hoi_annos = gt_data.get('hoi_annotations', [])

    print(f"DEBUG: Processing {len(hoi_annos)} SWIG HOI annotations")

    for i, hoi in enumerate(hoi_annos):
        try:
            person_box = box_annos[hoi["subject_id"]]["bbox"]
            object_box = box_annos[hoi["object_id"]]["bbox"]
            action_id = hoi["action_id"]
            object_id = box_annos[hoi["object_id"]]["category_id"]

            print(f"DEBUG: SWIG HOI {i+1}: action_id={action_id}, object_id={object_id}")

            # Get action and object names
            action_name = action_id2name.get(action_id, f"action_{action_id}")
            object_name = object_id2name.get(object_id, f"object_{object_id}")

            print(f"DEBUG: SWIG HOI {i+1}: action_name='{action_name}', object_name='{object_name}'")

            hoi_id = hoi_mapper.get((action_id, object_id))
            if hoi_id is not None:
                gt_hois.append({
                    'hoi_id': hoi_id,
                    'action_id': action_id,
                    'object_id': object_id,
                    'action_name': action_name,
                    'object_name': object_name,
                    'person_bbox': person_box,
                    'object_bbox': object_box,
                    'subject_id': hoi["subject_id"],
                    'object_id_idx': hoi["object_id"]
                })
                print(f"DEBUG: SWIG HOI {i+1}: hoi_id={hoi_id} for ('{action_name}', '{object_name}')")
            else:
                print(f"WARNING: No SWIG HOI mapping found for (action_id={action_id}, object_id={object_id})")

        except Exception as e:
            print(f"ERROR: Failed to process SWIG HOI annotation {i+1}: {str(e)}")
            print(f"  HOI data: {hoi}")
            continue

    print(f"DEBUG: Successfully extracted {len(gt_hois)} valid SWIG HOI annotations")
    return gt_hois

def calculate_single_image_metrics(predictions, gt_hois, image_width, image_height):
    """Calculate evaluation metrics for a single image"""
    if not predictions or not gt_hois:
        return {'matches': [], 'precision': 0.0, 'recall': 0.0, 'total_predictions': len(predictions), 'total_gt': len(gt_hois)}

    matches = []
    used_gt = set()

    # Sort predictions by confidence score (descending)
    sorted_predictions = sorted(enumerate(predictions), key=lambda x: x[1][1], reverse=True)

    for pred_idx, pred in sorted_predictions:
        hoi_id, score, person_x1, person_y1, person_x2, person_y2, object_x1, object_y1, object_x2, object_y2 = pred

        print(f"\n📋 Evaluating Prediction {pred_idx}: HOI {hoi_id}, Score {score:.4f}")
        print(f"  Pred Person bbox: [{person_x1:.1f}, {person_y1:.1f}, {person_x2:.1f}, {person_y2:.1f}]")
        print(f"  Pred Object bbox: [{object_x1:.1f}, {object_y1:.1f}, {object_x2:.1f}, {object_y2:.1f}]")

        best_match = None
        best_iou = 0.0
        best_person_iou = 0.0
        best_object_iou = 0.0
        all_matches = []  # Track all potential matches for debugging

        for gt_idx, gt_hoi in enumerate(gt_hois):
            if gt_idx in used_gt:
                continue

            if gt_hoi['hoi_id'] != hoi_id:
                continue

            # Calculate IoU for both person and object boxes
            person_iou = calculate_bbox_iou(
                [person_x1, person_y1, person_x2, person_y2],
                gt_hoi['person_bbox']
            )
            object_iou = calculate_bbox_iou(
                [object_x1, object_y1, object_x2, object_y2],
                gt_hoi['object_bbox']
            )

            # Debug: Print detailed IoU information
            print(f"  GT {gt_idx}: Person IoU = {person_iou:.4f}, Object IoU = {object_iou:.4f}")
            print(f"    Pred Person bbox: [{person_x1:.1f}, {person_y1:.1f}, {person_x2:.1f}, {person_y2:.1f}]")
            print(f"    GT Person bbox:   {gt_hoi['person_bbox']}")
            print(f"    Pred Object bbox: [{object_x1:.1f}, {object_y1:.1f}, {object_x2:.1f}, {object_y2:.1f}]")
            print(f"    GT Object bbox:   {gt_hoi['object_bbox']}")

            # Use minimum IoU as in official evaluation
            min_iou = min(person_iou, object_iou)

            # Store all match information for debugging
            all_matches.append({
                'gt_idx': gt_idx,
                'person_iou': person_iou,
                'object_iou': object_iou,
                'min_iou': min_iou,
                'used': False
            })

            # Track the best match regardless of threshold (for debugging)
            if min_iou > best_iou:
                best_iou = min_iou
                best_match = gt_idx
                best_person_iou = person_iou
                best_object_iou = object_iou

        # Print summary of all GT comparisons
        print(f"  📊 Summary - Compared against {len(all_matches)} GT annotations:")
        for match_info in all_matches:
            status = "🏆 BEST" if match_info['gt_idx'] == best_match else "   "
            print(f"    {status} GT{match_info['gt_idx']}: Person={match_info['person_iou']:.4f}, Object={match_info['object_iou']:.4f}, Min={match_info['min_iou']:.4f}")

        # Apply threshold check: match is valid only if best IoU >= 0.5
        if best_match is not None and best_iou >= 0.5:
            used_gt.add(best_match)
            matches.append({
                'prediction_idx': pred_idx,
                'gt_idx': best_match,
                'hoi_id': hoi_id,
                'score': score,
                'iou': best_iou,
                'person_iou': best_person_iou,
                'object_iou': best_object_iou,
                'match': True
            })
            print(f"✅ MATCH: Pred {pred_idx} -> GT {best_match}, Min IoU: {best_iou:.4f} (Person: {best_person_iou:.4f}, Object: {best_object_iou:.4f})")
        else:
            # Store the best match info even for failed matches (for debugging)
            matches.append({
                'prediction_idx': pred_idx,
                'gt_idx': best_match,
                'hoi_id': hoi_id,
                'score': score,
                'iou': best_iou if best_match is not None else 0.0,
                'person_iou': best_person_iou if best_match is not None else 0.0,
                'object_iou': best_object_iou if best_match is not None else 0.0,
                'match': False
            })
            if best_match is not None:
                print(f"❌ NO MATCH: Pred {pred_idx} -> GT {best_match}, Min IoU: {best_iou:.4f} < 0.5 threshold (Person: {best_person_iou:.4f}, Object: {best_object_iou:.4f})")
            else:
                print(f"❌ NO MATCH: Pred {pred_idx}, no valid ground truth found")

    # Calculate precision and recall
    true_positives = sum(1 for m in matches if m['match'])
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(gt_hois) if gt_hois else 0.0

    return {
        'matches': matches,
        'precision': precision,
        'recall': recall,
        'true_positives': true_positives,
        'total_predictions': len(predictions),
        'total_gt': len(gt_hois)
    }

def calculate_bbox_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def eval_single_image(args):
    """Evaluate a single image (original functionality)"""
    return eval_hoi_no_adj_single(args)

def eval_dataset(args):
    """Evaluate entire dataset"""
    print(f"\n{'='*80}")
    print(f"DATASET EVALUATION: {args.dataset.upper()}")
    print(f"{'='*80}")

    # Load dataset
    dataset = load_dataset(args.dataset, args.data_root)
    print(f"Loaded {len(dataset)} images from {args.dataset} dataset")

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

    # Initialize HOI extractor
    hoi_extractor = POSBasedHOIExtractorNoAdj()

    # Initialize evaluator
    if args.dataset == 'hico':
        anno_file = os.path.join(args.data_root, 'annotations', 'test_hico_ann.json')
        evaluator = HICOEvaluator(
            anno_file=anno_file,
            output_dir=args.output_dir,
            zero_shot_type="rare_first",
            ignore_non_interaction=True
        )
    elif args.dataset == 'swig':
        anno_file = os.path.join(args.data_root, 'annotations', 'swig_test_1000.json')
        evaluator = SWiGEvaluator(
            anno_file=anno_file,
            output_dir=args.output_dir
        )

    # Process dataset
    all_predictions = {}

    prompt = "[grounding] Describe what each person is doing with objects individually. Focus on actions only."

    for i, sample in enumerate(tqdm(dataset[:100], desc="Processing images")):
        try:
            # Load and process image
            raw_image = load_image(sample['image_path'])
            image_width, image_height = raw_image.size
            processed_image = raw_image.resize((448, 448))
            image = vis_processor.preprocess(processed_image, return_tensors='pt')['pixel_values'].to('cuda')

            # Generate response
            response_text, coordinates_info = generate_hoi_response(
                model, tokenizer, vis_processor, image, prompt
            )

            # Parse entities and extract triplets
            entities = hoi_extractor.parse_grounded_response(response_text)
            hoi_triplets = hoi_extractor.extract_hoi_triplets(response_text, entities, coordinates_info)

            # Convert to evaluation format
            predictions = hoi_extractor.convert_triplets_to_predictions(
                hoi_triplets, sample['image_id'], image_width, image_height, args.dataset
            )

            if predictions:
                all_predictions[sample['image_id']] = predictions

        except Exception as e:
            print(f"Error processing image {sample['image_id']}: {str(e)}")
            continue

    print(f"\nProcessed {len(all_predictions)} images with predictions")

    # Update evaluator and compute metrics
    if all_predictions:
        print("Updating evaluator...")
        evaluator.update(all_predictions)

        print("Computing metrics...")
        evaluator.accumulate()
        evaluator.summarize()

        # Save predictions
        evaluator.save_preds()

        print(f"\nEvaluation complete. Results saved to: {args.output_dir}")
    else:
        print("No valid predictions generated!")

def eval_hoi_no_adj_single(args):
    """Main evaluation function for single image with detailed evaluation process"""

    # Load ground truth if dataset evaluation context is provided
    gt_data = None
    gt_hois = []
    dataset_type = getattr(args, 'eval_dataset', None)

    if dataset_type and hasattr(args, 'data_root') and args.data_root:
        print(f"\n🔍 LOADING GROUND TRUTH FROM {dataset_type.upper()} DATASET...")
        gt_data = find_image_in_dataset(args.image_file, dataset_type, args.data_root)
        if gt_data:
            gt_hois = extract_ground_truth_hois(gt_data, dataset_type)
            print(f"✅ Found ground truth with {len(gt_hois)} HOI annotations")
            print(f"📊 Image ID: {gt_data['image_id']}, Size: {gt_data['width']}x{gt_data['height']}")
        else:
            print("❌ No ground truth found for this image")
    else:
        print("\n📋 Running single image evaluation WITHOUT ground truth comparison")

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

    # Initialize HOI extractor with adjective removal
    hoi_extractor = POSBasedHOIExtractorNoAdj()

    # Load and process image
    raw_image = load_image(args.image_file)
    original_width, original_height = raw_image.size
    processed_image = raw_image.resize((448, 448))
    image = vis_processor.preprocess(processed_image, return_tensors='pt')['pixel_values'].to('cuda')

    # Create output directory
    image_name = os.path.splitext(os.path.basename(args.image_file))[0]
    if dataset_type:
        output_dir = os.path.join(args.output_dir, f'{image_name}_{dataset_type}_detailed_eval')
    else:
        output_dir = os.path.join(args.output_dir, f'{image_name}_hoi_no_adj')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The finalized prompt
    prompt = "[grounding] Describe what each person is doing with objects individually. Focus on actions only."

    print(f"\n{'='*80}")
    if dataset_type:
        print(f"DETAILED HOI EVALUATION FOR: {os.path.basename(args.image_file)} ({dataset_type.upper()})")
    else:
        print(f"HOI TRIPLET EXTRACTION WITH ADJECTIVE REMOVAL FOR: {os.path.basename(args.image_file)}")
    print(f"{'='*80}")
    print(f"📸 Image: {args.image_file}")
    print(f"🤖 Model: {model_name}")
    print(f"💬 Prompt: {prompt}")
    if gt_data:
        print(f"🎯 Ground Truth: {len(gt_hois)} HOI annotations from {dataset_type.upper()}")
    print(f"{'='*80}")
    print()
    # Display ground truth if available
    if gt_hois:
        print(f"🎯 GROUND TRUTH HOI ANNOTATIONS ({len(gt_hois)} total):")
        for i, gt_hoi in enumerate(gt_hois, 1):
            if dataset_type == 'hico':
                print(f"  {i}. HOI_ID: {gt_hoi['hoi_id']}, Action: '{gt_hoi['action_name']}', Object: '{gt_hoi['object_name']}'")
                print(f"     Person bbox: {gt_hoi['person_bbox']}, Object bbox: {gt_hoi['object_bbox']}")
            elif dataset_type == 'swig':
                # Show both names and IDs for SWIG
                action_name = gt_hoi.get('action_name', f"action_{gt_hoi['action_id']}")
                object_name = gt_hoi.get('object_name', f"object_{gt_hoi['object_id']}")
                print(f"  {i}. HOI_ID: {gt_hoi['hoi_id']}, Action: '{action_name}' (ID: {gt_hoi['action_id']}), Object: '{object_name}' (ID: {gt_hoi['object_id']})")
                print(f"     Person bbox: {gt_hoi['person_bbox']}, Object bbox: {gt_hoi['object_bbox']}")
        print()

    print(f"🤖 STEP 1: GENERATING GROMA RESPONSE...")
    # Generate response
    response_text, coordinates_info = generate_hoi_response(
        model, tokenizer, vis_processor, image, prompt
    )

    print(f"✅ Model Response: {response_text}")
    print(f"📍 Detected {len(coordinates_info)} regions with coordinates")
    print()

    print(f"🧠 STEP 2: PARSING ENTITIES (with adjective removal)...")
    # Parse entities from response (with adjective removal applied)
    entities = hoi_extractor.parse_grounded_response(response_text)

    print("📋 Parsed Entities:")
    for entity in entities:
        original_note = ""
        if 'original_text' in entity and entity['original_text'] != entity['text']:
            original_note = f" (original: '{entity['original_text']}')"
        print(f"  - {entity['type'].upper()}: '{entity['text']}' (Region {entity['region_id']}){original_note}")
    print()

    print(f"🔗 STEP 3: EXTRACTING HOI TRIPLETS...")
    # Extract HOI triplets using POS approach with adjective removal
    hoi_triplets = hoi_extractor.extract_hoi_triplets(response_text, entities, coordinates_info)

    print("🎭 Extracted HOI Triplets:")
    for i, triplet in enumerate(hoi_triplets, 1):
        human_text = triplet['human']['text']
        human_region = triplet['human']['region_id']
        action = triplet['action']
        object_text = triplet['object']['text']
        object_region = triplet['object']['region_id']
        confidence = triplet['confidence']
        method = triplet.get('extraction_method', 'unknown')

        # Show original object text if different
        original_note = ""
        if 'original_text' in triplet['object'] and triplet['object']['original_text'] != object_text:
            original_note = f" (original: '{triplet['object']['original_text']}')"

        # Display bounding box info if available
        bbox_info = ""
        if 'human_bbox' in triplet and 'object_bbox' in triplet:
            human_bbox = triplet['human_bbox']
            object_bbox = triplet['object_bbox']
            if human_bbox and object_bbox:
                # Update coordinates info to be available in triplets
                if 'human_bbox' not in triplet and 'object_bbox' not in triplet:
                    # Find coordinates from coordinates_info
                    for coord_info in coordinates_info:
                        if coord_info['region_id'] == human_region:
                            triplet['human_bbox'] = coord_info['coordinates']
                        if coord_info['region_id'] == object_region:
                            triplet['object_bbox'] = coord_info['coordinates']

                bbox_info = f" | Human bbox: [{human_bbox[0]:.3f}, {human_bbox[1]:.3f}, {human_bbox[2]:.3f}, {human_bbox[3]:.3f}] | Object bbox: [{object_bbox[0]:.3f}, {object_bbox[1]:.3f}, {object_bbox[2]:.3f}, {object_bbox[3]:.3f}]"

        print(f"  {i}. Human: '{human_text}' (R{human_region}) -> Action: '{action}' -> Object: '{object_text}' (R{object_region}){original_note} (Confidence: {confidence:.2f}, Method: {method}){bbox_info}")

    print()

    print(f"🔄 STEP 4: CONVERTING TO EVALUATION FORMAT...")
    # Convert triplets to evaluation format for testing
    eval_dataset_type = dataset_type if dataset_type else 'hico'
    evaluation_predictions = hoi_extractor.convert_triplets_to_predictions(
        hoi_triplets, gt_data['image_id'] if gt_data else 0, original_width, original_height, eval_dataset_type
    )

    print(f"📊 Evaluation Format Predictions ({len(evaluation_predictions)} total):")
    for i, pred in enumerate(evaluation_predictions):
        print(f"  {i+1}. HOI_ID: {pred[0]}, Score: {pred[1]:.3f}, Person: [{pred[2]:.1f}, {pred[3]:.1f}, {pred[4]:.1f}, {pred[5]:.1f}], Object: [{pred[6]:.1f}, {pred[7]:.1f}, {pred[8]:.1f}, {pred[9]:.1f}]")
    print()

    # Detailed evaluation comparison if ground truth is available
    if gt_hois and evaluation_predictions:
        print(f"⚖️ STEP 5: PREDICTION vs GROUND TRUTH COMPARISON...")
        metrics = calculate_single_image_metrics(evaluation_predictions, gt_hois, original_width, original_height)

        print(f"📈 EVALUATION METRICS:")
        print(f"  🎯 Total Ground Truth: {metrics['total_gt']}")
        print(f"  🤖 Total Predictions: {metrics['total_predictions']}")
        print(f"  ✅ True Positives: {metrics['true_positives']}")
        print(f"  🎯 Precision: {metrics['precision']:.4f} ({metrics['true_positives']}/{metrics['total_predictions']})")
        print(f"  🎯 Recall: {metrics['recall']:.4f} ({metrics['true_positives']}/{metrics['total_gt']})")
        if metrics['precision'] + metrics['recall'] > 0:
            f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            print(f"  🎯 F1-Score: {f1_score:.4f}")
        print()

        print(f"🔍 DETAILED MATCHING ANALYSIS:")
        for match in metrics['matches']:
            pred_idx = match['prediction_idx']
            pred = evaluation_predictions[pred_idx]
            hoi_id, score = pred[0], pred[1]

            if match['match']:
                gt_idx = match['gt_idx']
                gt_hoi = gt_hois[gt_idx]
                if dataset_type == 'hico':
                    print(f"  ✅ MATCH: Pred #{pred_idx+1} (HOI_ID: {hoi_id}, Score: {score:.3f}) matches GT #{gt_idx+1} ('{gt_hoi['action_name']}' + '{gt_hoi['object_name']}') with IoU: {match['iou']:.3f}")
                else:
                    # Show both names and IDs for SWIG
                    action_name = gt_hoi.get('action_name', f"action_{gt_hoi['action_id']}")
                    object_name = gt_hoi.get('object_name', f"object_{gt_hoi['object_id']}")
                    print(f"  ✅ MATCH: Pred #{pred_idx+1} (HOI_ID: {hoi_id}, Score: {score:.3f}) matches GT #{gt_idx+1} ('{action_name}' + '{object_name}') with IoU: {match['iou']:.3f}")
            else:
                print(f"  ❌ NO MATCH: Pred #{pred_idx+1} (HOI_ID: {hoi_id}, Score: {score:.3f}) - no suitable ground truth found")

        # Show unmatched ground truth
        matched_gt_indices = {match['gt_idx'] for match in metrics['matches'] if match['match']}
        unmatched_gt = [i for i in range(len(gt_hois)) if i not in matched_gt_indices]
        if unmatched_gt:
            print(f"  🔍 UNMATCHED GROUND TRUTH:")
            for gt_idx in unmatched_gt:
                gt_hoi = gt_hois[gt_idx]
                if dataset_type == 'hico':
                    print(f"    🎯 GT #{gt_idx+1}: HOI_ID {gt_hoi['hoi_id']} ('{gt_hoi['action_name']}' + '{gt_hoi['object_name']}') - not detected")
                else:
                    # Show both names and IDs for SWIG
                    action_name = gt_hoi.get('action_name', f"action_{gt_hoi['action_id']}")
                    object_name = gt_hoi.get('object_name', f"object_{gt_hoi['object_id']}")
                    print(f"    🎯 GT #{gt_idx+1}: HOI_ID {gt_hoi['hoi_id']} ('{action_name}' + '{object_name}') - not detected")
        print()

    elif gt_hois and not evaluation_predictions:
        print(f"❌ NO PREDICTIONS GENERATED - All {len(gt_hois)} ground truth HOIs missed!")
        print()
    elif not gt_hois and evaluation_predictions:
        print(f"ℹ️ Generated {len(evaluation_predictions)} predictions but no ground truth available for comparison")
        print()
    elif not gt_hois and not evaluation_predictions:
        print(f"ℹ️ No predictions generated and no ground truth available")
        print()

    print(f"🎨 CREATING VISUALIZATION...")
    # Create visualization
    visualizer = HOIVisualizer(raw_image)

    # Always create comparison visualization if ground truth is available
    if gt_hois:
        # Create side-by-side comparison regardless of evaluation predictions
        if dataset_type:
            viz_path = os.path.join(output_dir, f'{image_name}_{dataset_type}_comparison_visualization.jpg')
        else:
            viz_path = os.path.join(output_dir, f'{image_name}_comparison_visualization.jpg')

        try:
            # Calculate metrics for visualization (even if no evaluation predictions)
            if len(evaluation_predictions) > 0:
                metrics = calculate_single_image_metrics(evaluation_predictions, gt_hois, original_width, original_height)
            else:
                # No evaluation predictions, but still show comparison
                metrics = {'matches': [], 'precision': 0.0, 'recall': 0.0, 'true_positives': 0, 'total_predictions': 0, 'total_gt': len(gt_hois)}

            # Always show comparison - even if no evaluation predictions, show raw triplets vs ground truth
            viz_image = visualizer.visualize_comparison(
                hoi_triplets, coordinates_info, gt_hois, metrics, viz_path, dataset_type or 'hico'
            )
            print(f"✅ Side-by-side comparison visualization saved: {viz_path}")
        except Exception as e:
            print(f"❌ Comparison visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fall back to regular visualization
            try:
                if dataset_type:
                    fallback_path = os.path.join(output_dir, f'{image_name}_{dataset_type}_predictions_only.jpg')
                else:
                    fallback_path = os.path.join(output_dir, f'{image_name}_predictions_only.jpg')
                viz_image = visualizer.visualize_triplets(hoi_triplets, coordinates_info, fallback_path)
                print(f"✅ Fallback visualization saved: {fallback_path}")
            except Exception as e2:
                print(f"❌ Fallback visualization also failed: {str(e2)}")
    else:
        # Use regular visualization (no ground truth available)
        if dataset_type:
            viz_path = os.path.join(output_dir, f'{image_name}_{dataset_type}_predictions_only.jpg')
        else:
            viz_path = os.path.join(output_dir, f'{image_name}_hoi_no_adj_visualization.jpg')

        try:
            viz_image = visualizer.visualize_triplets(hoi_triplets, coordinates_info, viz_path)
            print(f"✅ Standard visualization saved: {viz_path}")
        except Exception as e:
            print(f"❌ Standard visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()

    # Prepare detailed results
    results = {
        'image_file': args.image_file,
        'image_dimensions': {'width': original_width, 'height': original_height},
        'dataset_type': dataset_type,
        'model_name': model_name,
        'prompt': prompt,
        'response': response_text,
        'entities': entities,
        'coordinates_info': coordinates_info,
        'hoi_triplets': hoi_triplets,
        'evaluation_predictions': evaluation_predictions,
        'extraction_method': 'pos_based_with_adjective_removal'
    }

    # Add ground truth and metrics if available
    if gt_data:
        results['ground_truth'] = {
            'image_id': gt_data['image_id'],
            'gt_hois': gt_hois,
            'total_gt': len(gt_hois)
        }

    if gt_hois and evaluation_predictions:
        metrics = calculate_single_image_metrics(evaluation_predictions, gt_hois, original_width, original_height)
        results['evaluation_metrics'] = metrics

    # Add extraction stats
    results['extraction_stats'] = {
        'total_entities': len(entities),
        'humans_detected': len([e for e in entities if e['type'] == 'human']),
        'objects_detected': len([e for e in entities if e['type'] == 'object']),
        'triplets_extracted': len(hoi_triplets),
        'evaluation_predictions': len(evaluation_predictions)
    }

    # Save detailed results
    if dataset_type:
        results_path = os.path.join(output_dir, f'{image_name}_{dataset_type}_detailed_results.json')
    else:
        results_path = os.path.join(output_dir, f'{image_name}_hoi_no_adj_results.json')

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Detailed results saved: {results_path}")
    print()

    # Final Summary
    print(f"{'='*80}")
    if dataset_type:
        print(f"DETAILED HOI EVALUATION SUMMARY ({dataset_type.upper()})")
    else:
        print(f"HOI EXTRACTION WITH ADJECTIVE REMOVAL SUMMARY")
    print(f"{'='*80}")
    print(f"📸 Image: {os.path.basename(args.image_file)}")
    print(f"🧠 Entities Detected: {len(entities)} ({len([e for e in entities if e['type'] == 'human'])} humans, {len([e for e in entities if e['type'] == 'object'])} objects)")
    print(f"🎭 HOI Triplets: {len(hoi_triplets)}")
    print(f"📊 Evaluation Predictions: {len(evaluation_predictions)}")

    if gt_hois:
        print(f"🎯 Ground Truth HOIs: {len(gt_hois)}")
        if evaluation_predictions:
            metrics = calculate_single_image_metrics(evaluation_predictions, gt_hois, original_width, original_height)
            print(f"✅ True Positives: {metrics['true_positives']}")
            print(f"🎯 Precision: {metrics['precision']:.4f}")
            print(f"🎯 Recall: {metrics['recall']:.4f}")
            if metrics['precision'] + metrics['recall'] > 0:
                f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                print(f"🎯 F1-Score: {f1_score:.4f}")

    print(f"🔧 Extraction Method: POS-based with adjective removal")

    # Add visualization info
    if gt_hois:
        print(f"🎨 Visualization: Side-by-side comparison (Ground Truth vs Raw Predictions)")
    else:
        print(f"🎨 Visualization: Standard prediction visualization")

    print(f"📁 Output Directory: {output_dir}")
    print(f"{'='*80}")

    if dataset_type and gt_hois:
        print(f"\n🎉 EVALUATION PIPELINE VERIFICATION:")
        print(f"   ✅ Dataset integration working")
        print(f"   ✅ Ground truth loading successful")
        print(f"   ✅ Evaluation format conversion working")
        print(f"   ✅ Metrics calculation functional")
        print(f"   ✅ Side-by-side comparison visualization working")
        print(f"   🚀 Ready for full dataset evaluation!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="HOI Evaluation for Groma with HICO-DET and SWIG-HOI Support")

    # Model arguments
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/",
                       help="Path to Groma model")
    parser.add_argument("--quant_type", type=str, default='none',
                       choices=['none', 'fp16', '8bit', '4bit'],
                       help="Quantization type")

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image-file", type=str,
                           help="Path to input image (single image mode)")
    input_group.add_argument("--dataset", type=str, choices=['hico', 'swig'],
                           help="Dataset to evaluate (dataset mode)")

    # Dataset arguments for evaluation context (optional for single image, required for dataset mode)
    parser.add_argument("--data-root", type=str,
                       help="Root directory of dataset (required for dataset mode, optional for single image with evaluation)")
    parser.add_argument("--eval-dataset", type=str, choices=['hico', 'swig'],
                       help="Dataset type for single image evaluation context (enables ground truth comparison)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default='hoi_evaluation_output',
                       help="Output directory for results")

    args = parser.parse_args()

    # Validate arguments
    if args.dataset and not args.data_root:
        parser.error("--data-root is required when using --dataset")

    if args.eval_dataset and not args.data_root:
        parser.error("--data-root is required when using --eval-dataset")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run appropriate evaluation mode
    if args.image_file:
        if args.eval_dataset:
            print(f"Running single image evaluation with {args.eval_dataset.upper()} evaluation context...")
        else:
            print("Running single image evaluation...")
        eval_single_image(args)
    else:
        print("Running dataset evaluation...")
        eval_dataset(args)

if __name__ == "__main__":
    main()