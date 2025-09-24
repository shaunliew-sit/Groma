"""HOI Extractor for extracting Human-Object Interaction triplets using NLP analysis."""

import re
import spacy
from collections import defaultdict

from .hico_categories import HICO_INTERACTIONS
from .swig_v1_categories import SWIG_INTERACTIONS


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

    def extract_hoi_triplets(self, response_text, entities, coordinates_info, dataset_type='hico'):
        """Extract HOI triplets using POS-based verb detection with dataset-specific action normalization"""
        triplets = []

        # Remove grounding markup for text analysis
        clean_text = re.sub(r'<[^>]+>', '', response_text)
        clean_text = clean_text.replace('</s>', '').strip()

        if self.nlp:
            triplets.extend(self._extract_with_pos_spacy(clean_text, entities, coordinates_info, dataset_type))
        else:
            print("WARNING: spaCy not available, falling back to basic pattern matching")
            triplets.extend(self._extract_with_basic_patterns(clean_text, entities, coordinates_info, dataset_type))

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
        """Convert action to root form for HICO evaluation using spaCy"""
        action = action.lower().strip()
        return self._convert_verb_to_root(action)

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
        """Convert action to continuous form for SWIG evaluation using morphological rules"""
        action = action.lower().strip()
        return self._convert_verb_to_continuous(action)

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

    def _convert_verb_to_root(self, verb):
        """Convert verb to root form using spaCy lemmatization (for HICO)"""
        if not self.nlp:
            # Fallback: improved heuristics if spaCy not available
            verb = verb.lower().strip()

            # Handle -ing suffix
            if verb.endswith('ing'):
                base = verb[:-3]

                # Handle doubling consonants: running -> run, sitting -> sit
                if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in 'aeiou':
                    return base[:-1]

                # Handle 'e' dropping rule reversal: making -> make, taking -> take
                # Check if adding 'e' would make a valid word pattern
                if len(base) >= 2:
                    # Common patterns where 'e' was dropped before adding 'ing'
                    potential_e_endings = ['ak', 'av', 'ic', 'id', 'ig', 'im', 'in', 'ir', 'is', 'it', 'iv', 'iz']
                    if any(base.endswith(ending) for ending in potential_e_endings):
                        return base + 'e'

                    # Specific common action verbs with 'e' dropping
                    e_drop_verbs = {
                        'mak': 'make',
                        'tak': 'take',
                        'giv': 'give',
                        'hav': 'have',
                        'com': 'come',
                        'writ': 'write',
                        'driv': 'drive',
                        'lov': 'love',
                        'liv': 'live',
                        'mov': 'move',
                        'sav': 'save',
                        'clos': 'close',
                        'us': 'use',
                        'caus': 'cause',
                        'chang': 'change',
                        'exchang': 'exchange',
                        'arrang': 'arrange',
                        'manag': 'manage',
                        'damag': 'damage',
                        'handl': 'handle',
                        'shak': 'shake',
                        'bak': 'bake',
                        'rac': 'race',
                        'fac': 'face',
                        'plac': 'place',
                        'trac': 'trace',
                        'forc': 'force',
                        'danc': 'dance',
                        'balanc': 'balance',
                        'practic': 'practice',
                        'notic': 'notice',
                        'serv': 'serve',
                        'observ': 'observe',
                        'deserv': 'deserve',
                        'reserv': 'reserve',
                        'preserv': 'preserve'
                    }

                    if base in e_drop_verbs:
                        return e_drop_verbs[base]

                # Default: just remove 'ing'
                return base

            # Handle -ed suffix
            elif verb.endswith('ed'):
                if verb.endswith('ied'):
                    return verb[:-3] + 'y'
                else:
                    base = verb[:-2]
                    # Handle doubling consonants: stopped -> stop
                    if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in 'aeiou':
                        return base[:-1]
                    return base

            # No suffix to remove
            return verb

        # Use spaCy for proper lemmatization (preferred method)
        doc = self.nlp(verb.strip())
        if doc and len(doc) > 0:
            return doc[0].lemma_.lower()
        return verb.lower().strip()

    def _convert_verb_to_continuous(self, verb):
        """Convert verb to continuous form (-ing) using morphological rules (for SWIG)"""
        verb = verb.lower().strip()

        # Already in continuous form
        if verb.endswith('ing'):
            return verb

        # Handle special cases and irregular verbs
        irregular_verbs = {
            'be': 'being',
            'have': 'having',
            'do': 'doing',
            'go': 'going',
            'get': 'getting',
            'make': 'making',
            'take': 'taking',
            'come': 'coming',
            'see': 'seeing',
            'know': 'knowing',
            'think': 'thinking',
            'look': 'looking',
            'use': 'using',
            'find': 'finding',
            'give': 'giving',
            'tell': 'telling',
            'work': 'working',
            'call': 'calling',
            'try': 'trying',
            'ask': 'asking',
            'need': 'needing',
            'feel': 'feeling',
            'become': 'becoming',
            'leave': 'leaving',
            'put': 'putting',
            'mean': 'meaning',
            'keep': 'keeping',
            'let': 'letting',
            'begin': 'beginning',
            'seem': 'seeming',
            'help': 'helping',
            'show': 'showing',
            'hear': 'hearing',
            'play': 'playing',
            'run': 'running',
            'move': 'moving',
            'live': 'living',
            'believe': 'believing',
            'hold': 'holding',
            'bring': 'bringing',
            'happen': 'happening',
            'write': 'writing',
            'sit': 'sitting',
            'stand': 'standing',
            'lose': 'losing',
            'pay': 'paying',
            'meet': 'meeting',
            'include': 'including',
            'continue': 'continuing',
            'set': 'setting',
            'learn': 'learning',
            'change': 'changing',
            'lead': 'leading',
            'understand': 'understanding',
            'watch': 'watching',
            'follow': 'following',
            'stop': 'stopping',
            'create': 'creating',
            'speak': 'speaking',
            'read': 'reading',
            'spend': 'spending',
            'grow': 'growing',
            'open': 'opening',
            'walk': 'walking',
            'win': 'winning',
            'teach': 'teaching',
            'offer': 'offering',
            'remember': 'remembering',
            'love': 'loving',
            'consider': 'considering',
            'appear': 'appearing',
            'buy': 'buying',
            'serve': 'serving',
            'die': 'dying',
            'send': 'sending',
            'build': 'building',
            'stay': 'staying',
            'fall': 'falling',
            'cut': 'cutting',
            'reach': 'reaching',
            'kill': 'killing',
            'raise': 'raising',
            'pass': 'passing',
            'sell': 'selling',
            'decide': 'deciding',
            'return': 'returning',
            'explain': 'explaining',
            'hope': 'hoping',
            'develop': 'developing',
            'carry': 'carrying',
            'break': 'breaking',
            'receive': 'receiving',
            'agree': 'agreeing',
            'support': 'supporting',
            'hit': 'hitting',
            'produce': 'producing',
            'eat': 'eating',
            'cover': 'covering',
            'catch': 'catching',
            'draw': 'drawing'
        }

        if verb in irregular_verbs:
            return irregular_verbs[verb]

        # Apply general morphological rules
        # Rule 1: CVC pattern (consonant-vowel-consonant) -> double final consonant
        if len(verb) >= 3 and verb[-1] not in 'aeiou' and verb[-2] in 'aeiou' and verb[-3] not in 'aeiou':
            if verb[-1] not in 'wxyz':  # Don't double w, x, y, z
                return verb + verb[-1] + 'ing'

        # Rule 2: Words ending in 'e' (except 'ee', 'oe', 'ye') -> drop 'e' and add 'ing'
        if verb.endswith('e') and not verb.endswith(('ee', 'oe', 'ye')):
            return verb[:-1] + 'ing'

        # Rule 3: Words ending in 'ie' -> change 'ie' to 'y' and add 'ing'
        if verb.endswith('ie'):
            return verb[:-2] + 'ying'

        # Rule 4: Default -> just add 'ing'
        return verb + 'ing'

    def _find_swig_action_id(self, action_name):
        """Find SWIG action ID by name"""
        from .swig_v1_categories import SWIG_ACTIONS

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
        from .swig_v1_categories import SWIG_CATEGORIES

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

    def prepare_visualization_triplets(self, triplets, dataset_type='hico'):
        """Prepare all triplets for visualization with mapping status"""
        visualization_triplets = []

        for triplet in triplets:
            # Create enhanced triplet copy
            viz_triplet = triplet.copy()

            # Check if action can be mapped to HOI ID
            action = triplet['action']
            original_action = triplet.get('original_action', action)
            object_text = triplet['object']['text']

            hoi_id = self.map_to_hoi_id(action, object_text, dataset_type)

            # Add mapping metadata
            viz_triplet['mapping_status'] = 'mapped' if hoi_id is not None else 'unmapped'
            viz_triplet['hoi_id'] = hoi_id
            viz_triplet['evaluation_eligible'] = hoi_id is not None

            # Add warning message for unmapped actions
            if hoi_id is None:
                if original_action != action:
                    viz_triplet['warning'] = f"Action ({original_action} → {action}, {object_text}) not found in {dataset_type.upper()} dataset"
                else:
                    viz_triplet['warning'] = f"Action ({action}, {object_text}) not found in {dataset_type.upper()} dataset"

            visualization_triplets.append(viz_triplet)

        return visualization_triplets

    def convert_triplets_to_predictions(self, triplets, image_id, image_width, image_height, dataset_type='hico'):
        """Convert triplets to evaluation prediction format (only mappable actions)"""
        predictions = []
        unmapped_count = 0

        for triplet in triplets:
            # Action is already normalized in the triplet
            action = triplet['action']
            original_action = triplet.get('original_action', action)
            object_text = triplet['object']['text']

            hoi_id = self.map_to_hoi_id(action, object_text, dataset_type)

            if hoi_id is None:
                unmapped_count += 1
                if original_action != action:
                    print(f"⚠️  UNMAPPED: Action ({original_action} → {action}, {object_text}) not in {dataset_type.upper()} dataset - skipping evaluation")
                else:
                    print(f"⚠️  UNMAPPED: Action ({action}, {object_text}) not in {dataset_type.upper()} dataset - skipping evaluation")
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

        # Report summary
        total_triplets = len(triplets)
        mapped_count = len(predictions)
        if unmapped_count > 0:
            print(f"📊 Conversion Summary: {mapped_count} evaluable, {unmapped_count} unmappable (total: {total_triplets} triplets)")

        return predictions

    def _extract_with_pos_spacy(self, text, entities, coordinates_info, dataset_type='hico'):
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

                # Normalize action based on dataset requirements
                original_action = action
                if dataset_type == 'hico':
                    normalized_action = self._normalize_action(original_action)
                elif dataset_type == 'swig':
                    normalized_action = self._normalize_action_for_swig(original_action)
                else:
                    normalized_action = original_action

                # Create triplet with normalized action as primary
                triplet = {
                    'human': human_entity,
                    'human_bbox': human_coords,
                    'action': normalized_action,
                    'original_action': original_action,
                    'object': object_entity,
                    'object_bbox': object_coords,
                    'confidence': 0.9,
                    'extraction_method': 'pos_spacy_no_adj'
                }
                triplets.append(triplet)
                action_display = f"{normalized_action}" if original_action == normalized_action else f"{normalized_action} [{original_action} → {normalized_action}]"
                print(f"✅ POS SUCCESS: {human_entity['text']} (R{human_entity['region_id']}) -> {action_display} -> {object_entity['text']} (R{object_entity['region_id']})")
            else:
                print(f"❌ POS FAILED: verb '{action}' - human: {human_entity is not None}, object: {object_entity is not None}")

        return triplets

    def _find_human_subject_for_verb(self, verb_token, humans, doc):
        """Find human subject for a verb using enhanced dependency parsing"""
        print(f"DEBUG: Looking for human subject for verb '{verb_token.text}' (dep: {verb_token.dep_})")

        # Method 1: Direct subject dependency with contextual matching
        for child in verb_token.children:
            if child.dep_ in ['nsubj', 'nsubjpass']:
                print(f"  Found direct subject dependency: '{child.text}' ({child.dep_})")

                # Use enhanced contextual matching for ALL verbs, not just adverbial clauses
                print(f"  Using enhanced contextual matching for verb '{verb_token.text}' with subject '{child.text}'")
                contextual_human = self._find_best_human_match(verb_token, child, humans)
                if contextual_human:
                    print(f"  ✅ Best contextual match: '{contextual_human['text']}'")
                    return contextual_human

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

        # Method 4: Position-based subject detection using enhanced scoring
        print(f"  ⚠️ DEPENDENCY PARSING FAILED - Using position-based scoring for verb '{verb_token.text}'")

        # Use the new scoring system for fallback case
        # Create a dummy subject (the verb itself) for position-based matching
        dummy_subject = verb_token  # Use verb as subject for position-based scoring

        sentence_tokens = list(verb_token.sent)
        verb_idx = sentence_tokens.index(verb_token)
        print(f"  Verb '{verb_token.text}' found at position {verb_idx}")

        best_human = self._find_best_human_match_position_only(verb_token, verb_idx, sentence_tokens, humans)
        if best_human:
            print(f"  ✅ Position-based fallback match: '{best_human['text']}'")
            return best_human

        # Method 5: If everything else fails, return None instead of humans[0]
        print(f"  ❌ ALL MATCHING METHODS FAILED - No suitable human found for verb '{verb_token.text}'")
        return None  # Return None instead of humans[0] to avoid wrong assignments

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

    def _find_contextual_human_for_advcl(self, verb_token, subject_child, humans):
        """Find the correct human for an adverbial clause by looking at context"""
        print(f"    DEBUG: Finding contextual human for advcl verb '{verb_token.text}' with subject '{subject_child.text}'")

        # Get sentence tokens and find positions
        sentence_tokens = list(verb_token.sent)
        verb_idx = sentence_tokens.index(verb_token)
        subject_idx = sentence_tokens.index(subject_child)

        print(f"    Verb '{verb_token.text}' at position {verb_idx}")
        print(f"    Subject '{subject_child.text}' at position {subject_idx}")

        # Look for qualifying words before the subject (like "another", "third")
        qualifier_words = []

        # Check tokens before the subject for qualifiers
        search_start = max(0, subject_idx - 3)  # Look up to 3 tokens before
        for i in range(search_start, subject_idx):
            token_text = sentence_tokens[i].text.lower()
            if token_text in ['another', 'third', 'second', 'other', 'next']:
                qualifier_words.append(token_text)
                print(f"    Found qualifier '{token_text}' at position {i}")

        # Try to match humans based on qualifiers found
        for human in humans:
            human_text = human['text'].lower()
            print(f"    Checking human: '{human['text']}'")

            # If we found qualifiers, prioritize humans that contain them
            if qualifier_words:
                for qualifier in qualifier_words:
                    if qualifier in human_text:
                        print(f"    ✅ Matched human '{human['text']}' based on qualifier '{qualifier}'")
                        return human

            # Fallback: check if this human contains the subject word
            if subject_child.text.lower() in human_text:
                print(f"    Potential match: '{human['text']}' contains '{subject_child.text}'")
                # But only return it if no qualifiers were found, or if this human is the most specific
                if not qualifier_words:
                    print(f"    ✅ Matched human '{human['text']}' (no qualifiers found)")
                    return human

        print(f"    ❌ No contextual match found for adverbial clause")
        return None

    def _find_best_human_match(self, verb_token, subject_child, humans):
        """Find the best human match using position-aware contextual scoring"""
        print(f"    🔍 Finding best human match for verb '{verb_token.text}' with subject '{subject_child.text}'")

        # Get sentence structure
        sentence_tokens = list(verb_token.sent)
        verb_idx = sentence_tokens.index(verb_token)
        subject_idx = sentence_tokens.index(subject_child)

        print(f"    Verb '{verb_token.text}' at position {verb_idx}")
        print(f"    Subject '{subject_child.text}' at position {subject_idx}")

        # Score all potential human matches
        human_scores = []

        for human in humans:
            score = self._calculate_human_match_score(
                human, verb_token, subject_child, verb_idx, subject_idx, sentence_tokens
            )
            human_scores.append((human, score))
            print(f"    Human '{human['text']}' score: {score:.2f}")

        # Sort by score (highest first) and return best match
        if human_scores:
            human_scores.sort(key=lambda x: x[1], reverse=True)
            best_human, best_score = human_scores[0]

            # Only return if score is above threshold
            if best_score > 0:
                print(f"    ✅ Best match: '{best_human['text']}' with score {best_score:.2f}")
                return best_human
            else:
                print(f"    ❌ No human scored above threshold (best: {best_score:.2f})")

        return None

    def _calculate_human_match_score(self, human, verb_token, subject_child, verb_idx, subject_idx, sentence_tokens):
        """Calculate a score for how well a human matches a verb's subject"""
        score = 0.0
        human_text = human['text'].lower()
        subject_text = subject_child.text.lower()

        print(f"      Scoring '{human['text']}' for subject '{subject_child.text}'")

        # 1. Text matching quality (most important)
        if human_text == subject_text:
            score += 50.0  # Exact match
            print(f"        +50 exact text match")
        elif subject_text in human_text:
            # Check if it's a meaningful partial match vs just 'man'
            if len(subject_text) > 3 or human_text == subject_text:  # Avoid matching just 'man'
                score += 30.0  # Good partial match
                print(f"        +30 good partial match")
            else:
                score += 5.0   # Weak partial match
                print(f"        +5 weak partial match")
        elif human_text in subject_text:
            score += 10.0  # Reverse partial match
            print(f"        +10 reverse partial match")
        else:
            # Check word overlap
            human_words = set(human_text.split())
            subject_words = set(subject_text.split())
            overlap = len(human_words.intersection(subject_words))
            if overlap > 0:
                score += overlap * 5.0
                print(f"        +{overlap * 5} word overlap")

        # 2. Position-based scoring - find human position in sentence
        human_positions = self._find_human_positions_in_sentence(human, sentence_tokens)
        if human_positions:
            # Use closest position to the subject
            closest_human_pos = min(human_positions, key=lambda pos: abs(pos - subject_idx))
            distance = abs(closest_human_pos - subject_idx)

            # Strong bonus for being very close to the subject
            if distance <= 2:
                score += 20.0
                print(f"        +20 very close position (distance: {distance})")
            elif distance <= 5:
                score += 10.0
                print(f"        +10 close position (distance: {distance})")
            else:
                # Penalty for being far
                penalty = min(distance - 5, 10)  # Cap penalty at 10
                score -= penalty
                print(f"        -{penalty} far position penalty (distance: {distance})")

        # 3. Contextual qualifier bonus
        # Look for qualifiers near the subject
        search_start = max(0, subject_idx - 3)
        search_end = min(len(sentence_tokens), subject_idx + 3)

        qualifiers_found = []
        for i in range(search_start, search_end):
            token_text = sentence_tokens[i].text.lower()
            if token_text in ['another', 'third', 'fourth', 'second', 'other', 'next']:
                qualifiers_found.append(token_text)

        for qualifier in qualifiers_found:
            if qualifier in human_text:
                score += 25.0  # Big bonus for matching qualifier
                print(f"        +25 qualifier match: '{qualifier}'")

        # 4. Sentence structure bonus
        # Bonus for being in the same clause as the verb
        if human_positions:
            closest_human_pos = min(human_positions, key=lambda pos: abs(pos - verb_idx))
            verb_distance = abs(closest_human_pos - verb_idx)

            if verb_distance <= 5:
                score += 15.0
                print(f"        +15 same clause bonus (verb distance: {verb_distance})")

        final_score = max(0.0, score)  # Don't allow negative scores
        print(f"      Final score: {final_score:.2f}")
        return final_score

    def _find_best_human_match_position_only(self, verb_token, verb_idx, sentence_tokens, humans):
        """Find best human match based only on position and context (no subject to match)"""
        print(f"    🎯 Position-only matching for verb '{verb_token.text}' at position {verb_idx}")

        human_scores = []

        for human in humans:
            score = 0.0
            human_text = human['text'].lower()

            print(f"      Scoring '{human['text']}' for position-based matching")

            # 1. Find human positions in sentence
            human_positions = self._find_human_positions_in_sentence(human, sentence_tokens)

            if human_positions:
                # Use closest position to verb
                closest_pos = min(human_positions, key=lambda pos: abs(pos - verb_idx))
                distance = abs(closest_pos - verb_idx)

                # Score based on proximity to verb
                if distance <= 3:
                    score += 30.0  # Very close
                    print(f"        +30 very close to verb (distance: {distance})")
                elif distance <= 7:
                    score += 20.0  # Close
                    print(f"        +20 close to verb (distance: {distance})")
                elif distance <= 15:
                    score += 10.0  # Moderate
                    print(f"        +10 moderate distance (distance: {distance})")
                else:
                    score += 5.0   # Far but still considered
                    print(f"        +5 far from verb (distance: {distance})")

                # 2. Look for contextual qualifiers near the human
                search_start = max(0, closest_pos - 3)
                search_end = min(len(sentence_tokens), closest_pos + 3)

                qualifiers_found = []
                for i in range(search_start, search_end):
                    token_text = sentence_tokens[i].text.lower()
                    if token_text in ['another', 'third', 'fourth', 'second', 'other', 'next']:
                        qualifiers_found.append(token_text)

                # Bonus for having qualifiers that make this human distinct
                for qualifier in qualifiers_found:
                    if qualifier in human_text:
                        score += 20.0  # Qualifier match
                        print(f"        +20 qualifier match: '{qualifier}'")

                # 3. Priority bonus for specific human types
                if 'another' in human_text:
                    score += 15.0
                    print(f"        +15 'another' priority bonus")
                elif 'third' in human_text:
                    score += 15.0
                    print(f"        +15 'third' priority bonus")
                elif 'fourth' in human_text:
                    score += 15.0
                    print(f"        +15 'fourth' priority bonus")

            else:
                print(f"        No positions found for '{human['text']}'")

            final_score = max(0.0, score)
            human_scores.append((human, final_score))
            print(f"      Position-only score: {final_score:.2f}")

        # Return best scoring human if above threshold
        if human_scores:
            human_scores.sort(key=lambda x: x[1], reverse=True)
            best_human, best_score = human_scores[0]

            if best_score > 0:
                print(f"    ✅ Best position-only match: '{best_human['text']}' with score {best_score:.2f}")
                return best_human

        print(f"    ❌ No suitable position-based match found")
        return None

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

    def _find_human_positions_in_sentence(self, human, sentence_tokens):
        """Find all positions where a human entity appears in the sentence"""
        positions = []
        human_text = human['text'].lower()
        human_words = human_text.split()

        print(f"  DEBUG: Looking for '{human['text']}' in sentence")

        # Method 1: Look for consecutive word matches (most reliable)
        if len(human_words) >= 2:  # Multi-word humans like "another man", "third man"
            for i in range(len(sentence_tokens) - len(human_words) + 1):
                # Check if consecutive tokens match human words exactly
                match = True
                for j, human_word in enumerate(human_words):
                    token_text = sentence_tokens[i + j].text.lower()
                    if human_word != token_text:
                        match = False
                        break
                if match:
                    positions.append(i)
                    print(f"    Found multi-word match at position {i}: {[t.text for t in sentence_tokens[i:i+len(human_words)]]}")

        # Method 2: Look for single word matches (for simple cases)
        else:
            for i, token in enumerate(sentence_tokens):
                if token.text.lower() == human_text:
                    positions.append(i)
                    print(f"    Found single-word match at position {i}: '{token.text}'")

        # Method 3: Fuzzy matching for partial matches (fallback)
        if not positions:
            print(f"    No exact matches found, trying fuzzy matching...")
            for i, token in enumerate(sentence_tokens):
                if any(hw in token.text.lower() for hw in human_words if len(hw) > 2):
                    positions.append(i)
                    print(f"    Found fuzzy match at position {i}: '{token.text}' matches part of '{human['text']}'")

        print(f"    Final positions for '{human['text']}': {positions}")
        return list(set(positions))  # Remove duplicates

    def _calculate_human_verb_relevance(self, human, verb_token, human_pos, verb_pos, sentence_tokens):
        """Calculate relevance score between human and verb based on context"""
        relevance = 0.0
        distance = abs(human_pos - verb_pos)
        human_text = human['text'].lower()

        print(f"    Calculating relevance: '{human['text']}' (pos {human_pos}) -> '{verb_token.text}' (pos {verb_pos}), distance: {distance}")

        # STRONG bonus for proximity (exponential decay)
        if distance <= 3:
            proximity_score = 20.0  # Very close
        elif distance <= 6:
            proximity_score = 10.0  # Close
        else:
            proximity_score = max(0, 8 - distance)  # Diminishing returns
        relevance += proximity_score

        # STRONG bonus for human before verb (typical subject position)
        if human_pos < verb_pos:
            relevance += 15.0
            print(f"      +15 for subject-before-verb pattern")

            # Check if there's an auxiliary verb or "is" between human and main verb
            between_tokens = sentence_tokens[human_pos:verb_pos]
            has_auxiliary = any(token.pos_ == 'AUX' or token.text.lower() in ['is', 'are', 'was', 'were'] for token in between_tokens)
            if has_auxiliary:
                relevance += 10.0
                print(f"      +10 for auxiliary verb between human and main verb")

        # HUGE bonus for specific human qualifiers that indicate distinct subjects
        if 'another' in human_text:
            relevance += 25.0
            print(f"      +25 for 'another' qualifier")
        elif 'third' in human_text:
            relevance += 25.0
            print(f"      +25 for 'third' qualifier")
        elif 'second' in human_text:
            relevance += 20.0
            print(f"      +20 for 'second' qualifier")

        # Check for sentence boundary markers that separate clauses
        between_start = min(human_pos, verb_pos)
        between_end = max(human_pos, verb_pos)
        between_tokens = sentence_tokens[between_start:between_end]

        # Look for conjunctions, commas, or other clause separators
        has_clause_separator = False
        for token in between_tokens:
            if token.text.lower() in ['and', 'but', 'while', ','] or token.pos_ in ['CCONJ', 'SCONJ']:
                has_clause_separator = True
                break

        # If human is after a clause separator and before the verb, it's likely the new subject
        if has_clause_separator and human_pos < verb_pos:
            relevance += 15.0
            print(f"      +15 for being new subject after clause separator")

        # Penalty for very long distances (indicates unlikely association)
        if distance > 10:
            relevance -= 10.0
            print(f"      -10 penalty for excessive distance")

        final_score = max(0.0, relevance)
        print(f"      Final relevance score: {final_score}")
        return final_score

    def _get_human_priority(self, human):
        """Get priority score for human based on specificity"""
        human_text = human['text'].lower()

        # Higher priority for more specific human references
        if 'another' in human_text:
            return 3
        elif 'third' in human_text:
            return 3
        elif 'second' in human_text:
            return 2
        elif 'man' in human_text or 'person' in human_text:
            return 1
        else:
            return 0

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

    def _extract_with_basic_patterns(self, text, entities, coordinates_info, dataset_type='hico'):
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
                        triplets.extend(self._process_basic_action(human_text, action1, object1_text, humans, objects, coordinates_info, dataset_type))
                        # Process second action
                        triplets.extend(self._process_basic_action(human_text, action2, object2_text, humans, objects, coordinates_info, dataset_type))

                    elif len(groups) == 4:  # Pattern 2: "standing..., holding..."
                        human_text, action1, action2, object2_text = groups

                        # Process the holding action (more likely to be relevant)
                        triplets.extend(self._process_basic_action(human_text, action2, object2_text, humans, objects, coordinates_info, dataset_type))

                    elif len(groups) == 3:  # Pattern 3: simple subject-action-object
                        human_text, action, object_text = groups
                        triplets.extend(self._process_basic_action(human_text, action, object_text, humans, objects, coordinates_info, dataset_type))

                    elif len(groups) == 2:  # Pattern 4: action-object only
                        action, object_text = groups
                        # Try to match with any available human
                        for human in humans:
                            triplets.extend(self._process_basic_action(human['text'], action, object_text, humans, objects, coordinates_info, dataset_type))
                            break  # Just use first human for simplicity

        return triplets

    def _process_basic_action(self, human_text, action, object_text, humans, objects, coordinates_info, dataset_type='hico'):
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

            # Normalize action based on dataset requirements
            original_action = action_clean
            if dataset_type == 'hico':
                normalized_action = self._normalize_action(original_action)
            elif dataset_type == 'swig':
                normalized_action = self._normalize_action_for_swig(original_action)
            else:
                normalized_action = original_action

            triplet = {
                'human': human_entity,
                'human_bbox': human_coords,
                'action': normalized_action,
                'original_action': original_action,
                'object': object_entity,
                'object_bbox': object_coords,
                'confidence': 0.7,
                'extraction_method': 'basic_pattern_enhanced_no_adj'
            }
            single_triplets.append(triplet)
            action_display = f"{normalized_action}" if original_action == normalized_action else f"{normalized_action} [{original_action} → {normalized_action}]"
            print(f"    ✅ BASIC SUCCESS: {human_entity['text']} -> {action_display} -> {object_entity['text']}")
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