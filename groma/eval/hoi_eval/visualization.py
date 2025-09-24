"""HOI Visualization utilities for creating visual representations of HOI triplets."""

from PIL import Image, ImageDraw, ImageFont


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
            # Now triplet['action'] is the normalized action
            normalized_action = triplet['action']
            original_action = triplet.get('original_action', normalized_action)
            object_text = triplet['object']['text']
            object_region = triplet['object']['region_id']

            # Show action normalization if different
            action_note = ""
            if original_action != normalized_action:
                action_note = f" [{original_action} → {normalized_action}]"

            # Show original text if different
            original_note = ""
            if 'original_text' in triplet['object'] and triplet['object']['original_text'] != object_text:
                original_note = f" (original: '{triplet['object']['original_text']}')"
            print(f"  Triplet {i+1}: '{human_text}' (R{human_region}) -> {normalized_action}{action_note} -> '{object_text}' (R{object_region}){original_note}")

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

            # Now triplet['action'] is the normalized action
            normalized_action = action
            original_action = triplet.get('original_action', normalized_action)
            action_note = f" [{original_action} → {normalized_action}]" if original_action != normalized_action else ""
            print(f"DEBUG: Processing triplet: '{human_text}' (R{human_region}) -> {normalized_action}{action_note} -> '{object_text}' (R{object_region})")

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

            # Now triplet['action'] is the normalized action
            normalized_action = action
            original_action = triplet.get('original_action', normalized_action)
            action_note = f" [{original_action} → {normalized_action}]" if original_action != normalized_action else ""
            print(f"DEBUG: Drawing triplet {i+1}: '{human_text}' (R{human_region}) -> {normalized_action}{action_note} -> '{object_text}' (R{object_region})")

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

                # Get normalized action for display
                normalized_action = action
                original_action = triplet.get('original_action', normalized_action)

                # Show normalized action with original if different
                if original_action != normalized_action:
                    action_text = f"{normalized_action.upper()}\n({original_action})"
                else:
                    action_text = f"{normalized_action.upper()}"

                action_bbox = draw.textbbox((mid_x, mid_y), action_text, font=font)
                draw.rectangle(action_bbox, fill=self.colors['action'])
                draw.text((mid_x, mid_y), action_text, fill="white", font=font)

                print(f"DEBUG: Drew connection from {human_center} to {object_center} with action '{normalized_action}'{action_note}'")
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
        print(f"  🎯 Creating ground truth visualization with {len(gt_hois)} HOI annotations")

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
        print(f"  🤖 Creating prediction visualization with {len(prediction_triplets)} triplets")

        pred_img = self.image.copy()
        draw = ImageDraw.Draw(pred_img)

        # Load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Prediction colors - different colors for mapped vs unmapped actions
        mapped_colors = {
            'person': '#00AA00',        # Green for mapped actions (evaluable)
            'object': '#00CC00',        # Light green for mapped objects
            'connection': '#008800',    # Dark green for mapped connections
        }

        unmapped_colors = {
            'person': '#FF4444',        # Red for unmapped actions (visualization-only)
            'object': '#FF6666',        # Light red for unmapped objects
            'connection': '#CC2222',    # Dark red for unmapped connections
        }

        # Create mapping of region_id to coordinates
        coords_by_region = {}
        for info in coordinates_info:
            coords_by_region[info['region_id']] = info

        # Show all predictions with different colors based on mapping status
        for i, triplet in enumerate(prediction_triplets):
            human_region = triplet['human']['region_id']
            object_region = triplet['object']['region_id']
            action = triplet['action']  # Show raw predicted action
            human_text = triplet['human']['text']
            object_text = triplet['object']['text']  # Show raw predicted object
            confidence = triplet.get('confidence', 0.0)

            print(f"    Visualizing triplet {i+1}: '{human_text}' (R{human_region}) -> '{action}' -> '{object_text}' (R{object_region})")

            # Determine color scheme based on mapping status
            mapping_status = triplet.get('mapping_status', 'unknown')
            if mapping_status == 'mapped':
                colors = mapped_colors
                status_indicator = "✅"
            else:
                colors = unmapped_colors
                status_indicator = "⚠️"

            person_color = colors['person']
            object_color = colors['object']
            connection_color = colors['connection']

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

                # Add labels with mapping status indicators
                standardized_human = self._standardize_human_label_viz(human_text)
                person_label = f"PRED-P{i+1}: {standardized_human}"
                object_label = f"PRED-O{i+1}: {object_text}"

                # Include mapping status in action label
                if mapping_status == 'mapped':
                    action_label = f"{status_indicator} {action} ({confidence:.2f})"
                else:
                    warning_msg = triplet.get('warning', 'unmapped')
                    action_label = f"{status_indicator} {action} ({confidence:.2f}) - UNMAPPED"

                # Draw labels
                self._draw_label(draw, human_bbox, person_label, person_color, font_small)
                self._draw_label(draw, object_bbox, object_label, object_color, font_small)

                # Draw action label
                if human_center != object_center:
                    mid_x = (human_center[0] + object_center[0]) // 2
                    mid_y = (human_center[1] + object_center[1]) // 2
                    self._draw_action_label(draw, (mid_x, mid_y), action_label, connection_color, font)

        # Add legend to explain color coding
        self._draw_legend(draw, pred_img.size, font_small, mapped_colors, unmapped_colors)

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

    def _draw_legend(self, draw, img_size, font, mapped_colors, unmapped_colors):
        """Draw legend explaining color coding"""
        img_width, img_height = img_size

        # Position legend in top-right corner
        legend_x = img_width - 250
        legend_y = 10

        # Background for legend
        legend_bg = (legend_x - 10, legend_y - 5, img_width - 10, legend_y + 80)
        draw.rectangle(legend_bg, fill='white', outline='black', width=1)

        # Legend title
        draw.text((legend_x, legend_y), "Legend:", fill='black', font=font)

        # Mapped actions
        draw.rectangle([(legend_x, legend_y + 20), (legend_x + 15, legend_y + 30)],
                      fill=mapped_colors['connection'], outline='black')
        draw.text((legend_x + 20, legend_y + 18), "✅ Evaluable (in dataset)", fill='black', font=font)

        # Unmapped actions
        draw.rectangle([(legend_x, legend_y + 40), (legend_x + 15, legend_y + 50)],
                      fill=unmapped_colors['connection'], outline='black')
        draw.text((legend_x + 20, legend_y + 38), "⚠️ Unmapped (visualization-only)", fill='black', font=font)

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