"""Validation script for Groma grounding instruction format."""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple


def validate_instruction_format(instruction: Dict, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate that an instruction follows Groma grounding format.

    Args:
        instruction: Instruction dict to validate
        verbose: If True, print detailed validation info

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Check required keys
    required_keys = ['file_name', 'width', 'height', 'boxes', 'conversation']
    for key in required_keys:
        if key not in instruction:
            errors.append(f"Missing required key: '{key}'")

    if errors:
        return False, errors

    # Check boxes format
    boxes = instruction.get('boxes', [])
    if not isinstance(boxes, list):
        errors.append(f"'boxes' must be a list, got {type(boxes)}")
    else:
        for i, box in enumerate(boxes):
            if not isinstance(box, list):
                errors.append(f"Box {i} must be a list, got {type(box)}")
                continue

            if len(box) != 4:
                errors.append(f"Box {i} has {len(box)} values, expected 4 (x, y, w, h)")

            if not all(isinstance(x, (int, float)) for x in box):
                errors.append(f"Box {i} contains non-numeric values: {box}")

    # Check conversation structure
    conv = instruction.get('conversation', [])
    if not isinstance(conv, list):
        errors.append(f"'conversation' must be a list, got {type(conv)}")
    elif len(conv) % 2 != 0:
        errors.append(f"Conversation has {len(conv)} turns, expected even number (Q&A pairs)")
    elif len(conv) == 0:
        errors.append(f"Conversation is empty")

    # Check each conversation turn
    for i, turn in enumerate(conv):
        if not isinstance(turn, dict):
            errors.append(f"Turn {i} must be a dict, got {type(turn)}")
            continue

        if 'value' not in turn:
            errors.append(f"Turn {i} missing 'value' key")

        if 'box_inds' not in turn:
            errors.append(f"Turn {i} missing 'box_inds' key")

        if 'from' not in turn:
            errors.append(f"Turn {i} missing 'from' key")
        elif turn['from'] not in ['human', 'gpt']:
            errors.append(f"Turn {i} has invalid 'from' value: {turn['from']} (should be 'human' or 'gpt')")

        # Check that "from" alternates correctly
        expected_from = "human" if i % 2 == 0 else "gpt"
        if turn.get('from') != expected_from:
            errors.append(
                f"Turn {i} has from='{turn.get('from')}', expected '{expected_from}'"
            )

        # For assistant turns (odd indices), validate box_inds and region tokens
        if i % 2 == 1:  # Assistant turn
            box_inds = turn.get('box_inds', [])

            if not isinstance(box_inds, list):
                errors.append(f"Turn {i} 'box_inds' must be a list, got {type(box_inds)}")
                continue

            # Check box indices are valid
            if box_inds:
                max_idx = max(box_inds)
                if max_idx >= len(boxes):
                    errors.append(
                        f"Turn {i} references box index {max_idx}, but only {len(boxes)} boxes exist"
                    )

                # Check that all box_inds are used in response
                response = turn.get('value', '')

                # Count <ground_box> tokens in response
                ground_box_count = response.count('<ground_box>')

                # Check if number of <ground_box> tokens matches box_inds length
                if ground_box_count != len(box_inds):
                    errors.append(
                        f"Turn {i} has {len(box_inds)} box_inds but {ground_box_count} "
                        f"<ground_box> tokens in response"
                    )

                # Check for proper <p> and <roi> tag pairing
                if '<p>' in response:
                    if response.count('<p>') != response.count('</p>'):
                        errors.append(f"Turn {i} has mismatched <p> tags")

                if '<roi>' in response:
                    if response.count('<roi>') != response.count('</roi>'):
                        errors.append(f"Turn {i} has mismatched <roi> tags")

                # Check that response doesn't contain [grounding] or <sep>
                # (these are added by dataset class)
                if '[grounding]' in response:
                    errors.append(
                        f"Turn {i} contains [grounding] token (should be added by dataset class)"
                    )

                if '<sep>' in response:
                    errors.append(
                        f"Turn {i} contains <sep> token (should be added by dataset class)"
                    )

        # For user turns (even indices), box_inds should typically be empty
        else:  # User turn
            box_inds = turn.get('box_inds', [])
            if box_inds:
                # This is not an error, but worth noting
                if verbose:
                    print(f"  Note: Turn {i} (user) has non-empty box_inds: {box_inds}")

    return len(errors) == 0, errors


def validate_instruction_file(
    json_file: str,
    sample_size: int = 100,
    verbose: bool = False
) -> bool:
    """
    Validate an instruction JSON file.

    Args:
        json_file: Path to instruction JSON file
        sample_size: Number of samples to validate (0 for all)
        verbose: If True, print detailed validation info

    Returns:
        True if all validated samples are valid, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Validating Groma Instruction Format")
    print(f"{'='*80}\n")
    print(f"File: {json_file}")

    if not Path(json_file).exists():
        print(f"ERROR: File not found: {json_file}")
        return False

    with open(json_file, 'r') as f:
        instructions = json.load(f)

    print(f"Total instructions: {len(instructions)}")

    # Validate sample or all
    if sample_size > 0:
        sample = instructions[:sample_size]
        print(f"Validating first {len(sample)} samples...")
    else:
        sample = instructions
        print(f"Validating all {len(sample)} instructions...")

    valid_count = 0
    error_count = 0
    error_summary = {}

    for i, instruction in enumerate(sample):
        is_valid, errors = validate_instruction_format(instruction, verbose=verbose)

        if is_valid:
            valid_count += 1
        else:
            error_count += 1
            file_name = instruction.get('file_name', f'unknown_{i}')

            # Track error types
            for error in errors:
                error_type = error.split(':')[0] if ':' in error else error
                error_summary[error_type] = error_summary.get(error_type, 0) + 1

            if error_count <= 10:  # Show first 10 errors in detail
                print(f"\n❌ Instruction {i} ({file_name}):")
                for error in errors:
                    print(f"  - {error}")

            if error_count == 10:
                print(f"\n⚠️  More errors found, limiting detailed output...")

    print(f"\n{'='*80}")
    print(f"Validation Results")
    print(f"{'='*80}")
    print(f"Valid: {valid_count}/{len(sample)} ({valid_count/len(sample)*100:.1f}%)")
    print(f"Errors: {error_count}/{len(sample)} ({error_count/len(sample)*100:.1f}%)")

    if error_summary:
        print(f"\nError Summary:")
        for error_type, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count}")

    print(f"{'='*80}\n")

    return error_count == 0


def print_sample_instruction(json_file: str, index: int = 0):
    """
    Print a sample instruction for inspection.

    Args:
        json_file: Path to instruction JSON file
        index: Index of instruction to print
    """
    with open(json_file, 'r') as f:
        instructions = json.load(f)

    if index >= len(instructions):
        print(f"ERROR: Index {index} out of range (only {len(instructions)} instructions)")
        return

    instruction = instructions[index]

    print(f"\n{'='*80}")
    print(f"Sample Instruction #{index}")
    print(f"{'='*80}\n")

    print(f"File: {instruction['file_name']}")
    print(f"Size: {instruction['width']}x{instruction['height']}")
    print(f"Boxes: {len(instruction['boxes'])}")

    print(f"\nBoxes:")
    for i, box in enumerate(instruction['boxes']):
        print(f"  r{i}: {box}")

    print(f"\nConversation:")
    for i, turn in enumerate(instruction['conversation']):
        role = "USER" if i % 2 == 0 else "ASSISTANT"
        print(f"\n{role}:")
        print(f"  {turn['value']}")
        if turn['box_inds']:
            print(f"  [box_inds: {turn['box_inds']}]")

    print(f"\n{'='*80}\n")


def main():
    """Command-line interface for validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate Groma grounding instruction format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to instruction JSON file to validate'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of samples to validate (0 for all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed validation information'
    )
    parser.add_argument(
        '--print-sample',
        type=int,
        default=None,
        metavar='INDEX',
        help='Print a sample instruction at the given index'
    )

    args = parser.parse_args()

    if args.print_sample is not None:
        print_sample_instruction(args.json_file, args.print_sample)
        return 0

    success = validate_instruction_file(
        args.json_file,
        sample_size=args.sample_size,
        verbose=args.verbose
    )

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
