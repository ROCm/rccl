#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Convert a plain text file into structured XML.

Supported line formats:
  1) [Section Name]      -> starts a new section
  2) key=value           -> stored as <entry key="key">value</entry>
  3) key: value          -> stored as <entry key="key">value</entry>
  4) Any other text line -> stored as <line>...</line>

Example:
  python3 tools/scripts/text_to_xml.py input.txt output.xml
"""

from __future__ import annotations

import argparse
import pathlib
import re
import xml.etree.ElementTree as ET


SECTION_PATTERN = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
KEY_VALUE_PATTERN = re.compile(r"^(?P<key>[^:=\s][^:=]*?)\s*[:=]\s*(?P<value>.*)$")


def parse_text_to_xml_tree(
    input_path: pathlib.Path,
    root_tag: str = "document",
    section_tag: str = "section",
    entry_tag: str = "entry",
    line_tag: str = "line",
    skip_empty_lines: bool = True,
) -> ET.Element:
    """Parse a text file and return an XML root element."""
    root = ET.Element(root_tag)
    root.set("source", str(input_path))
    current_section = ET.SubElement(root, section_tag, {"name": "default"})

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped and skip_empty_lines:
                continue

            section_match = SECTION_PATTERN.match(stripped)
            if section_match:
                current_section = ET.SubElement(
                    root,
                    section_tag,
                    {"name": section_match.group("name"), "line": str(line_number)},
                )
                continue

            key_value_match = KEY_VALUE_PATTERN.match(stripped)
            if key_value_match:
                entry = ET.SubElement(
                    current_section,
                    entry_tag,
                    {
                        "key": key_value_match.group("key").strip(),
                        "line": str(line_number),
                    },
                )
                entry.text = key_value_match.group("value")
                continue

            line_element = ET.SubElement(current_section, line_tag, {"line": str(line_number)})
            line_element.text = line

    return root


def write_xml(root_element: ET.Element, output_path: pathlib.Path) -> None:
    """Write an XML element tree to disk with indentation."""
    tree = ET.ElementTree(root_element)
    ET.indent(tree, space="  ", level=0)  # Python 3.9+
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Parse a text file and store its elements in XML format."
    )
    parser.add_argument("input_file", type=pathlib.Path, help="Path to the input text file")
    parser.add_argument("output_file", type=pathlib.Path, help="Path for generated XML output")
    parser.add_argument(
        "--keep-empty-lines",
        action="store_true",
        help="Include empty lines in the generated XML",
    )
    parser.add_argument(
        "--root-tag",
        default="document",
        help="Root XML tag name (default: document)",
    )
    return parser


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input_file}")

    xml_root = parse_text_to_xml_tree(
        input_path=args.input_file,
        root_tag=args.root_tag,
        skip_empty_lines=not args.keep_empty_lines,
    )
    write_xml(xml_root, args.output_file)
    print(f"XML written to: {args.output_file}")


if __name__ == "__main__":
    main()
