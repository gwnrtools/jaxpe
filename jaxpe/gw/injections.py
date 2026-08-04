"""Utilities for loading and generating GW injections."""

import xml.etree.ElementTree as ET


def parse_sim_inspiral_table(xml_path):
    """Parse a LIGOLW sim_inspiral table from an XML file into a list of dicts.

    Args:
        xml_path (str or Path): Path to the XML file.

    Returns:
        list of dict: A list where each element is a dictionary representing
            an injection's parameters.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Namespace handling might be needed, but usually we can search by element name or attribute
    # Find the sim_inspiral table
    table = None
    for t in root.iter("Table"):
        if "sim_inspiral" in t.get("Name", ""):
            table = t
            break

    if table is None:
        raise ValueError("No sim_inspiral table found in XML file.")

    # Get column names in order
    columns = []
    for col in table.iter("Column"):
        # Format usually "sim_inspiral:mass1"
        name = col.get("Name", "").split(":")[-1]
        columns.append(name)

    # Get the data stream
    stream = table.find("Stream")
    if stream is None:
        raise ValueError("No Stream found in sim_inspiral table.")

    delimiter = stream.get("Delimiter", ",")
    data_text = stream.text.strip()

    injections = []
    # Split into rows (tokens separated by delimiter and sometimes newlines depending on LIGOLW flavor)
    tokens = [t.strip() for t in data_text.split(delimiter) if t.strip() != ""]

    num_cols = len(columns)
    if len(tokens) % num_cols != 0:
        raise ValueError(
            f"Number of tokens ({len(tokens)}) is not a multiple of number of columns ({num_cols})."
        )

    for i in range(0, len(tokens), num_cols):
        row_vals = tokens[i : i + num_cols]
        row_dict = {}
        for col_name, val_str in zip(columns, row_vals):
            # Try to convert to float/int if possible
            val_str = val_str.strip('"')
            try:
                if "." in val_str or "e" in val_str.lower():
                    val = float(val_str)
                else:
                    val = int(val_str)
            except ValueError:
                val = val_str
            row_dict[col_name] = val
        injections.append(row_dict)

    return injections
