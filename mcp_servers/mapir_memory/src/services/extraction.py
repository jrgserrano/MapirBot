"""Simplified extraction logic for Graphiti memory."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

def process_simple_extraction(text: str, original_prompt: str) -> dict[str, Any]:
    """
    Parse a simplified JSON list of entities and map them to Graphiti's expected IDs.
    
    Expected input format:
    [
      {"name": "Alice", "type": "Person"},
      {"name": "Coffee", "type": "Object"}
    ]
    """
    try:
        # Extract the Type-to-ID mapping from the original prompt
        # The prompt usually contains something like:
        # <ENTITY TYPES>
        # [{"name": "Person", "entity_type_id": 1}, ...]
        # </ENTITY TYPES>
        
        type_map = {}
        type_section = re.search(r"<ENTITY TYPES>(.*?)</ENTITY TYPES>", original_prompt, re.DOTALL)
        if type_section:
            try:
                # Use ast.literal_eval as the prompt might contain single quotes or non-standard JSON
                import ast
                types_raw = ast.literal_eval(type_section.group(1).strip())
                logger.info(f"ROBUST: Parsed {len(types_raw)} types from prompt")
                for t in types_raw:
                    # Support both 'name', 'label', 'entity_type_name'
                    name = t.get('name') or t.get('label') or t.get('entity_type_name')
                    # Support both 'entity_type_id' and 'id'
                    etype_id = t.get('entity_type_id')
                    if etype_id is None:
                        etype_id = t.get('id')
                    
                    if name and etype_id is not None:
                        type_map[str(name).lower()] = etype_id
                    else:
                        logger.warning(f"ROBUST: Skipping malformed entity type: {t}")
            except Exception as e:
                logger.error(f"Failed to parse type map from prompt: {e}. Raw section: {type_section.group(1)[:100]}...")

        # Clean JSON from markdown if necessary
        clean_text = text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"```[a-z]*\n?(.*?)\n?```", r"\1", clean_text, flags=re.DOTALL).strip()
        
        try:
            raw_entities = json.loads(clean_text)
        except json.JSONDecodeError:
            # Try to find something that looks like a list
            list_match = re.search(r"(\[.*\])", clean_text, re.DOTALL)
            if list_match:
                raw_entities = json.loads(list_match.group(1))
            else:
                raise

        processed = []
        if isinstance(raw_entities, list):
            for item in raw_entities:
                name = item.get("name")
                etype = item.get("type") or item.get("entity_type")
                
                if not name or not etype:
                    continue
                
                # Map type to ID
                etype_lower = str(etype).lower()
                tid = type_map.get(etype_lower)
                
                # Fuzzy match if exact match fails
                if tid is None:
                    for k, v in type_map.items():
                        if k in etype_lower or etype_lower in k:
                            tid = v
                            break
                
                if tid is not None:
                    processed.append({"name": name, "entity_type_id": tid})
                    
        if not processed:
            logger.warning(f"No entities extracted from: {text[:200]}...")
        else:
            logger.info(f"Extracted {len(processed)} entities")

        return {"extracted_entities": processed}
        
    except Exception as e:
        logger.error(f"Failed to parse simplified extraction: {e}")
        return {"extracted_entities": []}
