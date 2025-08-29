"""
filter_to_final_schema.py

Keep only the fields present (and NOT commented) in `final_dataset_schema` below.
- Works with .json and .jsonl (newline-delimited JSON), chosen by input extension.
- Preserves nested objects/arrays but prunes their keys according to the nested schema.
- No command-line arguments; edit INPUT_FILE / OUTPUT_FILE and run.

"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json

# ---- configure paths here ----
INPUT_FILE = "dataset/filtered_dataset.jsonl"   # or .json
OUTPUT_FILE = "dataset/final_dataset.jsonl" # or .json
# --------------------------------

# ---- your schema (only uncommented fields are kept) ----
final_dataset_schema: Dict[str, Any] = {
    "id": None,
    # "type": None,
    "applicationCategory": None,
    # "author": {
    #     "id": None,
    #     "type": None,
    #     "affiliation": None,
    #     "name": None,
    #     "legalName": None,
    #     "http://w3id.org/nfdi4ing/metadata4ing#orcidId": None,
    # },
    # "citation": None,
    # "codeRepository": None,
    # "dateCreated": None,
    # "datePublished": None,
    "description": None,
    "featureList": None,
    # "identifier": None,
    # "image": {
    #     "id": None,
    #     "type": None,
    #     "contentUrl": None,
    #     "keywords": None,
    # },
    "isAccessibleForFree": None,
    "isBasedOn": None,
    "keywords": None,
    # "license": None,
    # "memoryRequirements": None,
    "name": None,
    # "operatingSystem": None,
    # "processorRequirements": None,
    "programmingLanguage": None,
    "softwareRequirements": None,
    "supportingData": {
        # "id": None,
        # "type": None,
        "contentUrl": None,
        "description": None,
        "name": None,
        "datasetFormat": None,
        "hasDimensionality": None,
        # "measurementTechnique": None,
        "bodySite": None,
        "imagingModality": None,
        # "resolution": None,
        # "variableMeasured": None,
    },
    "url": None,
    # "fairLevel": None,
    # "graph": None,
    "imagingModality": None,
    "isPluginModuleOf": None,
    "relatedToOrganization": None,
    "requiresGPU": None,
    "runnableExample": {
        # "id": None,
        # "type": None,
        "description": None,
        "name": None,
        "url": None,
        "hostType": None,
    },
    # "hasDocumentation": None,
    # "hasExecutableInstructions": None,
    "hasExecutableNotebook": {
        # "id": None,
        # "type": None,
        "description": None,
        "name": None,
        "url": None,
    },
    # "hasAcknowledgements": None,
    # "hasFunding": {
    #     "id": None,
    #     "type": None,
    #     "identifier": None,
    #     "fundingGrant": None,
    #     "fundingSource": {
    #         "id": None,
    #         "type": None,
    #         "legalName": None,
    #     },
    # },
    # "hasParameter": {
    #     "id": None,
    #     "type": None,
    #     "description": None,
    #     "name": None,
    #     "valueRequired": None,
    #     "hasDimensionality": None,
    #     "hasFormat": None,
    #     "encodingFormat": None,
    # },
    # "readme": None,
    # "conditionsOfAccess": None,
    # "status": None,
}


def filter_by_schema(value: Any, schema: Any) -> Any:
    """
    Return `value` pruned to only include keys present in `schema`.
    Schema rules:
      - If schema is None: keep the value as-is (scalar, list, dict are all allowed).
      - If schema is a dict: keep only those keys; recurse into dicts/lists as needed.
    """
    # Leaf in schema: keep anything at this node
    if schema is None:
        return value

    # Nested schema: expect dict-like structure
    if isinstance(schema, dict):
        if isinstance(value, dict):
            out = {}
            for k, sub_schema in schema.items():
                if k in value:
                    out[k] = filter_by_schema(value[k], sub_schema)
            return out
        elif isinstance(value, list):
            # Apply nested schema to each item if it's a dict; otherwise keep the item
            filtered_list = []
            for item in value:
                if isinstance(item, dict):
                    filtered_list.append(filter_by_schema(item, schema))
                else:
                    filtered_list.append(item)
            return filtered_list
        else:
            # Value is scalar but schema expected nested; keep as-is (key is allowed)
            return value

    # Unknown schema type: fall back to original value
    return value


def process_json(inp: Path, out: Path, schema: Dict[str, Any]) -> None:
    data = json.loads(inp.read_text(encoding="utf-8"))
    filtered = filter_by_schema(data, schema)
    out.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")


def process_jsonl(inp: Path, out: Path, schema: Dict[str, Any]) -> None:
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                fout.write(line)
                continue
            obj = json.loads(line)
            obj = filter_by_schema(obj, schema)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    inp = Path(INPUT_FILE)
    out = Path(OUTPUT_FILE)
    if inp.suffix.lower() == ".jsonl":
        process_jsonl(inp, out, final_dataset_schema)
    else:
        process_json(inp, out, final_dataset_schema)


if __name__ == "__main__":
    main()
