import re

def to_camel_case_word(word: str) -> str:
    """Converts a single word to CamelCase (first letter capitalized, rest lowercase)."""
    if not word:
        return ""
    return word[0].upper() + word[1:].lower()

def general_standardize_field_name(field_name: str) -> str:
    """
    General standardization: splits by common delimiters, camel-cases parts, and joins with periods.
    Example: "institution. jurisdictionOfIncorporation country" -> "institution.JurisdictionOfIncorporation.Country"
    """
    if not isinstance(field_name, str):
        return str(field_name)

    cleaned_text = re.sub(r'[^a-zA-Z0-9.\s_-]+', ' ', cleaned_text)
    parts = re.split(r'[.\s_-]+', cleaned_text)
    
    standardized_parts = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            if i == 0:
                standardized_parts.append(part.lower())
            else:
                standardized_parts.append(to_camel_case_word(part))
        
    if not standardized_parts:
        return "untitled.field"

    return ".".join(standardized_parts)

# Example Usage (for testing the functions directly)
if __name__ == "__main__":
    print("--- General Standardization ---")
    print(f"Original: 'Institution.JurisdictionOfIncorporation Country'")
    print(f"Standardized: '{general_standardize_field_name('Institution.JurisdictionOfIncorporation Country')}'\n")

    print(f"Original: 'Customer ID'")
    print(f"Standardized: '{general_standardize_field_name('Customer ID')}'\n")