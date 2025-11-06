"""
Enhanced analysis generator that creates complete descriptions
from medicine name and predictions.
Uses template-based approach (can be enhanced with LLM later).
"""

import re

def generate_analysis(medicine_name, predictions):
    """
    Generate complete analysis/description from medicine name and predictions.
    
    Args:
        medicine_name: Generic name of the medicine
        predictions: Dictionary with all predictions from the model
    
    Returns:
        Complete analysis text (string)
    """
    # Extract predictions
    dosage_form = predictions.get('dosage_form', [])
    manufacturer = predictions.get('manufacturer', [])
    disposal_category = predictions.get('disposal_category', {})
    method_of_disposal = predictions.get('method_of_disposal', [])
    handling_method = predictions.get('handling_method', '')
    disposal_remarks = predictions.get('disposal_remarks', '')
    
    # Build analysis
    analysis_parts = []
    
    # Introduction
    analysis_parts.append(f"# Medicine Analysis: {medicine_name.title()}\n")
    analysis_parts.append(f"Based on the medicine **{medicine_name.title()}**, here is a comprehensive analysis:\n")
    
    # Dosage Form
    if dosage_form:
        top_dosage = dosage_form[0]
        analysis_parts.append(f"## Dosage Form\n")
        analysis_parts.append(f"The most likely dosage form is **{top_dosage['value'].title()}** ")
        analysis_parts.append(f"(confidence: {top_dosage['confidence']:.1%}).")
        
        if len(dosage_form) > 1:
            alternatives = [df['value'].title() for df in dosage_form[1:]]
            analysis_parts.append(f" Alternative forms include: {', '.join(alternatives)}.")
        analysis_parts.append("\n")
    
    # Manufacturer
    if manufacturer:
        top_mfg = manufacturer[0]
        analysis_parts.append(f"## Manufacturer\n")
        analysis_parts.append(f"The most likely manufacturer is **{top_mfg['value'].title()}** ")
        analysis_parts.append(f"(confidence: {top_mfg['confidence']:.1%}).")
        
        if len(manufacturer) > 1:
            alternatives = [mfg['value'].title() for mfg in manufacturer[1:]]
            analysis_parts.append(f" Alternative manufacturers include: {', '.join(alternatives)}.")
        analysis_parts.append("\n")
    
    # Disposal Category
    if disposal_category:
        cat = disposal_category.get('value', '')
        confidence = disposal_category.get('confidence', 0)
        analysis_parts.append(f"## Disposal Category\n")
        analysis_parts.append(f"This medicine falls under **{cat.title()}** ")
        analysis_parts.append(f"(confidence: {confidence:.1%}).\n")
    
    # Method of Disposal
    if method_of_disposal:
        analysis_parts.append(f"## Recommended Disposal Methods\n")
        methods = [method['value'].title() for method in method_of_disposal]
        analysis_parts.append(f"The following disposal methods are recommended:\n")
        for i, method in enumerate(methods, 1):
            confidence = next((m['confidence'] for m in method_of_disposal if m['value'] == method.replace(' ', ' ').lower()), 0)
            analysis_parts.append(f"  {i}. {method} (confidence: {confidence:.1%})\n")
        analysis_parts.append("\n")
    
    # Handling Method
    if handling_method:
        analysis_parts.append(f"## Handling Instructions\n")
        analysis_parts.append(f"{handling_method}\n\n")
    
    # Disposal Remarks
    if disposal_remarks:
        analysis_parts.append(f"## Important Disposal Remarks\n")
        analysis_parts.append(f"{disposal_remarks}\n\n")
    
    # Summary
    analysis_parts.append(f"## Summary\n")
    analysis_parts.append(f"For the medicine **{medicine_name.title()}**, ")
    
    if dosage_form:
        analysis_parts.append(f"it is typically available as {dosage_form[0]['value'].title()}, ")
    if manufacturer:
        analysis_parts.append(f"manufactured by {manufacturer[0]['value'].title()}. ")
    if disposal_category:
        analysis_parts.append(f"It should be disposed of as {disposal_category.get('value', '').title()}. ")
    
    if method_of_disposal:
        primary_method = method_of_disposal[0]['value'].title()
        analysis_parts.append(f"The primary disposal method is {primary_method}.")
    
    analysis_parts.append("\n")
    
    return ''.join(analysis_parts)

def generate_short_summary(medicine_name, predictions):
    """
    Generate a short summary (one paragraph).
    
    Args:
        medicine_name: Generic name of the medicine
        predictions: Dictionary with all predictions
    
    Returns:
        Short summary text (string)
    """
    dosage_form = predictions.get('dosage_form', [{}])[0].get('value', 'unknown form') if predictions.get('dosage_form') else 'unknown form'
    manufacturer = predictions.get('manufacturer', [{}])[0].get('value', 'unknown manufacturer') if predictions.get('manufacturer') else 'unknown manufacturer'
    disposal_category = predictions.get('disposal_category', {}).get('value', 'unknown category')
    
    summary = f"{medicine_name.title()} is typically available as {dosage_form.title()} from manufacturers like {manufacturer.title()}. "
    summary += f"It should be disposed of as {disposal_category.title()}. "
    
    if predictions.get('method_of_disposal'):
        methods = [m['value'].title() for m in predictions['method_of_disposal'][:2]]
        summary += f"Recommended disposal methods include: {', '.join(methods)}."
    
    return summary

def generate_json_analysis(medicine_name, predictions):
    """
    Generate structured JSON analysis.
    
    Args:
        medicine_name: Generic name of the medicine
        predictions: Dictionary with all predictions
    
    Returns:
        Dictionary with structured analysis
    """
    import json
    
    analysis = {
        'medicine_name': medicine_name.title(),
        'dosage_form': {
            'primary': predictions.get('dosage_form', [{}])[0].get('value') if predictions.get('dosage_form') else None,
            'alternatives': [df['value'] for df in predictions.get('dosage_form', [])[1:3]],
            'confidence': predictions.get('dosage_form', [{}])[0].get('confidence') if predictions.get('dosage_form') else None
        },
        'manufacturer': {
            'primary': predictions.get('manufacturer', [{}])[0].get('value') if predictions.get('manufacturer') else None,
            'alternatives': [mfg['value'] for mfg in predictions.get('manufacturer', [])[1:3]],
            'confidence': predictions.get('manufacturer', [{}])[0].get('confidence') if predictions.get('manufacturer') else None
        },
        'disposal_category': predictions.get('disposal_category', {}),
        'disposal_methods': predictions.get('method_of_disposal', []),
        'handling_method': predictions.get('handling_method', ''),
        'disposal_remarks': predictions.get('disposal_remarks', ''),
        'similar_medicine': predictions.get('similar_generic_name', ''),
        'similarity_score': predictions.get('similarity_distance', 0)
    }
    
    return analysis



