"""
Response utilities for the API server.
This module provides utility functions for generating consistent API responses.
"""

from typing import Dict, Any, Optional
from flask import jsonify

def success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """
    Generate a success response.
    
    Args:
        data: Response data
        message: Success message
        
    Returns:
        Dictionary with success response
    """
    response = {
        "success": True,
        "message": message,
    }
    
    if data is not None:
        response["data"] = data
    
    return jsonify(response)

def error_response(message: str, status_code: int = 400) -> tuple:
    """
    Generate an error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Tuple with error response and status code
    """
    response = {
        "success": False,
        "message": message,
    }
    
    return jsonify(response), status_code
