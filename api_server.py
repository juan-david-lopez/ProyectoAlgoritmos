"""
API REST Wrapper for main.py - CLI Version
Simple Flask API to trigger main.py operations via HTTP requests
"""

from flask import Flask, jsonify, request
import subprocess
import sys
import os
from pathlib import Path

app = Flask(__name__)

# Ensure main.py is in the path
MAIN_PY = Path(__file__).parent / "main.py"

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "bibliometric-cli",
        "version": "1.0.0"
    }), 200


@app.route('/api/run', methods=['POST'])
def run_pipeline():
    """
    Execute main.py with specified mode
    
    POST /api/run
    Body: {
        "mode": "scrape|deduplicate|preprocess|cluster|visualize|report|full",
        "sources": ["ieee", "scopus"] (optional)
    }
    """
    try:
        data = request.get_json()
        mode = data.get('mode', 'info')
        sources = data.get('sources', [])
        
        # Validate mode
        valid_modes = ['scrape', 'deduplicate', 'preprocess', 'cluster', 
                      'visualize', 'report', 'full', 'info']
        if mode not in valid_modes:
            return jsonify({
                "error": f"Invalid mode. Must be one of: {', '.join(valid_modes)}"
            }), 400
        
        # Build command
        cmd = [sys.executable, str(MAIN_PY), '--mode', mode]
        if sources:
            cmd.extend(['--sources', ','.join(sources)])
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        return jsonify({
            "status": "success" if result.returncode == 0 else "error",
            "mode": mode,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }), 200 if result.returncode == 0 else 500
        
    except subprocess.TimeoutExpired:
        return jsonify({
            "error": "Operation timed out (max 5 minutes)"
        }), 408
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get project information"""
    try:
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), '--mode', 'info'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        return jsonify({
            "status": "success",
            "output": result.stdout
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        "service": "Bibliometric Analysis CLI API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API documentation",
            "GET /health": "Health check",
            "GET /api/info": "Get project information",
            "POST /api/run": "Execute pipeline operation"
        },
        "usage": {
            "run_full_pipeline": {
                "method": "POST",
                "url": "/api/run",
                "body": {
                    "mode": "full"
                }
            },
            "run_scraping": {
                "method": "POST",
                "url": "/api/run",
                "body": {
                    "mode": "scrape",
                    "sources": ["ieee", "scopus"]
                }
            }
        }
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
