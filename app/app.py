"""
Main entry point for the AI Model Training application.
This will start the appropriate service based on the environment.
"""

import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the application.
    Parses command line arguments and starts the appropriate service.
    """
    parser = argparse.ArgumentParser(description='AI Model Training Platform')
    parser.add_argument(
        '--service', 
        type=str, 
        choices=['api', 'training', 'monitor', 'all'],
        default=os.environ.get('SERVICE_TYPE', 'api'),
        help='Service to start (api, training, monitor, or all)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=int(os.environ.get('PORT', 8000)),
        help='Port to run the service on'
    )
    parser.add_argument(
        '--host', 
        type=str, 
        default=os.environ.get('HOST', '0.0.0.0'),
        help='Host to run the service on'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        default=os.environ.get('DEBUG', 'false').lower() == 'true',
        help='Run in debug mode'
    )
    
    args = parser.parse_args()
    
    # Start the appropriate service
    if args.service == 'api':
        logger.info("Starting API service on %s:%d", args.host, args.port)
        from app.api.main import app
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.service == 'training':
        logger.info("Starting training worker")
        from app.services.training_worker import TrainingWorker
        worker = TrainingWorker()
        worker.start()
    
    elif args.service == 'monitor':
        logger.info("Starting monitoring service on %s:%d", args.host, args.port)
        from app.services.monitor import app
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.service == 'all':
        logger.info("Starting all services")
        # Use multiprocessing to start all services
        import multiprocessing
        
        # API service
        api_process = multiprocessing.Process(
            target=lambda: os.system(f"python -m app.api.main --port 8000")
        )
        
        # Training worker
        training_process = multiprocessing.Process(
            target=lambda: os.system(f"python -m app.services.training_worker")
        )
        
        # Monitoring service
        monitor_process = multiprocessing.Process(
            target=lambda: os.system(f"python -m app.services.monitor --port 8081")
        )
        
        # Start processes
        api_process.start()
        training_process.start()
        monitor_process.start()
        
        # Join processes
        api_process.join()
        training_process.join()
        monitor_process.join()
    
    else:
        logger.error("Unknown service: %s", args.service)
        sys.exit(1)

if __name__ == "__main__":
    main()
