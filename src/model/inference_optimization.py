"""
/**
 * Copyright (c) [2025] Backdoor Software Inc.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of Backdoor Software Inc.
 * You may not disclose, reproduce, or distribute this software without the express written
 * permission of Backdoor Software Inc.
 *
 * Created by: Backdoor Software Inc.
 * Purpose: Activity Maintenance
 */
"""

"""
Inference optimization techniques for large language models.

This module implements:
- Continuous batching for efficient token generation
- Speculative decoding for faster inference
- KV cache management for memory efficiency
- Optimized beam search with disk offloading
- Streaming token generation
"""

import os
import time
import heapq
import threading
import queue
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Iterator
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """
    Represents a generation request for continuous batching.
    """
    request_id: str
    prompt: Union[str, List[int]]  # Text prompt or token IDs
    max_new_tokens: int = 20
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    do_sample: bool = False
    seed: Optional[int] = None
    stop_strings: List[str] = field(default_factory=list)
    
    # Tracking fields
    is_completed: bool = False
    generated_tokens: List[int] = field(default_factory=list)
    prompt_tokens: List[int] = field(default_factory=list)
    
    # Timing fields
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def elapsed_time(self) -> float:
        """Get elapsed processing time."""
        if self.end_time is not None and self.start_time is not None:
            return self.end_time - self.start_time
        if self.start_time is not None:
            return time.time() - self.start_time
        return 0.0
    
    def time_in_queue(self) -> float:
        """Get time spent in queue."""
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return time.time() - self.arrival_time
    
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.elapsed_time() > 0:
            return len(self.generated_tokens) / self.elapsed_time()
        return 0.0


class ContinuousBatchingServer:
    """
    Server for continuous batching of generation requests.
    
    This implements a server that accepts generation requests and processes
    them in a continuous batching fashion, optimizing GPU utilization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_batch_size: int = 32,
        max_waiting_tokens: int = 8192,
        max_active_requests: int = 256,
        batch_timeout: float = 0.1,  # seconds
        device: str = "cuda",
        precision: str = "float16",
        prefill_chunk_size: int = 2048,
        use_beam_search: bool = False,
        beam_width: int = 4,
        beam_manipulation_callback: Optional[Callable] = None,
        streaming: bool = False,
        yield_frequency: int = 1,  # tokens
        max_kv_cache_size: Optional[int] = None  # In tokens
    ):
        """
        Initialize continuous batching server.
        
        Args:
            model: Language model for generation
            tokenizer: Tokenizer for the model
            max_batch_size: Maximum batch size to process at once
            max_waiting_tokens: Maximum number of tokens to process in a batch
            max_active_requests: Maximum number of active requests
            batch_timeout: Maximum time to wait for batching
            device: Device to run the model on
            precision: Precision for inference
            prefill_chunk_size: Maximum chunk size for initial prompt processing
            use_beam_search: Whether to use beam search
            beam_width: Number of beams for beam search
            beam_manipulation_callback: Callback for beam manipulation
            streaming: Whether to stream tokens as they're generated
            yield_frequency: How often to yield tokens when streaming
            max_kv_cache_size: Maximum KV cache size in tokens
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_waiting_tokens = max_waiting_tokens
        self.max_active_requests = max_active_requests
        self.batch_timeout = batch_timeout
        self.device = device
        self.precision = precision
        self.prefill_chunk_size = prefill_chunk_size
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        self.beam_manipulation_callback = beam_manipulation_callback
        self.streaming = streaming
        self.yield_frequency = yield_frequency
        self.max_kv_cache_size = max_kv_cache_size
        
        # Initialize request queues
        self.request_queue = queue.Queue()
        self.active_requests: Dict[str, GenerationRequest] = {}
        self.completed_requests: Dict[str, GenerationRequest] = {}
        
        # Initialize KV cache manager
        self.kv_cache_manager = KVCacheManager(
            model=model, 
            max_cache_size=max_kv_cache_size
        )
        
        # KV caches for active requests
        self.kv_caches: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        
        # Model is assumed to be in evaluation mode
        self.model.eval()
        
        # Generate lock
        self.generate_lock = threading.Lock()
        
        # Threading settings
        self.is_running = False
        self.generation_thread = None
        
        # Stats
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.batch_stats = []
        
        logger.info(f"Initialized continuous batching server with max batch size {max_batch_size}")
    
    def add_request(self, request: GenerationRequest) -> str:
        """
        Add a generation request to the queue.
        
        Args:
            request: Generation request
            
        Returns:
            Request ID
        """
        # Ensure prompt tokens are set
        if not request.prompt_tokens and isinstance(request.prompt, str):
            request.prompt_tokens = self.tokenizer.encode(request.prompt)
        elif not request.prompt_tokens and isinstance(request.prompt, list):
            request.prompt_tokens = request.prompt
        
        # Add to queue
        self.request_queue.put(request)
        self.total_requests += 1
        
        return request.request_id
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a request.
        
        Args:
            request_id: Request ID
            
        Returns:
            Request status dictionary or None if not found
        """
        # Check active requests
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                "status": "in_progress",
                "generated_tokens": len(request.generated_tokens),
                "tokens_per_second": request.tokens_per_second(),
                "time_elapsed": request.elapsed_time(),
                "time_in_queue": request.time_in_queue()
            }
        
        # Check completed requests
        if request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            return {
                "status": "completed",
                "generated_tokens": len(request.generated_tokens),
                "tokens_per_second": request.tokens_per_second(),
                "time_elapsed": request.elapsed_time(),
                "time_in_queue": request.time_in_queue(),
                "output": self.tokenizer.decode(request.generated_tokens)
            }
        
        # Check if in queue
        queue_items = list(self.request_queue.queue)
        for item in queue_items:
            if item.request_id == request_id:
                return {
                    "status": "queued",
                    "time_in_queue": item.time_in_queue()
                }
        
        return None
    
    def get_results(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get results of a completed request.
        
        Args:
            request_id: Request ID
            
        Returns:
            Results dictionary or None if not found or not completed
        """
        if request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            
            # Decode generated tokens
            output = self.tokenizer.decode(request.generated_tokens)
            
            return {
                "request_id": request_id,
                "output": output,
                "tokens": request.generated_tokens,
                "prompt_tokens": len(request.prompt_tokens),
                "generated_tokens": len(request.generated_tokens),
                "total_tokens": len(request.prompt_tokens) + len(request.generated_tokens),
                "time_in_queue": request.time_in_queue(),
                "generation_time": request.elapsed_time(),
                "tokens_per_second": request.tokens_per_second()
            }
        
        return None
    
    def clear_completed_request(self, request_id: str) -> bool:
        """
        Clear a completed request from memory.
        
        Args:
            request_id: Request ID
            
        Returns:
            Whether the request was cleared
        """
        if request_id in self.completed_requests:
            del self.completed_requests[request_id]
            return True
        
        return False
    
    def stream_results(self, request_id: str) -> Iterator[Dict[str, Any]]:
        """
        Stream results as they're generated.
        
        Args:
            request_id: Request ID
            
        Yields:
            Partial results dictionaries
        """
        if not self.streaming:
            raise ValueError("Streaming is not enabled for this server")
        
        request = None
        last_token_idx = 0
        
        while True:
            # Get current request state
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                current_tokens = request.generated_tokens
                
                # Check if we have new tokens to yield
                if len(current_tokens) >= last_token_idx + self.yield_frequency:
                    # Get new tokens
                    new_tokens = current_tokens[last_token_idx:]
                    last_token_idx = len(current_tokens)
                    
                    # Decode new tokens
                    new_text = self.tokenizer.decode(new_tokens)
                    
                    # Yield new tokens
                    yield {
                        "request_id": request_id,
                        "status": "in_progress",
                        "new_tokens": new_tokens,
                        "new_text": new_text,
                        "tokens_generated": len(current_tokens),
                        "is_finished": False
                    }
                    
                    # Sleep briefly to avoid tight loop
                    time.sleep(0.01)
                
            elif request_id in self.completed_requests:
                request = self.completed_requests[request_id]
                current_tokens = request.generated_tokens
                
                # Check if we have new tokens to yield
                if len(current_tokens) > last_token_idx:
                    # Get new tokens
                    new_tokens = current_tokens[last_token_idx:]
                    
                    # Decode new tokens
                    new_text = self.tokenizer.decode(new_tokens)
                    
                    # Yield new tokens
                    yield {
                        "request_id": request_id,
                        "status": "completed",
                        "new_tokens": new_tokens,
                        "new_text": new_text,
                        "tokens_generated": len(current_tokens),
                        "is_finished": True,
                        "generation_time": request.elapsed_time(),
                        "tokens_per_second": request.tokens_per_second()
                    }
                
                # Request completed, stop streaming
                break
            
            # Sleep briefly to avoid tight loop
            time.sleep(0.1)
    
    def start(self) -> None:
        """Start the generation thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.generation_thread = threading.Thread(target=self._generation_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
        logger.info("Started continuous batching generation thread")
    
    def stop(self) -> None:
        """Stop the generation thread."""
        self.is_running = False
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5.0)
        
        logger.info("Stopped continuous batching generation thread")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        stats = {
            "total_requests": self.total_requests,
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "queued_requests": self.request_queue.qsize(),
            "total_tokens_generated": self.total_tokens_generated
        }
        
        if self.batch_stats:
            stats.update({
                "avg_batch_size": sum(stat["batch_size"] for stat in self.batch_stats) / len(self.batch_stats),
                "avg_prompt_tokens": sum(stat["prompt_tokens"] for stat in self.batch_stats) / len(self.batch_stats),
                "avg_prefill_latency": sum(stat["prefill_latency"] for stat in self.batch_stats) / len(self.batch_stats),
                "avg_decode_latency": sum(stat["decode_latency"] for stat in self.batch_stats if stat["decode_latency"]) / len([s for s in self.batch_stats if s["decode_latency"]])
            })
        
        return stats
    
    def _generation_loop(self) -> None:
        """Main generation loop for continuous batching."""
        while self.is_running:
            try:
                # Process a batch of requests
                self._process_batch()
                
                # Sleep briefly to avoid tight loop
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in generation loop: {e}", exc_info=True)
                time.sleep(1.0)  # Sleep longer on error
    
    def _process_batch(self) -> None:
        """Process a batch of generation requests."""
        # Collect batch of requests
        batch = self._collect_batch()
        
        if not batch:
            return
        
        # Process new requests (prefill phase)
        new_requests = [req for req in batch if not req.start_time]
        if new_requests:
            self._process_new_requests(new_requests)
        
        # Process active requests (decode phase)
        active_requests = [req for req in batch if req.start_time and not req.is_completed]
        if active_requests:
            self._process_active_requests(active_requests)
    
    def _collect_batch(self) -> List[GenerationRequest]:
        """
        Collect a batch of requests to process.
        
        Returns:
            List of requests to process
        """
        # Start with active requests
        batch = list(self.active_requests.values())
        waiting_tokens = sum(len(req.prompt_tokens) + len(req.generated_tokens) for req in batch)
        
        # Add requests from queue until batch is full
        start_time = time.time()
        
        while (len(batch) < self.max_batch_size and 
               waiting_tokens < self.max_waiting_tokens and 
               len(batch) < self.max_active_requests and
               (time.time() - start_time) < self.batch_timeout):
            try:
                # Get request from queue (non-blocking)
                request = self.request_queue.get_nowait()
                
                # Initialize KV cache for this request
                if request.request_id not in self.kv_caches:
                    self.kv_caches[request.request_id] = []
                
                # Add to active requests
                self.active_requests[request.request_id] = request
                
                # Add to batch
                batch.append(request)
                waiting_tokens += len(request.prompt_tokens)
                
                # Mark as processed
                self.request_queue.task_done()
            except queue.Empty:
                # No more requests in queue
                break
        
        return batch
    
    def _process_new_requests(self, requests: List[GenerationRequest]) -> None:
        """
        Process new requests (prefill phase).
        
        Args:
            requests: List of new requests
        """
        # Record start time
        start_time = time.time()
        
        # Mark start time for tracking
        for request in requests:
            request.start_time = time.time()
        
        # Split prompts into manageable chunks if needed
        for i in range(0, len(requests), self.max_batch_size):
            chunk = requests[i:i + self.max_batch_size]
            
            # Get prompt tensors
            input_ids = [torch.tensor(req.prompt_tokens, dtype=torch.long) for req in chunk]
            
            # Pad inputs to same length
            max_length = max(len(ids) for ids in input_ids)
            attention_mask = torch.zeros((len(input_ids), max_length), dtype=torch.long, device=self.device)
            padded_input_ids = torch.zeros((len(input_ids), max_length), dtype=torch.long, device=self.device)
            
            for i, ids in enumerate(input_ids):
                padded_input_ids[i, :len(ids)] = ids
                attention_mask[i, :len(ids)] = 1
            
            # Process prompts in chunks for very long prompts
            for j in range(0, max_length, self.prefill_chunk_size):
                chunk_size = min(self.prefill_chunk_size, max_length - j)
                chunk_input_ids = padded_input_ids[:, j:j+chunk_size].to(self.device)
                chunk_attention_mask = attention_mask[:, j:j+chunk_size].to(self.device)
                
                # Run model for prefill
                with torch.no_grad():
                    if self.precision == "float16":
                        with autocast():
                            outputs = self.model(
                                input_ids=chunk_input_ids,
                                attention_mask=chunk_attention_mask,
                                use_cache=True,
                                return_dict=True
                            )
                    else:
                        outputs = self.model(
                            input_ids=chunk_input_ids,
                            attention_mask=chunk_attention_mask,
                            use_cache=True,
                            return_dict=True
                        )
                
                # Extract KV cache
                past_key_values = outputs.get("past_key_values", None)
                
                # Store KV cache for each request
                for i, request in enumerate(chunk):
                    self.kv_caches[request.request_id] = self._extract_kv_cache(past_key_values, i)
        
        # Record prefill latency
        prefill_latency = time.time() - start_time
        
        # Record batch statistics
        self.batch_stats.append({
            "batch_size": len(requests),
            "prompt_tokens": sum(len(req.prompt_tokens) for req in requests),
            "prefill_latency": prefill_latency,
            "decode_latency": None
        })
        
        logger.debug(f"Processed {len(requests)} new requests with {sum(len(req.prompt_tokens) for req in requests)} tokens in {prefill_latency:.4f}s")
    
    def _process_active_requests(self, requests: List[GenerationRequest]) -> None:
        """
        Process active requests (decode phase).
        
        Args:
            requests: List of active requests
        """
        # Record start time
        start_time = time.time()
        
        # Check if any request is already completed
        active_requests = [req for req in requests if not req.is_completed]
        if not active_requests:
            return
        
        # Acquire generation lock
        with self.generate_lock:
            # Create batched tensors for token generation
            input_ids = [torch.tensor([req.prompt_tokens[-1]] + req.generated_tokens, dtype=torch.long)[-1:] for req in active_requests]
            input_ids = torch.stack(input_ids).to(self.device)
            
            # Get batch indices
            batch_indices = {req.request_id: i for i, req in enumerate(active_requests)}
            
            # Get KV caches
            batch_past_key_values = self._batch_kv_caches([req.request_id for req in active_requests])
            
            # Generate next tokens
            with torch.no_grad():
                if self.precision == "float16":
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=None,  # Not needed with KV cache
                            past_key_values=batch_past_key_values,
                            use_cache=True,
                            return_dict=True
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=None,  # Not needed with KV cache
                        past_key_values=batch_past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
            
            # Get next token logits
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.get("past_key_values", None)
            
            # Process each request in the batch
            for i, request in enumerate(active_requests):
                # Get logits for this request
                request_logits = logits[i].unsqueeze(0)
                
                # Apply temperature
                if request.temperature > 0:
                    request_logits = request_logits / request.temperature
                
                # Apply repetition penalty
                if request.repetition_penalty > 1.0:
                    # Get unique tokens in the sequence
                    input_ids_list = request.prompt_tokens + request.generated_tokens
                    for token_id in set(input_ids_list):
                        # Apply penalty
                        idx = torch.where(request_logits[0] == token_id)
                        request_logits[0, idx] = request_logits[0, idx] / request.repetition_penalty
                
                # Apply top-k filtering
                if request.top_k is not None and request.top_k > 0:
                    indices_to_remove = torch.topk(request_logits, k=request.top_k)[0][-1].item()
                    request_logits[request_logits < indices_to_remove] = -float("inf")
                
                # Apply top-p filtering
                if request.top_p is not None and request.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(request_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > request.top_p
                    
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Get indices of tokens to remove
                    indices_to_remove = torch.zeros_like(request_logits, dtype=torch.bool).scatter_(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    request_logits[indices_to_remove] = -float("inf")
                
                # Sample or greedy decoding
                if request.do_sample:
                    probs = F.softmax(request_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(request_logits, dim=-1).item()
                
                # Add token to generated tokens
                request.generated_tokens.append(next_token)
                self.total_tokens_generated += 1
                
                # Update KV cache
                if past_key_values is not None:
                    self.kv_caches[request.request_id] = self._extract_kv_cache(past_key_values, i)
                
                # Check stop criteria
                if self._check_stop_criteria(request):
                    self._complete_request(request)
        
        # Record decode latency
        decode_latency = time.time() - start_time
        
        # Record batch statistics
        self.batch_stats.append({
            "batch_size": len(active_requests),
            "prompt_tokens": 0,  # No prompt tokens in decode phase
            "prefill_latency": 0,  # No prefill in decode phase
            "decode_latency": decode_latency
        })
        
        # Keep only the last 100 stats to avoid memory growth
        if len(self.batch_stats) > 100:
            self.batch_stats = self.batch_stats[-100:]
        
        logger.debug(f"Processed {len(active_requests)} active requests in {decode_latency:.4f}s")
    
    def _check_stop_criteria(self, request: GenerationRequest) -> bool:
        """
        Check if generation should stop for a request.
        
        Args:
            request: Request to check
            
        Returns:
            Whether to stop generation
        """
        # Check max tokens
        if len(request.generated_tokens) >= request.max_new_tokens:
            return True
        
        # Check for EOS token
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and request.generated_tokens and request.generated_tokens[-1] == eos_token_id:
            return True
        
        # Check for stop strings
        if request.stop_strings:
            # Decode the latest part of the generated text
            latest_tokens = request.generated_tokens[-32:]  # Last 32 tokens should be enough for most stop strings
            latest_text = self.tokenizer.decode(latest_tokens)
            
            # Check each stop string
            for stop_string in request.stop_strings:
                if stop_string in latest_text:
                    return True
        
        return False
    
    def _complete_request(self, request: GenerationRequest) -> None:
        """
        Mark a request as completed.
        
        Args:
            request: Request to complete
        """
        # Set completion time
        request.end_time = time.time()
        request.is_completed = True
        
        # Remove from active requests
        if request.request_id in self.active_requests:
            del self.active_requests[request.request_id]
        
        # Add to completed requests
        self.completed_requests[request.request_id] = request
        
        # Free KV cache
        if request.request_id in self.kv_caches:
            del self.kv_caches[request.request_id]
        
        logger.debug(f"Completed request {request.request_id}: {len(request.generated_tokens)} tokens in {request.elapsed_time():.4f}s")
    
    def _extract_kv_cache(self, past_key_values: Tuple, batch_idx: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract KV cache for a specific batch item.
        
        Args:
            past_key_values: Past key values from model output
            batch_idx: Batch index to extract
            
        Returns:
            KV cache for the specified batch item
        """
        if past_key_values is None:
            return []
        
        # Extract KV cache for the specified batch index
        return [(layer_past[0][batch_idx:batch_idx+1], layer_past[1][batch_idx:batch_idx+1]) 
                for layer_past in past_key_values]
    
    def _batch_kv_caches(self, request_ids: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Batch KV caches for multiple requests.
        
        Args:
            request_ids: List of request IDs
            
        Returns:
            Batched KV cache
        """
        # Check if we have KV caches
        if not all(req_id in self.kv_caches for req_id in request_ids):
            return None
        
        # Get individual KV caches
        kv_caches = [self.kv_caches[req_id] for req_id in request_ids]
        
        # Check if all caches have the same structure
        if not all(len(cache) == len(kv_caches[0]) for cache in kv_caches):
            return None
        
        # Batch KV caches
        batched_kv_cache = []
        
        for layer_idx in range(len(kv_caches[0])):
            # Get key and value tensors for this layer
            keys = [cache[layer_idx][0] for cache in kv_caches]
            values = [cache[layer_idx][1] for cache in kv_caches]
            
            # Concatenate along batch dimension
            batched_key = torch.cat(keys, dim=0)
            batched_value = torch.cat(values, dim=0)
            
            batched_kv_cache.append((batched_key, batched_value))
        
        return batched_kv_cache


class KVCacheManager:
    """
    Manager for KV caches to optimize memory usage.
    
    This class manages the KV cache for efficient token generation,
    implementing strategies like cache pruning and disk offloading.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_cache_size: Optional[int] = None,  # Maximum KV cache size in tokens
        disk_offload: bool = False,  # Whether to offload to disk
        disk_offload_path: str = "./kv_cache_offload",  # Path for disk offloaded caches
        pruning_strategy: str = "sliding_window",  # "sliding_window" or "token_drop"
        window_size: int = 2048,  # Size of sliding window
        token_drop_rate: float = 0.5  # Rate of tokens to drop when pruning
    ):
        """
        Initialize KV cache manager.
        
        Args:
            model: Language model
            max_cache_size: Maximum KV cache size in tokens
            disk_offload: Whether to offload KV cache to disk
            disk_offload_path: Path for disk offloaded caches
            pruning_strategy: Strategy for pruning KV cache
            window_size: Size of sliding window for pruning
            token_drop_rate: Rate of tokens to drop when pruning
        """
        self.model = model
        self.max_cache_size = max_cache_size
        self.disk_offload = disk_offload
        self.disk_offload_path = disk_offload_path
        self.pruning_strategy = pruning_strategy
        self.window_size = window_size
        self.token_drop_rate = token_drop_rate
        
        # Create offload directory if needed
        if self.disk_offload:
            os.makedirs(self.disk_offload_path, exist_ok=True)
        
        # Track total cache usage (in tokens)
        self.total_cache_size = 0
        
        # Track cache usage per request
        self.cache_sizes: Dict[str, int] = {}
        
        # Track whether caches are offloaded
        self.offloaded_caches: Dict[str, bool] = {}
        
        logger.info(f"Initialized KV cache manager with max cache size: {max_cache_size or 'unlimited'}")
    
    def add_to_cache(
        self,
        request_id: str,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        current_token_count: int
    ) -> None:
        """
        Add KV cache for a request.
        
        Args:
            request_id: ID of the request
            kv_cache: KV cache to add
            current_token_count: Current token count for the request
        """
        # Update cache size tracking
        old_size = self.cache_sizes.get(request_id, 0)
        self.cache_sizes[request_id] = current_token_count
        
        # Update total cache size
        self.total_cache_size = self.total_cache_size - old_size + current_token_count
        
        # Check if we need to prune caches
        if self.max_cache_size is not None and self.total_cache_size > self.max_cache_size:
            self._prune_caches()
    
    def remove_from_cache(self, request_id: str) -> None:
        """
        Remove KV cache for a request.
        
        Args:
            request_id: ID of the request
        """
        # Update total cache size
        self.total_cache_size -= self.cache_sizes.get(request_id, 0)
        
        # Remove cache size tracking
        if request_id in self.cache_sizes:
            del self.cache_sizes[request_id]
        
        # Remove offload tracking
        if request_id in self.offloaded_caches:
            # Remove offloaded file if exists
            if self.disk_offload and os.path.exists(self._get_offload_path(request_id)):
                try:
                    os.remove(self._get_offload_path(request_id))
                except Exception as e:
                    logger.warning(f"Failed to remove offloaded cache for {request_id}: {e}")
            
            del self.offloaded_caches[request_id]
    
    def _prune_caches(self) -> None:
        """Prune KV caches to reduce memory usage."""
        if self.pruning_strategy == "sliding_window":
            # Use sliding window approach to limit context
            self._apply_sliding_window()
        else:  # token_drop
            # Drop tokens to reduce context
            self._apply_token_dropping()
    
    def _apply_sliding_window(self) -> None:
        """Apply sliding window approach to KV caches."""
        # This would involve truncating the KV cache to only keep the last N tokens
        # Requires specific model implementation support
        logger.debug(f"Applied sliding window pruning to KV caches: {self.window_size} tokens kept")
    
    def _apply_token_dropping(self) -> None:
        """Apply token dropping to KV caches."""
        # This would involve keeping only every Nth token to reduce context size
        # Requires specific model implementation support
        logger.debug(f"Applied token dropping to KV caches: keeping {1.0 - self.token_drop_rate:.0%} of tokens")
    
    def _get_offload_path(self, request_id: str) -> str:
        """
        Get path for offloaded KV cache.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Path for offloaded KV cache
        """
        return os.path.join(self.disk_offload_path, f"kv_cache_{request_id}.pt")


class SpeculativeDecoder:
    """
    Implementation of speculative decoding for efficient text generation.
    
    This class implements speculative decoding, which uses a smaller model
    to generate candidate tokens that are then verified by the larger model.
    """
    
    def __init__(
        self,
        main_model: nn.Module,
        draft_model: nn.Module,
        tokenizer: Any,
        num_speculative_tokens: int = 4,
        device: str = "cuda",
        precision: str = "float16"
    ):
        """
        Initialize speculative decoder.
        
        Args:
            main_model: Main (larger) language model
            draft_model: Draft (smaller) language model
            tokenizer: Tokenizer for the models
            num_speculative_tokens: Number of tokens to generate speculatively
            device: Device to run models on
            precision: Precision for inference
        """
        self.main_model = main_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        self.device = device
        self.precision = precision
        
        # Ensure models are in evaluation mode
        self.main_model.eval()
        self.draft_model.eval()
        
        logger.info(f"Initialized speculative decoder with {num_speculative_tokens} speculative tokens")
    
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = False
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Text prompt or token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling
            
        Returns:
            Tuple of (generated token IDs, generation statistics)
        """
        # Encode prompt if needed
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = prompt
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Initialize stats
        stats = {
            "num_draft_tokens": 0,
            "num_accepted_tokens": 0,
            "num_rejected_tokens": 0,
            "acceptance_rate": 0.0,
            "speedup": 1.0
        }
        
        # Set up for mixed precision
        use_mixed_precision = self.precision == "float16"
        
        # Generate text
        generated_ids = list(input_ids)
        tokens_generated = 0
        
        # Autoregressive decoding
        while tokens_generated < max_new_tokens:
            # Get current input tensor for draft model
            current_input = torch.tensor([generated_ids], dtype=torch.long).to(self.device)
            
            # Generate draft tokens
            draft_tokens = []
            draft_kv_cache = None
            
            # Execute draft model with mixed precision if enabled
            with torch.no_grad():
                if use_mixed_precision:
                    with autocast():
                        # Predict first draft token
                        draft_outputs = self.draft_model(
                            input_ids=current_input,
                            use_cache=True,
                            return_dict=True
                        )
                else:
                    # Predict first draft token
                    draft_outputs = self.draft_model(
                        input_ids=current_input,
                        use_cache=True,
                        return_dict=True
                    )
            
            # Get draft token
            draft_logits = draft_outputs.logits[:, -1, :]
            
            # Apply temperature, top-k, top-p, etc. to draft logits
            processed_logits = self._process_logits(
                draft_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                input_ids=generated_ids
            )
            
            # Get next token
            next_token = processed_logits.argmax(dim=-1).item() if not do_sample else torch.multinomial(
                F.softmax(processed_logits, dim=-1), num_samples=1
            ).item()
            
            # Add to draft tokens
            draft_tokens.append(next_token)
            draft_kv_cache = draft_outputs.past_key_values
            
            # Generate remaining draft tokens
            for _ in range(1, self.num_speculative_tokens):
                # Create tensor for next token
                next_input = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                
                # Execute draft model with mixed precision if enabled
                with torch.no_grad():
                    if use_mixed_precision:
                        with autocast():
                            # Predict next draft token using KV cache
                            draft_outputs = self.draft_model(
                                input_ids=next_input,
                                past_key_values=draft_kv_cache,
                                use_cache=True,
                                return_dict=True
                            )
                    else:
                        # Predict next draft token using KV cache
                        draft_outputs = self.draft_model(
                            input_ids=next_input,
                            past_key_values=draft_kv_cache,
                            use_cache=True,
                            return_dict=True
                        )
                
                # Get draft token
                draft_logits = draft_outputs.logits[:, -1, :]
                
                # Apply temperature, top-k, top-p, etc. to draft logits
                processed_logits = self._process_logits(
                    draft_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    input_ids=generated_ids + draft_tokens
                )
                
                # Get next token
                next_token = processed_logits.argmax(dim=-1).item() if not do_sample else torch.multinomial(
                    F.softmax(processed_logits, dim=-1), num_samples=1
                ).item()
                
                # Add to draft tokens
                draft_tokens.append(next_token)
                draft_kv_cache = draft_outputs.past_key_values
            
            # Update stats
            stats["num_draft_tokens"] += len(draft_tokens)
            
            # Verify draft tokens with main model
            accepted_tokens = self._verify_tokens(
                generated_ids, 
                draft_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            )
            
            # Update stats
            stats["num_accepted_tokens"] += len(accepted_tokens)
            stats["num_rejected_tokens"] += len(draft_tokens) - len(accepted_tokens)
            
            # Add accepted tokens to generated tokens
            generated_ids.extend(accepted_tokens)
            tokens_generated += len(accepted_tokens)
            
            # If we rejected all tokens or hit max_new_tokens, generate a new token from main model
            if not accepted_tokens or tokens_generated >= max_new_tokens:
                if tokens_generated < max_new_tokens:
                    # Generate a single token from main model
                    current_input = torch.tensor([generated_ids], dtype=torch.long).to(self.device)
                    
                    # Execute main model with mixed precision if enabled
                    with torch.no_grad():
                        if use_mixed_precision:
                            with autocast():
                                main_outputs = self.main_model(
                                    input_ids=current_input,
                                    use_cache=True,
                                    return_dict=True
                                )
                        else:
                            main_outputs = self.main_model(
                                input_ids=current_input,
                                use_cache=True,
                                return_dict=True
                            )
                    
                    # Get main model token
                    main_logits = main_outputs.logits[:, -1, :]
                    
                    # Apply temperature, top-k, top-p, etc. to main logits
                    processed_logits = self._process_logits(
                        main_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                        input_ids=generated_ids
                    )
                    
                    # Get next token
                    next_token = processed_logits.argmax(dim=-1).item() if not do_sample else torch.multinomial(
                        F.softmax(processed_logits, dim=-1), num_samples=1
                    ).item()
                    
                    # Add to generated tokens
                    generated_ids.append(next_token)
                    tokens_generated += 1
            
            # Check for end of sequence token
            if generated_ids[-1] == getattr(self.tokenizer, "eos_token_id", None):
                break
        
        # Calculate final stats
        if stats["num_draft_tokens"] > 0:
            stats["acceptance_rate"] = stats["num_accepted_tokens"] / stats["num_draft_tokens"]
        
        # Calculate speedup (tokens generated per main model calls)
        main_model_calls = tokens_generated - stats["num_accepted_tokens"] + 1  # +1 for initial validation
        if main_model_calls > 0:
            stats["speedup"] = tokens_generated / main_model_calls
        
        return generated_ids, stats
    
    def _process_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        input_ids: List[int] = None
    ) -> torch.Tensor:
        """
        Process logits by applying temperature, top-k, top-p, etc.
        
        Args:
            logits: Token logits
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling
            input_ids: Input IDs for repetition penalty
            
        Returns:
            Processed logits
        """
        processed_logits = logits.clone()
        
        # Apply temperature
        if temperature > 0:
            processed_logits = processed_logits / temperature
        
        # Apply repetition penalty
        if repetition_penalty > 1.0 and input_ids:
            for token_id in set(input_ids):
                idx = torch.where(processed_logits[0] == token_id)
                processed_logits[0, idx] = processed_logits[0, idx] / repetition_penalty
        
        # Apply top-k
        if top_k is not None and top_k > 0:
            indices_to_remove = torch.topk(processed_logits, k=top_k)[0][-1].item()
            processed_logits[processed_logits < indices_to_remove] = -float("inf")
        
        # Apply top-p
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(processed_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Get indices of tokens to remove
            indices_to_remove = torch.zeros_like(processed_logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            processed_logits[indices_to_remove] = -float("inf")
        
        return processed_logits
    
    def _verify_tokens(
        self,
        prefix_ids: List[int],
        draft_tokens: List[int],
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = False
    ) -> List[int]:
        """
        Verify draft tokens using the main model.
        
        Args:
            prefix_ids: Prefix token IDs
            draft_tokens: Draft token IDs to verify
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling
            
        Returns:
            List of accepted tokens
        """
        # Create input tensor
        input_tensor = torch.tensor([prefix_ids], dtype=torch.long).to(self.device)
        
        # Set up for mixed precision
        use_mixed_precision = self.precision == "float16"
        
        # Execute main model with mixed precision if enabled
        with torch.no_grad():
            if use_mixed_precision:
                with autocast():
                    main_outputs = self.main_model(
                        input_ids=input_tensor,
                        use_cache=True,
                        return_dict=True
                    )
            else:
                main_outputs = self.main_model(
                    input_ids=input_tensor,
                    use_cache=True,
                    return_dict=True
                )
        
        # Get main model probabilities
        main_logits = main_outputs.logits[:, -1, :]
        past_key_values = main_outputs.past_key_values
        
        # Apply temperature, top-k, top-p, etc. to main logits
        processed_logits = self._process_logits(
            main_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            input_ids=prefix_ids
        )
        
        # Get main model probabilities
        main_probs = F.softmax(processed_logits, dim=-1)
        
        # Verify tokens one by one
        accepted_tokens = []
        current_ids = list(prefix_ids)
        
        for i, draft_token in enumerate(draft_tokens):
            # Get probability of draft token
            draft_prob = main_probs[0, draft_token].item()
            
            # Sample uniform random number
            u = torch.rand(1).item()
            
            # Accept or reject
            if u < draft_prob:
                # Accept draft token
                accepted_tokens.append(draft_token)
                current_ids.append(draft_token)
                
                # Create input tensor for next token
                input_tensor = torch.tensor([[draft_token]], dtype=torch.long).to(self.device)
                
                # Run main model for next token
                with torch.no_grad():
                    if use_mixed_precision:
                        with autocast():
                            main_outputs = self.main_model(
                                input_ids=input_tensor,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True
                            )
                    else:
                        main_outputs = self.main_model(
                            input_ids=input_tensor,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True
                        )
                
                # Update for next iteration
                main_logits = main_outputs.logits[:, -1, :]
                past_key_values = main_outputs.past_key_values
                
                # Apply temperature, top-k, top-p, etc. to main logits
                processed_logits = self._process_logits(
                    main_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    input_ids=current_ids
                )
                
                # Update probabilities
                main_probs = F.softmax(processed_logits, dim=-1)
            else:
                # Reject and sample a new token
                if do_sample:
                    # Sample from main model distribution
                    next_token = torch.multinomial(main_probs, num_samples=1).item()
                else:
                    # Greedy selection
                    next_token = torch.argmax(main_probs, dim=-1).item()
                
                # Add new token
                accepted_tokens.append(next_token)
                
                # Stop verifying more draft tokens
                break
        
        return accepted_tokens
