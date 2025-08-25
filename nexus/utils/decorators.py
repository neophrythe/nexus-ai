"""Utility decorators for error handling, performance monitoring, and validation"""

import functools
import asyncio
import time
import threading
from typing import Any, Callable, Optional, Type, Union, Dict, List
from collections import defaultdict
import inspect
import structlog

from nexus.core.exceptions import TimeoutError as NexusTimeoutError, ValidationError, handle_exception

logger = structlog.get_logger()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0, 
                    exceptions: tuple = (Exception,)):
    """Retry function on failure with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                current_delay = delay
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt == max_retries:
                            logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                            raise
                        
                        logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {current_delay}s: {e}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                
                raise last_exception
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                current_delay = delay
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt == max_retries:
                            logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                            raise
                        
                        logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {current_delay}s: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                
                raise last_exception
            return sync_wrapper
    return decorator


def timeout(seconds: float):
    """Add timeout to function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {seconds}s")
                raise NexusTimeoutError(func.__name__, seconds)
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator


def async_timeout(seconds: float):
    """Add timeout to async function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Async function {func.__name__} timed out after {seconds}s")
                raise NexusTimeoutError(func.__name__, seconds)
        return wrapper
    return decorator


def measure_time(log_slow: float = None):
    """Measure function execution time"""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if log_slow and duration > log_slow:
                        logger.warning(f"Slow execution: {func.__name__} took {duration:.3f}s")
                    else:
                        logger.debug(f"Function {func.__name__} took {duration:.3f}s")
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if log_slow and duration > log_slow:
                        logger.warning(f"Slow execution: {func.__name__} took {duration:.3f}s")
                    else:
                        logger.debug(f"Function {func.__name__} took {duration:.3f}s")
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
                    raise
            return sync_wrapper
    return decorator


def rate_limit(calls_per_second: float):
    """Rate limit function calls"""
    min_interval = 1.0 / calls_per_second
    
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]
        lock = threading.Lock()
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with asyncio.Lock():
                    now = time.time()
                    elapsed = now - last_called[0]
                    
                    if elapsed < min_interval:
                        wait_time = min_interval - elapsed
                        await asyncio.sleep(wait_time)
                    
                    last_called[0] = time.time()
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with lock:
                    now = time.time()
                    elapsed = now - last_called[0]
                    
                    if elapsed < min_interval:
                        wait_time = min_interval - elapsed
                        time.sleep(wait_time)
                    
                    last_called[0] = time.time()
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


class CircuitBreakerState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def circuit_breaker(failure_threshold: int = 5, timeout: float = 60.0, expected_exception: Type[Exception] = Exception):
    """Circuit breaker pattern to prevent cascading failures"""
    def decorator(func: Callable) -> Callable:
        state = [CircuitBreakerState.CLOSED]
        failure_count = [0]
        last_failure_time = [0.0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Check if we should try again after timeout
            if state[0] == CircuitBreakerState.OPEN:
                if now - last_failure_time[0] > timeout:
                    state[0] = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker for {func.__name__} moving to HALF_OPEN state")
                else:
                    logger.warning(f"Circuit breaker for {func.__name__} is OPEN, rejecting call")
                    raise expected_exception(f"Circuit breaker is open for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if state[0] == CircuitBreakerState.HALF_OPEN:
                    state[0] = CircuitBreakerState.CLOSED
                    failure_count[0] = 0
                    logger.info(f"Circuit breaker for {func.__name__} reset to CLOSED state")
                elif state[0] == CircuitBreakerState.CLOSED:
                    failure_count[0] = 0
                
                return result
                
            except expected_exception as e:
                failure_count[0] += 1
                last_failure_time[0] = now
                
                if failure_count[0] >= failure_threshold:
                    state[0] = CircuitBreakerState.OPEN
                    logger.error(f"Circuit breaker for {func.__name__} OPENED after {failure_count[0]} failures")
                
                logger.warning(f"Function {func.__name__} failed ({failure_count[0]}/{failure_threshold}): {e}")
                raise
        
        return wrapper
    return decorator


def validate_types(**type_hints):
    """Validate function arguments against type hints"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_hints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise ValidationError(
                            param_name, 
                            value, 
                            f"Expected {expected_type.__name__}, got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_calls(include_args: bool = False, include_result: bool = False):
    """Log function calls with optional arguments and results"""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                call_info = {"function": func.__name__}
                
                if include_args:
                    call_info["args"] = args
                    call_info["kwargs"] = kwargs
                
                logger.info(f"Calling {func.__name__}", extra=call_info)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    if include_result:
                        logger.info(f"Function {func.__name__} returned", extra={"result": result})
                    else:
                        logger.info(f"Function {func.__name__} completed successfully")
                    
                    return result
                except Exception as e:
                    logger.error(f"Function {func.__name__} raised exception: {e}")
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                call_info = {"function": func.__name__}
                
                if include_args:
                    call_info["args"] = args
                    call_info["kwargs"] = kwargs
                
                logger.info(f"Calling {func.__name__}", extra=call_info)
                
                try:
                    result = func(*args, **kwargs)
                    
                    if include_result:
                        logger.info(f"Function {func.__name__} returned", extra={"result": result})
                    else:
                        logger.info(f"Function {func.__name__} completed successfully")
                    
                    return result
                except Exception as e:
                    logger.error(f"Function {func.__name__} raised exception: {e}")
                    raise
            return sync_wrapper
    return decorator


def cache_result(maxsize: int = 128, ttl: Optional[float] = None):
    """Cache function results with optional TTL"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {} if ttl else None
        access_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check TTL expiration
            if ttl and key in cache_times:
                if time.time() - cache_times[key] > ttl:
                    cache.pop(key, None)
                    cache_times.pop(key, None)
                    if key in access_order:
                        access_order.remove(key)
            
            # Return cached result if available
            if key in cache:
                # Update access order
                if key in access_order:
                    access_order.remove(key)
                access_order.append(key)
                return cache[key]
            
            # Call function and cache result
            result = func(*args, **kwargs)
            
            # Evict oldest if at capacity
            if len(cache) >= maxsize and key not in cache:
                oldest_key = access_order.pop(0)
                cache.pop(oldest_key, None)
                if cache_times:
                    cache_times.pop(oldest_key, None)
            
            cache[key] = result
            if cache_times is not None:
                cache_times[key] = time.time()
            access_order.append(key)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear() or (cache_times.clear() if cache_times else None) or access_order.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize, "ttl": ttl}
        
        return wrapper
    return decorator