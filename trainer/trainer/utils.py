"""Utility functions for trainer modules."""
import os


def init_debugpy_if_enabled():
    """Initialize debugpy for multi-rank debugging if ENABLE_DEBUGPY environment variable is set.
    
    Each rank will listen on port 5678 + local_rank and wait for debugger attachment.
    - Rank 0: port 5678
    - Rank 1: port 5679
    - Rank N: port 5678 + N
    """
    if os.environ.get("ENABLE_DEBUGPY", "0") == "1":
        import debugpy
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        debug_port = 5678 + local_rank
        
        debugpy.listen(("0.0.0.0", debug_port))
        print(f"[Rank {local_rank}] Waiting for debugger on port {debug_port}...")
        debugpy.wait_for_client()
        print(f"[Rank {local_rank}] Debugger attached! Starting training...")
