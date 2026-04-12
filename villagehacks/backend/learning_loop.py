import redis

class SystemMemory:
    """
    The Learning Loop. This connects to Redis to store phonetic corrections 
    and track the frequency of clinic vocabulary over time.
    """
    def __init__(self, host='localhost', port=6379, db=0):
        # decode_responses=True automatically converts Redis bytes to Python strings
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            # Ping it to make sure the connection is alive
            self.redis_client.ping()
        except redis.ConnectionError:
            print("⚠️ ERROR: Could not connect to Redis. Is the Redis server running?")
            self.redis_client = None

        # The keys we will use to store our data in Redis
        self.map_key = "carecaller:phonetic_map"
        self.freq_key = "carecaller:keyterms"

    def record_correction(self, raw_word: str, verified_word: str):
        """
        Takes a bad transcription and the verified fix, and commits it to memory.
        """
        if not self.redis_client:
            return
            
        # Don't learn "[UNVERIFIED]" or empty words
        if not raw_word or not verified_word or "UNVERIFIED" in verified_word:
            return
            
        raw_word = raw_word.lower()
        verified_word = verified_word.lower()

        # 1. Save the mapping (e.g., metoformin -> metformin)
        self.redis_client.hset(self.map_key, raw_word, verified_word)

        # 2. Increment the leaderboard score for the verified drug
        self.redis_client.zincrby(self.freq_key, 1.0, verified_word)
        print(f"🧠 Learned: '{raw_word}' -> '{verified_word}'")

    def get_known_correction(self, raw_word: str) -> str | None:
        """
        Checks if the system already knows how to fix this word.
        If it does, we can skip the expensive Tavily API call!
        """
        if not self.redis_client:
            return None
        return self.redis_client.hget(self.map_key, raw_word.lower())

    def get_clinic_vocabulary(self, limit: int = 50) -> list[str]:
        """
        Returns the top N most frequently used medical terms at this clinic.
        Person A will call this to inject vocabulary into the STT engine.
        """
        if not self.redis_client:
            return []
        # ZREVRANGE returns the items with the highest scores first
        return self.redis_client.zrevrange(self.freq_key, 0, limit - 1)

    def get_health_stats(self) -> dict:
        """
        Returns stats for the frontend dashboard so judges can literally 
        watch the system get smarter in real time.
        """
        if not self.redis_client:
            return {"status": "disconnected"}
            
        return {
            "status": "connected",
            "phonetic_map_size": self.redis_client.hlen(self.map_key),
            "keyterm_count": self.redis_client.zcard(self.freq_key)
        }

# --- Quick Test ---
if __name__ == "__main__":
    memory = SystemMemory()
    
    if memory.redis_client:
        print("\n--- Testing Learning Loop ---")
        # 1. Simulate learning from the previous pipeline run
        memory.record_correction("metoformin", "metformin")
        memory.record_correction("hydrocloro", "hydrochlorothiazide")
        
        # Simulate seeing metformin again
        memory.record_correction("metformn", "metformin")
        
        # 2. Check the leaderboard
        top_terms = memory.get_clinic_vocabulary(limit=5)
        print(f"\nTop Clinic Terms: {top_terms}")
        
        # 3. Check the memory
        known_fix = memory.get_known_correction("metoformin")
        print(f"Known fix for 'metoformin': {known_fix}")
        
        # 4. Check the stats for the Judges Dashboard
        print(f"\nLive Stats: {memory.get_health_stats()}")