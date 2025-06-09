from supabase import create_client
from config import settings

class ClientFactory:
    @staticmethod
    def setup_clients():
        """
        Initialize and return all required clients.
        
        Returns:
            tuple: A tuple containing:
                - supabase_client: Supabase client instance
        """
        # Initialize Supabase client
        supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

        return supabase_client 