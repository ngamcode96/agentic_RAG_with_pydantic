from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(supabase_url=os.getenv('SUPABASE_URL'), supabase_key=os.getenv('SUPABASE_API_KEY'))

response = supabase.table("site_pages").select("content").execute()
print(response)

# CREATE TABLE site_pages (
#     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
#     url TEXT,
#     chunk_number INTEGER,
#     title TEXT,
#     summary TEXT,
#     content TEXT,
#     metadata JSONB,
#     embedding VECTOR(1536)
# );