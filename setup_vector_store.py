import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

COURSE_FOLDER = "course_files"

# 1. Create vector store
vector_store = client.vector_stores.create(
    name="Analyst Lantern Course Files"
)

print("Vector store created:", vector_store.id)

# 2. Upload files
file_ids = []
for filename in os.listdir(COURSE_FOLDER):
    path = os.path.join(COURSE_FOLDER, filename)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            uploaded = client.files.create(
                file=f,
                purpose="assistants"
            )
            file_ids.append(uploaded.id)
            print("Uploaded:", filename, "->", uploaded.id)

# 3. Add files to vector store
batch = client.vector_stores.file_batches.create(
    vector_store_id=vector_store.id,
    file_ids=file_ids
)

print("File batch created:", batch.id)
print("SAVE THIS VECTOR STORE ID:")
print(vector_store.id)
