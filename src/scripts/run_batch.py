from src.utils.batch_processor import BatchProcessor
# Initialize processor
processor = BatchProcessor()

# Process all documents
summary = processor.process_directory(
    directory="data/documents",
    custom_metadata={
        "organization": "research_dept",
        "access_level": "internal"
    },
    recursive=True,
    file_types=['.pdf', '.txt', '.docx']
)

# Make documents searchable immediately
processor.refresh_index()