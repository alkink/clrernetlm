
import sys
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("No pypdf library found")
        sys.exit(1)

reader = PdfReader("/home/alki/projects/clrernetlm/preprints202504.1582.v1.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(text[:5000]) # Print first 5000 chars

