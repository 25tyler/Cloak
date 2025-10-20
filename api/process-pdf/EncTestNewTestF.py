import sys
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def remap_text(text: str) -> str:
    SPECIAL_MAP = {" ": "r", "\x00": "\n"}  # map null byte to newline
    UPPER_MAP = { 
        "R":"A","F":"B","M":"C","S":"D","E":"E","H":"F","D":"G","G":"H","N":"I","A":"J",
        "K":"K","C":"L","Y":"M","U":"N","L":"O","P":"P","X":"Q","V":"R","I":"S","Q":"T",
        "T":"U","O":"V","W":"W","B":"X","Z":"Y","J":"Z"
    }
    LOWER_MAP = {
        "r":"a","f":"b","m":"c","s":"d","e":"e","h":"f","d":"g","g":"h",
        "n":"i","a":"j","k":"k","c":"l","y":"m","u":"n","l":"o","p":"p",
        "x":"q","v":" ","i":"s","q":"t","t":"u","o":"v","w":"w",
        "b":"x","z":"y","j":"z"
    }

    out = []
    for ch in text:
        if ch in SPECIAL_MAP:
            out.append(SPECIAL_MAP[ch])
        elif "A" <= ch <= "Z":
            out.append(UPPER_MAP.get(ch, ch))
        elif "a" <= ch <= "z":
            out.append(LOWER_MAP.get(ch, ch))
        else:
            out.append(ch)
    
    return "".join(out)

def process_pdf(input_pdf: str, font_path: str, output_pdf: str):
    # Register the custom font
    pdfmetrics.registerFont(TTFont("CustomFont", font_path))

    # Open input PDF
    doc = fitz.open(input_pdf)

    # Create new PDF with reportlab
    c = canvas.Canvas(output_pdf, pagesize=letter)
    c.setFont("CustomFont", 12)

    for page in doc:
        text = page.get_text("text")
        mapped_text = remap_text(text)

        # Use TextObject so newlines are honored
        textobj = c.beginText(72, 720)  # start at margin
        for line in mapped_text.splitlines():
            textobj.textLine(line)
        c.drawText(textobj)
        c.showPage()

    c.save()
    print(f"Saved remapped PDF to {output_pdf}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 rscencrypt.py input.pdf font.ttf output.pdf")
        sys.exit(1)

    input_pdf = sys.argv[1]
    font_path = sys.argv[2]
    output_pdf = sys.argv[3]

    process_pdf(input_pdf, font_path, output_pdf)
