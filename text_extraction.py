
import pymupdf


def extract_txt(content: bytes) -> str:
    return content.decode("utf-8-sig")


def extract_pdf(content: bytes) -> str:
    text: list[str] = []

    with pymupdf.open(stream=content, filetype="pdf") as document:
        if not document.is_pdf or document.needs_pass:
            raise ValueError("Invalid or password-protected PDF")

        for page in document:
            text.append(page.get_text())

    return "\n".join(text)
