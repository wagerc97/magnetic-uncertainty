"""
Convert Jupyter notebook to PDF
"""
import nbconvert
import nbformat

def notebook_to_pdf(notebook_path, pdf_path):
    """
    Convert Jupyter notebook to PDF

    Parameters:
    notebook_path (str): Path to the Jupyter notebook file (.ipynb)
    pdf_path (str): Path to save the generated PDF file (.pdf)
    """

    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Create a PDF exporter
    pdf_exporter = nbconvert.PDFExporter()

    # Export the notebook to PDF
    pdf_data, _ = pdf_exporter.from_notebook_node(notebook)

    # Save the PDF data to a file
    with open(pdf_path, 'wb') as f:
        f.write(pdf_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert Jupyter notebook to PDF')
    parser.add_argument('notebook_path', type=str, help='Path to the Jupyter notebook file (.ipynb)')
    parser.add_argument('pdf_path', type=str, help='Path to save the generated PDF file (.pdf)')

    args = parser.parse_args()

    notebook_to_pdf(args.notebook_path, args.pdf_path)
