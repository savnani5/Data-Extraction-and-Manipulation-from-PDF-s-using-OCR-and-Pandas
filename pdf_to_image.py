
## This code saves the pdf file to a set of images

import fitz  # Library name pyMupdf

def image_conversion(pdf_file):

    doc = fitz.open(pdf_file) #lib name: pymupdf
    zoom_x = 4.0  # horizontal zoom
    zomm_y = 4.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zomm_y)  # zoom factor 2 in each dimension

    for i in range(len(doc)):
        page = doc.loadPage(i) #number of page
        pix = page.getPixmap(matrix = mat)
        output = f"ocr_results/images/{i}.png"
        pix.writePNG(output)

    return len(doc)
