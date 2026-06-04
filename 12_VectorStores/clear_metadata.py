import nbformat

nb = nbformat.read("vector_store_chromaDB.ipynb", as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, "vector_store_chromaDB.ipynb")
print("done")
