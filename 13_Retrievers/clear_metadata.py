import nbformat

nb = nbformat.read("langchain_retrievers.ipynb", as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, "langchain_retrievers.ipynb")
print("done")
