import nbformat

nb = nbformat.read("agents_in_langchain.ipynb", as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, "agents_in_langchain.ipynb")
print("done")
