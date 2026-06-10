import nbformat

nb = nbformat.read("tools_in_langchain.ipynb", as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, "tools_in_langchain.ipynb")
print("done")
