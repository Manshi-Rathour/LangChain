import nbformat

# nb = nbformat.read("tools_in_langchain.ipynb", as_version=4)
# nb = nbformat.read("tool_calling_basics.ipynb", as_version=4)
nb = nbformat.read("Currency_Conversion.ipynb", as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

# nbformat.write(nb, "tools_in_langchain.ipynb")
# nbformat.write(nb, "tool_calling_basics.ipynb")
nbformat.write(nb, "Currency_Conversion.ipynb")
print("done")
