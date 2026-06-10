import nbformat

nb = nbformat.read("YouTubeChatbot.ipynb", as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, "YouTubeChatbot.ipynb")
print("done")
