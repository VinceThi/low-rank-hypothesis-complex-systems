# Antoine Allard

# See https://gist.github.com/antoineallard/763a5773b9c5cd248a7faebcace200dd

from pathlib import Path

graphml_filename = 'path to file'

# Gets the text from the GraphML file.
graphml_content = Path(graphml_filename).read_text()
# What substring should be replaced by what.
strmap = {
    'attr.type="vector_string"': 'attr.type="string"',
    'attr.type="vector_float"': 'attr.type="string"'
    # more types could be added here
}

# Does the subtitution.
for oldstr, newstr in strmap.items():
    graphml_content = graphml_content.replace(oldstr, newstr)

# The standardized GraphML content can then be directly loaded in NetworkX
# import networkx as nx
# graph = nx.parse_graphml(graphml_content)

# or it can be saved to file
with open(graphml_filename.replace('.xml', '_std.xml').replace('.graphml',
                                                               '_std.graphml'),
          'w') as f:
    f.write(graphml_content)
