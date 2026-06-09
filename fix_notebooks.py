import nbformat as nbf

files = {
    '01_short_cantilever.ipynb': ('Short Cantilever', 'short_cantilever_optimized.png'),
    '02_mbb_beam.ipynb': ('MBB Beam', 'mbb_beam_optimized.png'),
    '03_l_shape_bracket.ipynb': ('L-Shape Bracket', 'l_shape_bracket_optimized.png')
}

for fname, (title, img) in files.items():
    with open(f"docs/{fname}", "r") as f:
        nb = nbf.read(f, as_version=4)
    
    # Remove previous Markdown cells added by fix_notebooks if any
    # They are at the very beginning and very end
    if len(nb.cells) > 0 and nb.cells[0].cell_type == 'markdown' and nb.cells[0].source.startswith('# '):
        nb.cells.pop(0)
    if len(nb.cells) > 0 and nb.cells[-1].cell_type == 'markdown' and nb.cells[-1].source.startswith('### Optimized Result'):
        nb.cells.pop(-1)
        
    # Prepend a markdown cell
    md_top = nbf.v4.new_markdown_cell(f"# {title}\n\nThis is an interactive Jupyter Notebook implementation. The optimization runs for 50 iterations to ensure proper convergence.")
    nb.cells.insert(0, md_top)
    
    # Append a markdown cell with the image
    md_bot = nbf.v4.new_markdown_cell(f"### Optimized Result\n\n![Optimized Design](_static/{img})")
    nb.cells.append(md_bot)
    
    with open(f"docs/{fname}", "w") as f:
        nbf.write(nb, f)
