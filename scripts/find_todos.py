with open("tcc_diego/arquivo_2021.tex") as f:
    for i, line in enumerate(f):
        stripped = line.strip()
        if stripped.startswith('%'):
            continue
        if r'\todo' in line:
            print(f"Line {i+1}: {stripped}")
