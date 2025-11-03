# Virtual Environment Kernel Setup

When creating a new uv `.venv` for Jupyter notebooks:

1. `uv venv .venv`
2. Install packages
3. Register as kernel: 
   ```bash
   .venv/bin/python -m ipykernel install --user --name <env-name> --display-name "<Display Name>"
   ```
4. Use in notebook by selecting kernel from dropdown in VS Code

## Example
```bash
cd "AMATH 390"
uv venv .venv
uv pip install numpy matplotlib ipykernel
.venv/bin/python -m ipykernel install --user --name amath390 --display-name "AMATH390 (.venv)"
```