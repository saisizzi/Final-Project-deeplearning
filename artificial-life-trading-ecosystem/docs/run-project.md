# Run the project

Open **two PowerShell terminals** in the project folder:

```powershell
Set-Location 'C:\Users\Admin\OneDrive\Desktop\Deep-learning\ArtificialLIfe\artificial-life-trading-ecosystem'
```

## Terminal 1: Start the API

```powershell
$env:PYTHONPATH = ".\src"
python -m uvicorn artificial_life_trading_ecosystem.api.app:app --host 127.0.0.1 --port 8000
```

## Terminal 2: Start the frontend

```powershell
Set-Location 'C:\Users\Admin\OneDrive\Desktop\Deep-learning\ArtificialLIfe\artificial-life-trading-ecosystem\frontend'
npm install   # only needed the first time
npm run dev -- --host 127.0.0.1 --port 5173
```

Open the dashboard at <http://127.0.0.1:5173/>.

The API documentation is at <http://127.0.0.1:8000/docs>.

The project uses Python 3.11. Install the Python dependencies from the repository root before starting the API if they are not already installed:

```powershell
python -m pip install -r requirements.txt
```
